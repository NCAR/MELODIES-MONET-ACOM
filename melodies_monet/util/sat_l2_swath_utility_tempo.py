# SPDX-License-Identifier: Apache-2.0
#

# read all swath data for the time range
# developed for TEMPO Level2 NO2
#

"""Python utility for TEMPO use."""

import collections
import glob
import logging
import warnings

import numba
import numpy as np
import xarray as xr
import xesmf as xe

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


def calc_grid_corners(ds, lat="latitude", lon="longitude"):
    """Adds latitude and longitude bounds inplace.
    If the grid is rectilinear, it should be quite precise.
    If it is curvilinear, is a rough estimate.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to which the latitude and longitude will be added.
    lat : str
        name of the lat variable.
    lon : str
        name of the lon variable.

    Returns
    -------
    None
    """
    try:
        import cf_xarray as cfxr
    except ImportError:
        raise ImportError("Calculating gridcell bounds requires cf_xarray. Please install")
    corners = ds[[lat, lon]].cf.add_bounds([lat, lon])
    ds["lat_b"] = cfxr.bounds_to_vertices(corners[f"{lat}_bounds"], "bounds", order=None)
    ds["lon_b"] = cfxr.bounds_to_vertices(corners[f"{lon}_bounds"], "bounds", order=None)
    return


def speedup_regridding(dset, variables="all"):
    """Makes modobj latitude and longitude C_contiguous, which speeds up regridding.
    It makes the changes inplace

    Parameters
    ----------
    dset : xr.Dataset
        Dataset containing latitude and longitude

    Returns
    -------
    None
    """
    if variables == "all":
        variables = dset.variables
    for v in variables:
        if not dset[v].values.flags["C_CONTIGUOUS"]:
            dtype = dset[v].dtype
            dset[v] = dset[v].astype(dtype, order="C")


def tempo_interp_mod2swath(obsobj, modobj, method="conservative", weights=None):
    """Interpolate model to satellite swath/swaths

    Parameters
    ----------
    obsobj : xr.Dataset
        satellite with swath data.
    modobj : xr.Dataset
        model data (with no2 col calculated)
    method : str
        Choose regridding method. Can be "conservative", "conservative_normed",
        "bilinear" or "patch". Check xesmf documentation for details.
    weights : str
        Path to the weightfile. If present, the weights won't be calculated again.

    Returns
    -------
    xr.Dataset
        Regridded model data at swath or swaths. If type is xr.Dataset, a single
        swath is returned. If type is collections.OrderedDict, it returns an
        OrderedDict in which each time represents the reference time of the swath.
    """

    mod_at_swathtime = modobj.interp(time=obsobj.time.mean())
    if weights is None:
        regridder = xe.Regridder(
            mod_at_swathtime,
            obsobj,
            method,
            ignore_degenerate=True,
            unmapped_to_nan=True,
        )
        modswath = regridder(mod_at_swathtime)
    else:
        regridder = xe.Regridder(
            mod_at_swathtime,
            obsobj,
            method,
            ignore_degenerate=True,
            unmapped_to_nan=True,
            filename=weights,
            reuse_weights=True,
        )
        modswath = regridder(mod_at_swathtime)

    return modswath


def _calc_dp(obsobj):
    """Calculate delta pressure in satellite layers

    Parameters
    ----------
    obsobj : xr.Dataset
        satellite observations containing pressure (in Pa)

    Returns
    -------
    xr.DataArray
        Pressure difference in layer
    """

    # REMINDER: pressure is higher at lower vertical levels: dp is positive
    # only if defined as lower - higher.
    dp_vals = (
        obsobj["pressure"].isel(swt_level_stagg=slice(None, -1)).values
        - obsobj["pressure"].isel(swt_level_stagg=slice(1, None)).values
    )
    dp = xr.DataArray(
        data=dp_vals,
        dims=("swt_level", "x", "y"),
        coords={
            "lon": (("x", "y"), obsobj["lon"].values),
            "lat": (("x", "y"), obsobj["lat"].values),
        },
        attrs={
            "units": "Pa",
            "description": "Delta pressure in layer",
            "long_name": "delta_p",
        },
    )
    return dp


@numba.jit(nopython=True)
def _interp_vert(orig, target, data):
    """Performs the numpy interpolation. It is separated from other functions
    for the sake of using the numba jit.

    Parameters:
    -----------
    orig : np.ndarray
        Original grid from which to interpolate. The expected dimensions are (z, x, y),
        in that order. The horizontal and time dimensions are expected to be previously
        interpolated. The original pressure levels should be in decreasing order.
    target : np.ndarray
        Target data with vertical grid information. The expected dimensions are (z, x, y),
        in that order. The target pressure layers should be in decreasing order.
    data : np.ndarray
        Data to be interpolated. It should have the same grid (including vertical) and dimensions
        as orig.

    Returns
    -------
    np.ndarray
        Interpolated data
    """
    assert orig.shape == data.shape, "Grid shape does not match data"
    nz, nx, ny = target.shape
    interp = np.zeros((nz, nx, ny))
    for x in range(nx):
        for y in range(ny):
            interp[:, x, y] = np.flip(
                np.interp(
                    np.flip(target[:, x, y]),
                    np.flip(orig[:, x, y]),
                    np.flip(data[:, x, y]),
                )
            )
    return interp


def calc_altitude_from_thickness(dz_m):
    """Calculate layer altitude above ground

    Parameters
    ----------
    data : xr.DataArray
        DataArray containing dz_m

    Returns
    -------
    Model altitude in satellite space
    """
    altitude_interface = xr.zeros_like(dz_m)
    altitude_interface[{"z": 0}] = dz_m[{"z": 0}]
    for lev in altitude_interface["z"][1:]:
        altitude_interface[{"z": lev}] = altitude_interface[{"z": lev - 1}] + dz_m[{"z": lev}]
    altitude_interface.attrs = {
        "description": "Altitude AGL in m at layer interface",
        "units": "m",
        "long_name": "altitude_agl",
    }
    return altitude_interface


def calc_dz_m_from_altitude(altitude):
    """Calculates dz_m from altitude AGL (in m).

    Parameters
    ----------
    altitude : xr.DataArray
        DataArray containing the layer interface altitude AGL at the
        interface.

    Returns
    -------
    xr.DataArray
        DataArray containing the layer thickness (dz_m) in m.
    """
    dz_m = xr.zeros_like(altitude)
    dz_m[{"z": 0}] = altitude[{"z": 0}]
    dz_m[{"z": slice(1, None)}] = (
        altitude[{"z": slice(1, None)}].values - altitude[{"z": slice(0, -1)}].values
    )
    dz_m.attrs = {
        "description": "Layer thickness in m",
        "units": "m",
        "long_name": "layer_thickness",
    }
    return dz_m


def interp_vertical_mod2swath(obsobj, modobj, variables="NO2_col"):
    """Interpolates model vertical layers to TEMPO vertical layers

    Parameters
    ----------
    modobj : xr.Dataset
        Model data (as provided by MONETIO)
    obsobj : xr.Dataset
        TEMPO data (as provided by MONETIO). Must include pressure.
    variables : str | list[str]
        Variables to interpolate.

    Returns
    -------
    xr.Dataset
        Model data (interpolated to TEMPO vertical layers
    """
    assert np.all(modobj["lon"].fillna(0).values == obsobj["lon"].fillna(0).values)
    assert np.all(modobj["lat"].fillna(0).values == obsobj["lat"].fillna(0).values)

    modsatlayers = xr.Dataset()
    p_mid_tempo = (
        obsobj["pressure"].isel(swt_level_stagg=slice(None, -1)).values
        + obsobj["pressure"].isel(swt_level_stagg=slice(1, None)).values
    ) / 2
    p_orig = modobj["pres_pa_mid"].values
    dimensions = ("z", "x", "y")
    coords = {
        "lon": (("x", "y"), modobj["lon"].values),
        "lat": (("x", "y"), modobj["lat"].values),
    }
    for var in list(variables):
        interpolated = _interp_vert(p_orig, p_mid_tempo, modobj[var].values)
        modsatlayers[var] = xr.DataArray(
            data=interpolated, dims=dimensions, coords=coords, attrs=modobj[var].attrs
        )
    modsatlayers["pres_pa_mid"] = xr.DataArray(
        data=p_mid_tempo,
        dims=dimensions,
        coords=coords,
        attrs=modobj["pres_pa_mid"].attrs,
    )
    _interp_description = "Mid layer pressure interpolated to TEMPO mid swt_layer pressures"
    modsatlayers["pres_pa_mid"].attrs["description"] = _interp_description
    return modsatlayers


def calc_partialcolumn(modobj, var="NO2"):
    """Calculates the partial column of a species from its concentration.

    Parameters
    ----------
    modobj : xr.Dataset
        Model data
    var : str
        variable to calculate the partial column from

    Returns
    -------
    xr.DataArray
        DataArray containing the partial column of the species.
    """
    R = 8.314  # m3 * Pa / K / mol
    NA = 6.022e23
    ppbv2molmol = 1e-9
    m2_to_cm2 = 1e4
    fac_units = ppbv2molmol * NA / m2_to_cm2
    partial_col = (
        modobj[var]
        * modobj["pres_pa_mid"]
        * modobj["dz_m"]
        * fac_units
        / (R * modobj["temperature_k"])
    )
    return partial_col


def apply_weights_mod2tempo_no2_hydrostatic(obsobj, modobj, species="NO2"):
    """Apply the scattering weights and air mass factors according to
    Cooper et. al, 2020, doi: https://doi.org/10.5194/acp-20-7231-2020,
    assuming the hydrostatic equation. It does not require temperature
    nor geometric layer thickness.

    Parameters
    ----------
    obsobj : xr.Dataset
        TEMPO data, including pressure and scattering weights
    modobj : xr.Dataset
        Model data, already interpolated to TEMPO grid

    Returns
    -------
    xr.DataArray
        A xr.DataArray containing the NO2 model data after applying
        the air mass factors and scattering weights
    """
    unit_c = 6.022e23 * 9.8 / 1e4  # NA * g / m2_to_cm2
    dp = _calc_dp(obsobj).rename({"swt_level": "z"})
    ppbv2molmol = 1e-9
    tropopause_pressure = obsobj["tropopause_pressure"]
    scattering_weights = obsobj["scattering_weights"].transpose("swt_level", "x", "y")
    scattering_weights = scattering_weights.rename({"swt_level": "z"})
    scattering_weights = scattering_weights.where(modobj["pres_pa_mid"] >= tropopause_pressure)
    modno2 = modobj[species].where(modobj["pres_pa_mid"] >= tropopause_pressure)
    amf_troposphere = obsobj["amf_troposphere"]
    modno2col_trfmd = (dp * scattering_weights * modno2).sum(dim="z") * unit_c * ppbv2molmol
    modno2col_trfmd = modno2col_trfmd.where(modno2.isel(z=0).notnull())
    modno2col_trfmd = modno2col_trfmd / amf_troposphere
    modno2col_trfmd.attrs = {
        "units": "molecules/cm2",
        "description": "model NO2 tropospheric column after applying TEMPO scattering weights and AMF",
        "history": "Created by MELODIES-MONET, apply_weights_mod2tempo_no2_hydrostatic, TEMPO util",
    }
    return modno2col_trfmd.where(np.isfinite(modno2col_trfmd))


def apply_weights_mod2tempo_no2(obsobj, modobj, species="NO2", column_type="tropospheric"):
    """Apply the scattering weights and air mass factors according to
    Cooper et. al, 2020, doi: https://doi.org/10.5194/acp-20-7231-2020

    Parameters
    ----------
    obsobj : xr.Dataset
        TEMPO data, including pressure and scattering weights
    modobj : xr.Dataset
        Model data, already interpolated to TEMPO grid
    column_type : str
        Whether the "tropospheric" or "total" column should be used for the calculation

    Returns
    -------
    xr.DataArray
        A xr.DataArray containing the NO2 model data after applying
        the air mass factors and scattering weights
    """
    partial_col = modobj[f"{species}_col"]

    scattering_weights = obsobj["scattering_weights"].transpose("swt_level", "x", "y")
    scattering_weights = scattering_weights.rename({"swt_level": "z"})
    if column_type == "tropospheric":
        tropopause_pressure = obsobj["tropopause_pressure"] * 100
        scattering_weights = scattering_weights.where(modobj["pres_pa_mid"] >= tropopause_pressure)
        amf_troposphere = obsobj["amf_troposphere"]
    modno2col_trfmd = (scattering_weights * partial_col).sum(dim="z") / amf_troposphere
    modno2col_trfmd = modno2col_trfmd.where(modobj[f"{species}_col"].isel(z=0).notnull())
    modno2col_trfmd.attrs = {
        "units": "molecules/cm2",
        "description": "model NO2 tropospheric column after applying TEMPO scattering weights and AMF",
        "history": "Created by MELODIES-MONET, apply_weights_mod2tempo_no2, TEMPO util",
    }
    return modno2col_trfmd.where(np.isfinite(modno2col_trfmd))


def apply_weights_mod2tempo_hcho_hydrostatic(obsobj, modobj, species="HCHO"):
    """Apply the scattering weights and air mass factors according to
    Cooper et. al, 2020, doi: https://doi.org/10.5194/acp-20-7231-2020,
    assuming the hydrostatic equation. It does not require temperature
    nor geometric layer thickness.

    Parameters
    ----------
    obsobj : xr.Dataset
        TEMPO data, including pressure and scattering weights
    modobj : xr.Dataset
        Model data, already interpolated to TEMPO grid

    Returns
    -------
    xr.DataArray
        A xr.DataArray containing the NO2 model data after applying
        the air mass factors and scattering weights
    """
    unit_c = 6.022e23 * 9.8 / 1e4  # NA * g / m2_to_cm2
    dp = _calc_dp(obsobj).rename({"swt_level": "z"})
    ppbv2molmol = 1e-9
    scattering_weights = obsobj["scattering_weights"].transpose("swt_level", "x", "y")
    scattering_weights = scattering_weights.rename({"swt_level": "z"})
    modhcho = modobj[species]
    amf = obsobj["amf"]
    modhcho_col = (dp * scattering_weights * modhcho).sum(dim="z") * unit_c * ppbv2molmol
    modhcho_col = modhcho_col / amf
    modhcho_col.attrs = {
        "units": "molecules/cm2",
        "description": "model HCHO column after applying TEMPO scattering weights and AMF",
        "history": "Created by MELODIES-MONET, apply_weights_mod2tempo_hcho_hydrostatic, TEMPO util",
    }
    return modhcho_col.where(np.isfinite(modhcho_col))


def apply_weights_mod2tempo_hcho(obsobj, modobj, species="HCHO"):
    """Apply the scattering weights and air mass factors according to
    Cooper et. al, 2020, doi: https://doi.org/10.5194/acp-20-7231-2020

    Parameters
    ----------
    obsobj : xr.Dataset
        TEMPO data, including pressure and scattering weights
    modobj : xr.Dataset
        Model data, already interpolated to TEMPO grid

    Returns
    -------
    xr.DataArray
        A xr.DataArray containing the NO2 model data after applying
        the air mass factors and scattering weights
    """
    partial_col = modobj[f"{species}_col"]

    scattering_weights = obsobj["scattering_weights"].transpose("swt_level", "x", "y")
    scattering_weights = scattering_weights.rename({"swt_level": "z"})
    amf = obsobj["amf"]
    modhcho_col = (scattering_weights * partial_col).sum(dim="z") / amf
    modhcho_col = modhcho_col.where(modobj[f"{species}_col"].isel(z=0).notnull())
    modhcho_col.attrs = {
        "units": "molecules/cm2",
        "description": "model HCHO column after applying TEMPO scattering weights and AMF",
        "history": "Created by MELODIES-MONET, apply_weights_mod2tempo_hcho, TEMPO util",
    }
    return modhcho_col.where(np.isfinite(modhcho_col))


def is_nonpairable(obsobj, k, modobj):
    """Discards inplace granules from obsobj that do not match modobj's
    domain, or granules that are all NaN. If the domain is small,
    it can considerably speed up the regridding process.

    Parameters
    ----------
    obsobj : dict[str, xr.Dataset]
        tempo data
    modobj : xr.Dataset
        model data

    Return
    ------
    collections.OrderedDict[str, xr.Dataset]
    """
    if obsobj[k]["lon"].max() < modobj["longitude"].min():
        return True
    elif obsobj[k]["lon"].min() > modobj["longitude"].max():
        return True
    elif obsobj[k]["lat"].max() < modobj["latitude"].min():
        return True
    elif obsobj[k]["lat"].min() > modobj["latitude"].max():
        return True
    return False


def _regrid_and_apply_weights(
    obsobj, modobj, method="conservative", weights=None, species=["NO2"], tempo_sp="NO2"
):
    """Does the complete process of regridding and
    applying scattering weights. Assumes that obsobj is a Dataset

    Parameters
    ----------
    obsobj : xr.Dataset
        TEMPO observations
    modobj : xr.Dataset
        Model data
    method : str
        Choose regridding method. Can be "conservative", "conservative_normed",
        "bilinear" or "patch". Check xesmf documentation for details.
    weights : str
        Path to the weightfile. If present, the weights won't be calculated again.
    tempo_sp: str
        NO2 or HCHO, to apply the correct Air Mass Factors and scattering weights.

    Returns
    -------
    xr.DataArray
        Model data regridded to the TEMPO grid,
        with the averaging kernel.
    """
    if tempo_sp == "NO2":
        apply_weights = apply_weights_mod2tempo_no2
        apply_weights_hydrostatic = apply_weights_mod2tempo_no2_hydrostatic
    else:
        assert tempo_sp == "HCHO", "TEMPO species must be HCHO or NO2."
        apply_weights = apply_weights_mod2tempo_hcho
        apply_weights_hydrostatic = apply_weights_mod2tempo_hcho_hydrostatic
    if method == "conservative":
        if "lat_b" not in modobj:
            calc_grid_corners(modobj)
        if "lat_b" not in obsobj:
            calc_grid_corners(obsobj, lat="lat", lon="lon")
    modobj_hs = tempo_interp_mod2swath(obsobj, modobj, method=method, weights=weights)
    if "dz_m" in modobj.keys():
        modobj_hs["altitude"] = calc_altitude_from_thickness(modobj_hs["dz_m"])
        modobj_swath = interp_vertical_mod2swath(
            obsobj, modobj_hs, [f"{species[0]}", "altitude", "temperature_k"]
        )
        modobj_swath["dz_m"] = calc_dz_m_from_altitude(modobj_swath["altitude"])
        modobj_swath[f"{species[0]}_col"] = calc_partialcolumn(modobj_swath, var=species[0])
        da_out = apply_weights(obsobj, modobj_swath, species=f"{species[0]}")
    else:
        warnings.warn(
            "There is no dz_m variable, and the partial column"
            + "cannot be directly calculated. Assuming hydrostatic equation."
        )
        modobj_swath = interp_vertical_mod2swath(obsobj, modobj_hs, species)
        da_out = apply_weights_hydrostatic(obsobj, modobj_swath, species=species[0])
    return da_out.where(np.isfinite(da_out))


def regrid_and_apply_weights(
    obsobj,
    modobj,
    pair=True,
    verbose=True,
    method="conservative",
    weights=None,
    species=["NO2"],
    tempo_sp="NO2",
):
    """Does the complete process of regridding
    and applying scattering weights.

    Parameters
    ----------
    obsobj : xr.Dataset | collections.OrderedDict
        TEMPO observations
    modobj : xr.Dataset
        Model output
    pair : boolean
        If True, returns paired data.
    verbose : boolean
        If True, let's the user know when each timestamp is being regridded.
        Only has an effect if the input is an OrderedDict
    method : str
        Choose regridding method. Can be "conservative", "conservative_normed",
        "bilinear" or "patch". Check xesmf documentation for details.
    weights : None | str
        If present, a weightfile (as in "weights") is applied
    discard_useless: boolean
        If True, satellite granules that don't match the model domain are not used.
    tempo_sp: str
        NO2 for the NO2 product, HCHO for the HCHO product

    Returns
    -------
    xr.Dataset | collections.OrderedDict
        Model with regridded data. If obsobj is of type collections.OrderedDict,
        an OrderedDict is returned.
    """

    if tempo_sp == "NO2":
        sat_species_name = "vertical_column_troposphere"
    else:
        assert tempo_sp == "HCHO", "TEMPO species must be HCHO or NO2."
        sat_species_name = "vertical_column"

    if isinstance(obsobj, xr.Dataset):
        regridded = _regrid_and_apply_weights(
            obsobj, modobj, method=method, weights=weights, species=species, tempo_sp=tempo_sp
        )
        output = regridded.to_dataset(name=species[0])
        output.attrs["reference_time_string"] = obsobj.attrs["reference_time_string"]
        output.attrs["final_time_string"] = obsobj["time"][-1].values.astype(str)
        output.attrs["scan_num"] = obsobj.attrs["scan_num"]
        output.attrs["granule_number"] = obsobj.attrs["granule_number"]
        if pair:
            output = xr.merge([output, obsobj[sat_species_name]])
        if "lat" in output.variables:
            output = output.rename({"lat": "latitude", "lon": "longitude"})
        return output
    if isinstance(obsobj, dict):
        output_multiple = {}
        for ref_time in obsobj.keys():
            if is_nonpairable(obsobj, ref_time, modobj):
                warnings.warn(f"{ref_time} granule domain has no overlap with model. Discarding.")
                continue
            if verbose:
                print(f"Regridding {ref_time} and applying AMF and weights")
            output_multiple[ref_time] = _regrid_and_apply_weights(
                obsobj[ref_time],
                modobj,
                method=method,
                weights=weights,
                species=species,
                tempo_sp=tempo_sp,
            ).to_dataset(name=species[0])
            output_multiple[ref_time].attrs["reference_time_string"] = ref_time
            output_multiple[ref_time].attrs["scan_num"] = obsobj[ref_time].attrs["scan_num"]
            output_multiple[ref_time].attrs["granule_number"] = obsobj[ref_time].attrs[
                "granule_number"
            ]
            output_multiple[ref_time].attrs["final_time_string"] = obsobj[ref_time]["time"][
                -1
            ].values.astype(str)
            if pair:
                output_multiple[ref_time] = xr.merge(
                    [
                        output_multiple[ref_time],
                        obsobj[ref_time][sat_species_name],
                    ]
                )
            if "lat" in output_multiple[ref_time].variables:
                output_multiple[ref_time] = output_multiple[ref_time].rename(
                    {"lat": "latitude", "lon": "longitude"}
                )
        return output_multiple
    raise TypeError("Obsobj must be xr.Dataset or dict")


def back_to_modgrid(
    paireddict,
    modobj,
    keys_to_merge="all",
    add_time=True,
    to_netcdf=False,
    path="Regridded_object_XYZ.nc",
    method="bilinear",
    grid_path=None,
):
    """Grids object in sat-space to modgrid. Designed to grid back to modgrid after applying
    the scattering weights and air mass factors. It is designed for a single scan.

    Parameters
    ----------
    paireddict : collections.OrderedDict[str, xr.Dataset]
        An OrderedDict with time_reference strings as keys.
    modobj : xr.Dataset
        A modobj including the modgrid.
    keys_to_merge : str | list[str]
        If 'all', all keys are assumed to be part of the same scan and merged.
        Else, only the keys provided are merged.
    add_time : bool
        If True, add reference time as a coordinate for the scan.
        Can be useful to concatenate later if multiple scans are required.
    to_netcdf : bool
        If True, save a netcdf with the paired data
    path : str
        The base name to save the files if to_netcdf is True. XX will be replaced
        with the scan number and the reference time. If to_netcdf is False, this will
        be ignored.
    method : str
        Method of regridding used by xESMF
    grid_path : str
        If None, defaults to the model grid. Otherwise, the grid in path is used.
        If the method is conservative, lat_b and lon_b are required.

    Returns
    -------
    xr.Dataset
        Dataset with obj2grid regridded to modobj.
    """
    if keys_to_merge == "all":
        ordered_keys = sorted(list(paireddict.keys()))
    else:
        ordered_keys = sorted(list(keys_to_merge))
    concatenated = paireddict[ordered_keys[0]]
    scan_num = concatenated.attrs["scan_num"]
    granules = [concatenated.attrs["granule_number"]]
    ref_times = [concatenated.attrs["reference_time_string"][:-1]]  # Remove unneeded Z
    if len(paireddict) > 1:
        for k in ordered_keys[1:]:
            ds_to_add = paireddict[k]
            if ds_to_add.attrs["scan_num"] != scan_num:
                raise ValueError(
                    "back_to_modgrid is prepared to work with data of a single scan. "
                    + f"However, {ordered_keys[0]} is from scan {scan_num} and "
                    + f"{k} if from scan {ds_to_add.attrs['scan_num']}."
                )
            concatenated = xr.concat([concatenated, paireddict[k]], dim="x")
            granules.append(paireddict[k].attrs["granule_number"])
            ref_times.append(paireddict[k].attrs["reference_time_string"][:-1])

    end_time = np.array(
        paireddict[ordered_keys[-1]].attrs["final_time_string"], dtype="datetime64[ns]"
    )
    if grid_path is not None:
        grid = xr.open_dataset(grid_path)
        regridder = xe.Regridder(concatenated, grid, method=method, unmapped_to_nan=True)
    else:
        regridder = xe.Regridder(concatenated, modobj, method=method, unmapped_to_nan=True)
    out_regridded = regridder(concatenated)
    for v in out_regridded.variables:
        if v in concatenated.variables:
            out_regridded[v].attrs = concatenated[v].attrs
        else:
            warnings.warn(f"Variable {v} not found in mod2grid nor obs2grid. Continuing.")
    out_regridded.attrs["reference_time_string"] = ref_times
    out_regridded.attrs["granules"] = np.array(granules)
    scan_num = concatenated.attrs["scan_num"]
    out_regridded.attrs["scan_num"] = scan_num
    out_regridded = out_regridded.where(np.isfinite(out_regridded))
    if add_time:
        time = [np.array(ref_times[0], dtype="datetime64[ns]")]
        da_time = xr.DataArray(
            name="time",
            data=time,
            dims=["time"],
            attrs={"description": "Reference start time of first selected granule in scan."},
            coords={"time": (("time",), time)},
        )
        out_regridded = out_regridded.expand_dims(time=da_time)
        out_regridded["end_time"] = (("time",), [end_time])
        out_regridded["end_time"].attrs = {
            "description": "time at which the last swath of the scan starts"
        }
    if to_netcdf:
        if "XYZ" in path:
            scan_num = out_regridded.attrs["scan_num"]
            out_regridded.to_netcdf(
                path.replace(
                    "XYZ",
                    f"S{scan_num:03d}_{out_regridded['time'].values.astype(str)[0][0:19]}",
                )
            )
        else:
            out_regridded.to_netcdf(path)

    return out_regridded


def back_to_modgrid_multiscan(
    paireddict,
    modobj,
    to_netcdf=False,
    path="Regridded_object_XYZ.nc",
    method="bilinear",
    grid_path=None,
):
    """Grids object in sat-space to modgrid. Designed to grid back to modgrid after applying
    the scattering weights and air mass factors. It is designed for multiple scans, and uses
    back_to_modgrid under the hood. Generally, back_to_modgrid should only be used if
    you can ensure that you are reading only one scan at a time.

    Parameters
    ----------
    paireddict : collections.OrderedDict[str, xr.Dataset]
        An OrderedDict with time_reference strings as keys.
    modobj : xr.Dataset
        A modobj including the modgrid.
    to_netcdf : bool
        If True, save a netcdf with the paired data
    path : str
        The base name to save the files if to_netcdf is True. XX will be replaced
        with the first and list times. If to_netcdf is False, this will
        be ignored.
    method : str
        Method of regridding used by xESMF

    Returns
    -------
    xr.Dataset
        Dataset with obj2grid regridded to modobj.
    """
    out_regridded = xr.Dataset()
    ordered_keys = sorted(list(paireddict.keys()))
    scan_num = paireddict[ordered_keys[0]].attrs["scan_num"]
    keys_in_scan = [ordered_keys[0]]
    if len(ordered_keys) > 1:
        for k in ordered_keys[1:]:
            if paireddict[k].attrs["scan_num"] == scan_num:
                keys_in_scan.append(k)
            else:
                regridded_scan = back_to_modgrid(
                    paireddict, modobj, keys_in_scan, method=method, grid_path=grid_path
                )
                out_regridded = xr.merge([out_regridded, regridded_scan])
                scan_num = paireddict[k].attrs["scan_num"]
                keys_in_scan = [k]
    regridded_scan = back_to_modgrid(
        paireddict, modobj, keys_in_scan, add_time=True, method=method, grid_path=grid_path
    )
    out_regridded = xr.merge([out_regridded, regridded_scan])

    if to_netcdf:
        if "XYZ" in path:
            first_time = out_regridded["time"][0].values.astype(str)[0:19]
            last_time = out_regridded["time"][-1].values.astype(str)[0:19]
            out_regridded.to_netcdf(path.replace("XYZ", f"{first_time}_{last_time}"))
        else:
            out_regridded.to_netcdf(path)

    return out_regridded


def read_paired_gridded_tempo_model(path):
    """Reads in paired gridded tempo and model data

    Parameters
    ----------
    path : str or globobject

    Returns
    -------
    xr.Dataset
        combined dataset with paired tempo and gridded data.
    """

    pathlist = sorted(glob.glob(path))
    return xr.open_mfdataset(pathlist)


def read_paired_swath(path):
    """Read in paired swath data

    Parameters
    ----------
    path : str or globobject

    Returns
    -------
    collections.OrderedDict[str, xr.Dataset]
        OrderedDict with datasets containing every swath
    """
    pathlist = sorted(glob.glob(path))
    swaths = collections.OrderedDict()
    for f in pathlist:
        ds = xr.open_dataset(f)
        time = ds.attrs["reference_time_string"]
        swaths[time] = ds
    return swaths


def save_paired_swath(moddict, path="Paired_swath_XYZ.nc"):
    """Saves each swath individually

    Parameters
    ----------
    moddict : collections.OrderedDict[str, xr.Dataset]
        Ordered dict containing all of the paired model and swath.
    path : str
        Path to save the swath. If XYZ is present, it will be replaced
        by the date, number of scan and number of granule.

    Returns
    -------
    None
    """
    for i, k in enumerate(moddict.keys()):
        scan_num = moddict[k].attrs["scan_num"]
        gran_num = moddict[k].attrs["granule_number"]
        if isinstance(path, str):
            if "XYZ" in path:
                pathout = path.replace("XYZ", f"{k[:-1]}_S{scan_num:03d}G{gran_num:03d}")
            else:
                pathout = path.replace(".nc", f"{i}.nc")
        else:
            pathout = f"{i}_paired.nc"
        moddict[k].to_netcdf(pathout)


def select_by_keys(data_names, period="per_scan"):
    """Selects data containing the same scan or day. It does
    so by file name, so it is important that the standard naming
    is not altered.

    Parameters
    ----------
    data_names : list[str]
        list containing the names of the files to select from.
    selection_criteria : str
        str with the selection criteria. If "all", the function does nothing.
        If "per_scan", it will return a list of lists[str], with each inner
        list containing one complete scan.
        If "per_day", it will return a list of lists[str], with each inner
        list containing one complete day.

    Returns
    -------
    list[str] | list[list[str]]
        A sorted list of strings if "all" is provided, a list[list[str]]
        if a different criteria is provided.
    """
    import re

    date_names_sorted = sorted(data_names)

    if period not in ("all", "per_day", "per_scan"):
        warnings.warn(
            "Could not understand looping recommendations for processing files."
            ' Not in "all", "per_day" nor "per_scan". Adopting "all".'
        )
        period = "all"

    if period != "all":
        days = sorted({re.search(r"((\d{8}))T(\d{6})", s).group(1) for s in date_names_sorted})
        subgroups = []
        for day in days:
            subgroups.append([d for d in date_names_sorted if day in d])

        if period == "per_day":
            return subgroups

        if period == "per_scan":
            scans = []
            for sg in subgroups:
                scan = sorted({re.search(r"(S(\d{3}))G(\d{2})", s).group(1) for s in sg})
                for s in scan:
                    scans.append([f for f in sg if s in f])
            return scans

    # This only should happen if period == "all"
    return [date_names_sorted]
