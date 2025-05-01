# SPDX-License-Identifier: Apache-2.0
#
"""Module to control and do the preprocessing"""


import xarray as xr
import numpy as np
from .tools import calc_totalcolumn


def average_between_hours(data, start_hours, nhours):
    """Calculates the average from start_hours, including nhours forward.

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Dataset containing the data
    start_hours : xr.DataArray[datetime64] | np.ndarray[datetime64]
        Starting times for the average
    nhours : int | float
        Number of hours forward

    Returns
    -------
    xr.Dataset | xr.DataArray
        Dataset or DataArray (depending on input) containing averaged
        data.
    """

    existing_h = np.intersect1d(start_hours, data["time"])
    data_out = xr.zeros_like(data.sel(time=existing_h))
    for t, start in enumerate(existing_h):
        data_out[{"time": t}] = data.sel(
            time=slice(start, start + np.timedelta64(nhours, "h"))
        ).mean(dim="time", keep_attrs=True)
    data_out.attrs["description"] = (
        f"{nhours} means, starting at reported time. {data_out.attrs.get('description', '')}"
    )
    return data_out


def preprocessing(data, preproc_kwargs):
    """Does the required preprocessing for the data.

    Parameters
    ----------
    data : xr.Dataset | xr.DataArray
        Dataset containing the data to process
    preproc_kwargs: dict
        Dictionary containing the required keyword arguments for the
        preprocessing

    Returns
    -------
    xr.Dataset
    Dataset or DataArray (consistent with the input) containing the
    processed data
    """
    data_out = xr.Dataset()
    for preproc_type, kw in preproc_kwargs.items():
        if preproc_type == "average":
            data_out[f"{kw['var']}_average"] = average_between_hours(
                data, kw["start_hours"], kw["nhours"]
            )
        elif preproc_type == "total_column":
            data_out[f"{kw['var']}_total_column"] = calc_totalcolumn(data, kw["var"])
    return data_out
