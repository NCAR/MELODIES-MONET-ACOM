# Copyright (C) 2022 National Center for Atmospheric Research and National Oceanic and Atmospheric Administration
# SPDX-License-Identifier: Apache-2.0
#
import os
import warnings
import pandas as pd
import xarray as xr


def read_saved_data(analysis, filenames, method, attr, xr_kws={}):
    """Read previously saved dict containing melodies-monet data (:attr:`paired`, :attr:`models`, or :attr:`obs`)
    from pickle file or netcdf file, populating the :attr:`paired`, :attr:`models`, or :attr:`obs` dict.

    Parameters
    ----------
    analysis : melodies_monet.driver.analysis
        Instance of the analysis class from driver script.
    filenames : str or iterable
        str or list for reading in pkl. For netCDF, must be dict with format {group1:str or iterable of filenames, group2:...}
    method : str
        One of either 'pkl' or 'netcdf'.
    attr : str
        The analysis attribute that will be populated with the saved data. One of either 'paired' or 'models' or 'obs'.
    **kwargs : optional
        Additional keyword arguments for xr.open_dataset()

    Returns
    -------
    None
    """
    from glob import glob
    from .. import tutorial
    
    # Determine where to read files from
    if getattr(analysis,'output_dir_read') is not None:
        read_dir = getattr(analysis,'output_dir_read')
    else:
        read_dir = ''
    
    # expand any wildcards in the filenames
    if method=='pkl':
        if isinstance(filenames,str):
            files = sorted([file for sublist in [glob(os.path.join(read_dir,file)) for file in [filenames]] for file in sublist])
        else:
            files = sorted([file for sublist in [glob(os.path.join(read_dir,file)) for file in filenames] for file in sublist])
        if not files:
            raise FileNotFoundError('No such file: ',filenames)
    elif method=='netcdf':
        if isinstance(filenames,dict): 
            files = {}
            for group in filenames.keys():
                if isinstance(filenames[group],str):
                    files[group] = sorted([file for sublist in [glob(os.path.join(read_dir,file)) for file in [filenames[group]]] for file in sublist])
                else:
                     if filenames[group][0].startswith("example:"):
                        files[group] = sorted([file for sublist in [
                            [tutorial.fetch_example(":".join(s.strip() for s in file.split(":")[1:]))] for file in filenames[group]] for file in sublist])
                     else:
                         files[group] = sorted([file for sublist in [glob(os.path.join(read_dir,file)) for file in filenames[group]] for file in sublist])
                if not files[group]:
                    raise FileNotFoundError('No such file: ', filenames[group])
        else:
            raise TypeError('NetCDF format filenames need to be specified as a dict, with format {group1:str or iterable of filenames, group2:...}')
    
    # Set analysis.read such that it now contains expanded filenames so user has list of read files
    expanded_filenames = getattr(analysis,'read')
    expanded_filenames[attr]['filenames'] = files 
    setattr(analysis, 'read', expanded_filenames)

    # for converting name of attribute to name of class for constructing
    class_names = {'paired':'pair','models':'model','obs':'observation'}

    if method=='pkl':
        if len(files)==1:
            setattr(analysis, attr, read_pkl(files[0]))
        elif len(files)>1:
            for count, file in enumerate(files):
                if count==0:
                    attr_out = read_pkl(file)
                else:
                    attr_append = read_pkl(file)
                    for group in attr_out.keys():
                        attr_out[group].obj = xr.merge([attr_out[group].obj,attr_append[group].obj])
            setattr(analysis, attr,  attr_out)

    elif method=='netcdf':
        xr_dict = {}
        for group in files.keys():
            if isinstance(files[group],str):
                group_files = [files[group]]
            else:
                group_files = files[group]
            xr_dict[group] = read_analysis_ncf(group_files,xr_kws)
        setattr(analysis, attr,  xarray_to_class(class_type=class_names[attr],group_ds=xr_dict))

def read_pkl(filename):
    """Function to read a pickle file containing part of the analysis class (models, obs, paired)

    Parameters
    ----------
    filename : type
        Description of parameter `filename`.
        
    Returns
    -------
    obj : type
        Description of returned object.

    """
    from joblib import load
    
    print('Reading:', filename)
    with open(filename, 'rb') as file:
        obj = load(file)
        
    return obj

def read_analysis_ncf(filenames,xr_kws={}):
    """Function to read netcdf4 files containing an object within an attribute of a part of the
    analysis class (models, obs, paired). For example, a single model/obs pairing or a single model. 
    If the object is saved in multiple files, the function will merge the files.

    Parameters
    ----------
    filenames : str or iterable
        Description of parameter `filename`.
    xr_kws : optional
        Additional keyword arguments for xr.open_dataset()
        
    Returns
    -------
    ds_out : type
        Xarray dataset containing merged files.

    """
    if len(filenames)==1:
        print('Reading:', filenames[0])
        ds_out = xr.open_dataset(filenames[0],**xr_kws)
        
    elif len(filenames)>1:
        for count, file in enumerate(filenames):
            print('Reading:', file)

            if count==0:
                ds_out = xr.open_dataset(file,**xr_kws)
                group_name1 =  ds_out.attrs['group_name']

            else:
                ds_append = xr.open_dataset(file,**xr_kws)
                # Test if all the files have the same group to prevent merge issues
                if group_name1 != ds_append.attrs['group_name']:
                    raise Exception('The group names are not consistent between the netcdf files being read.') 
                else:
                    ds_out = xr.merge([ds_out,ds_append])
            
    return ds_out

def xarray_to_class(class_type,group_ds):
    """Remake dict containing driver class instances from dict of xarray datasets. Dict of xarray datasets must contain 
    global attribute that contains json formatted class attributes.

    Parameters
    ----------
    class_type : str
        One of 'model', 'pair' or 'observation'
    group_ds : dict
        dict containing xarray datasets from read_grouped_ncf.

    Returns
    -------
    class_dict
    """
    import json
    from melodies_monet import driver
    
    class_dict = {}
    for group in group_ds.keys():
        if class_type == 'pair':
            c=driver.pair()
        elif class_type == 'model':
            c=driver.model()
        elif class_type == 'observation':
            c=driver.observation()

        obj_dict = json.loads(group_ds[group].attrs['dict_json'])
        
        for attr in obj_dict.keys():
            setattr(c, attr, obj_dict[attr])
        c.obj = group_ds[group]
        class_dict[group]=c

    return class_dict

def read_aircraft_obs_csv(filename,time_var=None):
    """Function to read .csv formatted aircraft observations.

    Parameters
    ----------
    filename : str 
        Filename of .csv file to be read
    time_var : optional
        The variable in the dataset that should be converted to 
        datetime format, renamed to `time` and set as a dimension.
        
    Returns
    -------
    ds_out : xarray.Dataset
        Xarray dataset containing information from .csv file

    """
    df = pd.read_csv(filename)
    if time_var is not None:
        df.rename(columns={time_var:'time'},inplace=True)
        df['time']  = pd.to_datetime(df['time'])
        
    # Sort the values based on time
    df.sort_values(by='time',inplace=True,ignore_index=True)
        
    df.set_index('time',inplace=True)
    
    return xr.Dataset.from_dataframe(df)
""" Reads and formats the excel files from CDPHE VOC Canisters"""


def read_site_excel(data_path, site_data, site_number=None):
    """Load a site from from an MS Excel sheet.
    Currently optimized for CDPHE's VOC canister data.

    Parameters
    ----------
    data_path: str
        Path to the excel file containing the data
    site_data: dict
        dict containing the data of the site.
        Required keys are:
            'coords': {'latitude': float, 'longitude': float}
                Coordinates of the site
            'sheet_name': str
                name of the excel sheet containing the site
        Optional keys:
            skiprows: int
                rows that should be skipped when reading the excel sheet.
                Defaults to 0.
            headers: int | list[int]
                rows with headers for building the data frame.
                Take into account that 'skiprows' takes precedence.
                I.e., if the headers are in rows [15,16], but you
                already typed 'skiprows': 15, you should type
                'headers': [0,1].
                Defaults to 0.
            site_id: str
                ID of the site. Defaults to sheet_name.
            qualifier_name: str
                Name of row containing the qualifiers.
                If None, 'Qualifier' is assumed.
            ignore_qualifiers: None
                If None, only data without qualifiers is plotted.
            na_values: scalar | str | list-like | dict | default None
                Values that are NaN
    site_number: int
        Number of site. If only one site is provided, it should be 0.
        This keyword is set for clearer compilation of multiple sites.

    Returns
    -------
    xr.Dataset
        Dataset containing the information of the site
    """
    params = {
        "skiprows": None,
        "headers": 0,
        "site_id": site_data["sheet_name"],
        "qualifier_name": "Qualifier",
        "keep_qualifiers": None,
        "na_values": "None" ** site_data,
    }
    site_number = 0 if site_number is None else site_number
    data = pd.read_excel(
        data_path,
        sheet_name=params["sheet_name"],
        skiprows=params["skiprows"],
        header=params["skiprows"],
        na_values=params["na_values"],
    )
    keep_qualifiers = params["keep_qualifiers"]
    if keep_qualifiers is None or keep_qualifiers != "no":
        data = data[data[params["qualifier_name"].isnull()]]
    elif params["keep_qualifiers"] != "all":
        data = data[
            data["Qualifier"].isnull() or data["Qualifier"].isin(list(params[keep_qualifiers]))
        ]
    time = data["Sample"]["Date"].dt.tz_localize("America/Denver").dt.tz_convert(None)
    data.loc[:, ("Sample", "Date")] = time
    variables = data["Analysis"]["Analyte"].unique()
    compiled_data = xr.Dataset()
    for v in variables:
        tmp_data = data.loc[data["Analysis"]["Analyte"] == v]
        tmp_ds = xr.Dataset()
        tmp_ds["time"] = (("time",), tmp_data["Sample"]["Date"].values)
        tmp_ds["x"] = (("x",), [site_number])
        tmp_ds[v] = (("time", "x"), tmp_data["CAS"]["Result"].values[..., None])
        tmp_ds[v].attrs = {
            "values": tmp_data["Detection"]["Units"].unique(),
        }
        compiled_data = xr.merge([compiled_data, tmp_ds])

    compiled_data["longitude"] = (("x",), [params["site_coords"]["longitude"]])
    compiled_data["latitude"] = (("x",), [params["site_coords"]["latitude"]])
    compiled_data["siteid"] = (("x",), [params["site_id"]])
    compiled_data = compiled_data.drop_duplicates("time")
    return compiled_data


def compile_sites_excel(data_path, site_dict):
    """Compiles all sites in a file
    Currently optimized for CDPHE VOC canister data, but should work for most.

    Parameters
    ----------
    data_path : str
        Path to the excel file containing the data.
    site_dict : dict
        Dictionary containing a key per site (ideally, the site's name)
        and a dict as site_value, to pass to read_cdphe_site

    Returns
    -------
    xr.Dataset
        Dataset with all sites compiled
    """

    compiled_data = xr.Dataset()
    for n, s in enumerate(list(site_dict.keys())):
        site_data = site_dict[s]
        ds = read_site_excel(data_path, site_data, site_number=n)
        compiled_data = xr.merge([compiled_data, ds])
    return compiled_data


def control_reading_excel(data_path, site_type, site_dict):
    """Controls the reading and file preparation process.

    Parameters
    ----------
    path : str | list | glob object
        Path to files that should be opened
    site_type : str
        Type of site/excel to read. Currently, only CDPHE VOC Canisters are implemented
    site_dict : dict

    Returns
    -------
    xr.Dataset
        Dataset with excel compiled
    """
    if site_type != "cdphe_canisters":
        warnings.warn("site_type is not cdphe_canisters. Will attempt to read anyway")
    return compile_sites_excel(data_path, site_dict)
