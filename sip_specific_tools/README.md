# Tools specifically designed for the SIP of Colorado

## Preprocessors
This preprocessors are designed for preprocessing non public datasets that are to be used by
the SIP of Colorado
### BoulderAir
BoulderAir data should be pre-processed to bo completely consistent with MELODIES-MONET's format.
This can be done by using the BoulderAir preprocessor, as follows

```
./boulderair.py -c coordinates_file.csv -p path_to_data_XYZ_etc.csv -v variable1,variable2,...,variablen -r resample_freq -m method -o output_name.nc
```

options: 
`-c, --coordinates`
`-p, --path`
`-v, --variables`
`-r, --resample_freq`
`-m, --method`
`-o, --output`

`coordinates_file.csv`: CSV containing the following fields: site_abbreviation,lat,lon,m_asl
`path_to_data_XYZ_etc.csv`: BoulderAir data. XYZ will be replaced automatically by the code with every site_abbreviation on the coordinates_file.csv. Wildcards (like `"*"` can be used).
`variablex`: Variables of interest.
`resample`(optional): If the data should be resampled. Options: `h`, `d`. If None is provided, the time in the files will be kept without resampling.
`method`: How to perform the resample. Default: `inst` -> The data will be interpolated linearly to represent instantaneous data at the resampling time. Other options: `median`, `mean`, `max`, `min`.
`output_name.nc`: Name of the output file. It will be netCDF
