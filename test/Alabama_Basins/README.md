## Building the Alabama Test -- Restarting T-Route from the Analysis and Assimilation CHRTOUT files.
This example is designed to show T-Route initializing execution from the web-available output of the
Analysis and Assimilation runs on the operational National Water Model. The operational model restarts
routing from a more comprehensive WRF-Hydro restart file, but that file is not shared through the
NODD resources on AWS and Google, so we needed to prepare this method as an alternate restarting
capability.

### Where to get the channel inflows to force the t-route simulation?
To re-create this test, download the CHRTOUT files from the short-range simulation on the day following
a significant rainstorm in Alabama:
``` bash
for i in {00..18}; do wget https://storage.googleapis.com/national-water-model/nwm.20260619/short_range/nwm.t04z.short_range.channel_rt.f0"$i".conus.nc; done
for i in {21..04}; do echo $i; g=$(printf "%02d" "$((i-3))"); h=$(printf "%02d" "$((i+1))"); mv nwm.*.f0"$g".conus.nc 20260619"$h"00.CHRTOUT_DOMAIN1; done
```

Filter down the forcing files:
``` py
filelist = [
    "202606190500.CHRTOUT_DOMAIN1",
    "202606190600.CHRTOUT_DOMAIN1",
    "202606190700.CHRTOUT_DOMAIN1",
    "202606190800.CHRTOUT_DOMAIN1",
    "202606190900.CHRTOUT_DOMAIN1",
    "202606191000.CHRTOUT_DOMAIN1",
    "202606191100.CHRTOUT_DOMAIN1",
    "202606191200.CHRTOUT_DOMAIN1",
    "202606191300.CHRTOUT_DOMAIN1",
    "202606191400.CHRTOUT_DOMAIN1",
    "202606191500.CHRTOUT_DOMAIN1",
    "202606191600.CHRTOUT_DOMAIN1",
    "202606191700.CHRTOUT_DOMAIN1",
    "202606191800.CHRTOUT_DOMAIN1",
    "202606191900.CHRTOUT_DOMAIN1",
    "202606192000.CHRTOUT_DOMAIN1",
    "202606192100.CHRTOUT_DOMAIN1",
    "202606192200.CHRTOUT_DOMAIN1",
]

import os
import xarray as xr
for f in filelist:
    # Make xarray readable name
    temp_name =  f + ".nc"
    os.rename("channel_forcing/" + f, "channel_forcing/" + temp_name)
    with xr.open_dataset("channel_forcing/" + temp_name) as ds:
        subset = ds.where(ds['feature_id'].isin(AL_sites_filter), drop=True).load()
    subset.to_netcdf("channel_forcing/" + f)

```


### How to rebuild the restart file?
Then, get the correct analysis file:
``` bash
wget https://storage.googleapis.com/national-water-model/nwm.20260619/analysis_assim/nwm.t04z.analysis_assim.channel_rt.tm00.conus.nc
```
There are steps to take to parse this down to a properly ordered file with fields needed for restart.
Those will be documented further soon, but for now, you can reference https://github.com/alk05/routing-comparison-nwm-nextgen-troute
After downloading, you can filter the CONUS result thus:
``` py
import xarray as xr
with xr.open_dataset("restart/troute_restart2026061904.nc") as ds:
    subset = ds.where(ds['links'].isin(AL_sites_filter), drop=True).load()
subset.to_netcdf("restart/troute_restart2026061904.nc")
```


### What is the source for the routing hydrofabric used in the test?
Download the RouteLink_CONUS.nc file from the NCO parameter store for the National Water Model. For example:
``` bash
wget https://www.nco.ncep.noaa.gov/pmb/codes/nwprod/nwm.v3.0.20/parm/domain/RouteLink_CONUS.nc
```
*Note:* This link will change with updates to model versions. Open the [base URL](https://www.nco.ncep.noaa.gov/pmb/codes/nwprod) to find the latest link.

... And subset it for the basins of interest:
``` py
import xarray as xr
ds = xr.open_dataset("domain/RouteLink_CONUS.nc")
with open("domain/specific_AL_sites.txt") as f:
    AL_sites_filter = [int(line.strip(", \n")) for line in f if line.strip(", \n")]

RouteLink_AL_test = ds.where(ds['link'].isin(AL_sites_filter), drop=True)

RouteLink_AL_test.to_netcdf("domain/RouteLink_AL_test.nc")
```

### How did we get the Feature ID List?
We used the package here to create the collection of feature_id's
https://github.com/jameshalgren/troute-network-analysis


## How do I run the test?
Simulation parameters for the test are contained in `t-route/test/Alabama_Basins/test_AnA_V4_NHD.yaml`.

To execute the test:

``` bash
# navigate to t-route repository directory
$ cd t-route

# compile routing kernels, reservoir modules, cython scripts, etc. to install python into the virtual environment
$ ./compiler.sh

# run the test
$ cd t-route/test/Alabama_Basins
$ python -m nwm_routing -f -V4 test_AnA_V4_NHD.yaml
```
