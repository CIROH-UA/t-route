import os
import asyncio
import time
import pathlib
import io
import re
import requests
import xarray as xr
import pandas as pd
from bs4 import BeautifulSoup
import httpx
from typing import Any
from tqdm.asyncio import tqdm as tqdm_asyncio
from tqdm import tqdm

# Define the macros as environment variables or constants
KISTERS_USER = os.environ['KISTERS_USER']
KISTERS_PASS = os.environ['KISTERS_PASS']
KISTERS_BASE_URL = os.environ['KISTERS_BASE_URL']
MAX_CONCURRENT_REQUESTS = 2

def results_to_df(results: list[dict[str, Any]]) -> pd.DataFrame:
    dataframes = []
    for result in tqdm(results, desc="Combining DataFrames"):
        comid = result["comid"]
        data_entries = result.get("data", [])

        # Create a DataFrame with datetime index and comid as the column
        df = pd.DataFrame(
            data_entries, columns=["timestamp", "value", "quality", "remark"]
        )
        if df.empty:  # Skip if the DataFrame is empty
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df[["value"]].rename(columns={"value": comid})

        dataframes.append(df)
    # TODO: Sometimes this fails with no data obtained from the server
    combined_df = pd.concat(dataframes, axis=1)
    return combined_df

async def fetch_data(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, params: dict
) -> dict:
    url = f"{KISTERS_BASE_URL}/channels/{params['channel_id']}/timeSeries/data"
    request_params = {
        "to": params["to"],
        "from": params["from"],
        "channel_id": params["channel_id"],
        "tsId": params["timeseries_id"],
        "format": "JSON",
    }

    async with sem:  # Limit the concurrency
        try:
            response = await client.get(url, params=request_params)
            response.raise_for_status()
            result = response.json()
            return {
                "success": True,
                "data": {
                    "comid": params["comid"],
                    "channel_id": params["channel_id"],
                    "timeseries_id": params["timeseries_id"],
                    **result[0]["locations"][0]["timeseries"][0],
                },
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"{str(e)} - {response.text}",
                "payload": params,
            }

async def download_gage_data(data_list: list[dict]):
    # Lists to store results
    successful_requests = []
    unsuccessful_requests = []
    successful_data_list = []
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with httpx.AsyncClient(auth=(KISTERS_USER, KISTERS_PASS)) as client:
        tasks = [fetch_data(client, sem, params) for params in data_list]

        # Use tqdm.asyncio.tqdm to add a progress bar to data fetching
        results = await tqdm_asyncio.gather(*tasks, desc="Fetching Data", total=len(data_list))
    # Separate successful and unsuccessful requests
    for idx, result in enumerate(results):
        if result["success"]:
            successful_requests.append(result["data"])
            successful_data_list.append(data_list[idx])  # Add corresponding row to successful_data_list
        else:
            unsuccessful_requests.append({
                "error": result["error"],
                "payload": result["payload"],
            })

    # Save results to CSVs
    if successful_requests:
        success_df = results_to_df(successful_requests)
        success_df = success_df.interpolate().bfill().ffill()
        success_df.columns = success_df.columns.astype(str)
        return success_df
    else:
        return None


def get_forcing_directories():
    baseurl = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/nwm/prod'
    regex = re.compile('^nwm\.\d{8}/$')
    response = requests.get(baseurl)
    if response.status_code == 200:
        doc = BeautifulSoup(response.text, 'lxml')
        body = doc.find(name='body')
        links = body.find_all(name='a')
        urls = []
        for link in links:
            if 'href' in link.attrs:
                url = link.attrs['href']
                match = regex.match(url)
                if match:
                    urls.append(url)
        return [f'{baseurl}/{url}' for url in urls]
    else:
        return []

def get_latest_forecast_hour(nwm_dir):
    regex = re.compile('^nwm\.t(\d{2})z\.short_range.channel_rt')
    response = requests.get(f'{nwm_dir}/short_range/')
    if response.status_code == 200:
        doc = BeautifulSoup(response.content, 'lxml')
        body = doc.find(name='body')
        links = body.find_all(name='a')
        forecast_start_hours = set()
        for link in links:
            if 'href' in link.attrs:
                url = link.attrs['href']
                match = regex.match(url)
                if match:
                    next_hour = int(match.group(1))
                    forecast_start_hours.add(next_hour)
        return max(forecast_start_hours)
    else:
        return None

def get_forecast_path(nwm_dirs):
    for url in reversed(sorted(nwm_dirs)):
        forecast_hour = get_latest_forecast_hour(url)
        if forecast_hour is None:
            continue
        else:
            date = pathlib.Path(url).name.split('.')[1]
            hour = pd.to_timedelta(forecast_hour, unit='h')
            timestamp = pd.to_datetime(date, utc=True) + hour
            return url, forecast_hour, timestamp
    raise LookupError('No files found.')

def download_nwm_streamflow(nwm_dir, forecast_hour, comids, sleeptime=0.):
        # TODO: Should this include tm01 and tm02 as well?
        nc_url = f'analysis_assim/nwm.t{forecast_hour:02}z.analysis_assim.channel_rt.tm00.conus.nc'
        url = os.path.join(nwm_dir, nc_url)
        response = requests.get(url)
        if response.status_code == 200:
            dataset = xr.load_dataset(io.BytesIO(response.content), engine='h5netcdf')
        else:
            raise PermissionError(response.status_code)
        datetime = pd.to_datetime(dataset['time'].values.item(), utc=True)
        streamflow = dataset['streamflow'].sel(feature_id=comids).values
        streamflow = pd.DataFrame(pd.Series(streamflow, index=comids), columns=[datetime]).T
        streamflow.columns = streamflow.columns.astype(str)
        return streamflow

def download_nwm_forcings(nwm_dir, forecast_hour, comids, sleeptime=0.):
    # Download NetCDF forcings
    datasets = {}
    for lookahead_hour in range(1, 19):
        nc_url = f'short_range/nwm.t{forecast_hour:02}z.short_range.channel_rt.f{lookahead_hour:03}.conus.nc'
        url = os.path.join(nwm_dir, nc_url)
        response = requests.get(url)
        if response.status_code == 200:
            dataset = xr.load_dataset(io.BytesIO(response.content), engine='h5netcdf')
        else:
            raise PermissionError(response.status_code)
        datasets[lookahead_hour] = dataset
        time.sleep(sleeptime)
    # Parse NetCDF forcings
    qSfcLatRunoff = {}
    qBucket = {}
    for key, dataset in datasets.items():
        datetime = pd.to_datetime(dataset['time'].values.item(), utc=True)
        runoff = dataset['qSfcLatRunoff'].sel(feature_id=comids).values
        bucket = dataset['qBucket'].sel(feature_id=comids).values
        qSfcLatRunoff[datetime] = runoff
        qBucket[datetime] = bucket
    qSfcLatRunoff = pd.DataFrame.from_dict(qSfcLatRunoff, orient='index', columns=comids)
    qBucket = pd.DataFrame.from_dict(qBucket, orient='index', columns=comids)
    inputs = qSfcLatRunoff + qBucket
    inputs.columns = inputs.columns.astype(str)
    return inputs    