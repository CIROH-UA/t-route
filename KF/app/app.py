import time
import json as jsonlib
from datetime import datetime, timezone, timedelta
import asyncio
import numpy as np
import pandas as pd
import requests
from tx_fast_hydrology.muskingum import ModelCollection
from tx_fast_hydrology.da import KalmanFilter
from tx_fast_hydrology.simulation import AsyncSimulation, CheckPoint
from tx_fast_hydrology.download import download_gage_data, get_forcing_directories, get_forecast_path, download_nwm_forcings, download_nwm_streamflow
from sanic import Sanic, response, json, text, empty
from sanic.log import logger
from sanic.worker.manager import WorkerManager
from sanic_ext import render

# Create server app
APP_NAME = 'muskingum'
app = Sanic(APP_NAME)
# Set timeout threshold for app startup
WorkerManager.THRESHOLD = 2400

# Set up static directory
app.static('/static', './static')

# Constants
GAGE_LOOKBACK_HOURS = 2

# Response codes
OK = 200
CREATED = 201
BAD_REQUEST = 400

# Load COMIDS
# TODO: Clean this up
all_ids_path = "/home/akagi/Documents/Git/tx-fast-hydrology/data/usgs_subset_attila.csv"
all_ids = pd.read_csv(all_ids_path).drop_duplicates(subset='comid')
comids = pd.read_csv("/home/akagi/Documents/Git/tx-fast-hydrology/data/COMIDs.csv",
                        index_col=0)['0'].values


async def tick(app):
    tick_dt = app.ctx.tick_dt
    simulation = app.ctx.simulation
    last_timestamp = app.ctx.current_timestamp
    logger.info(f'Sleeping for {tick_dt} seconds...')
    await asyncio.sleep(tick_dt)
    urls = get_forcing_directories()
    nwm_dir, forecast_hour, timestamp = get_forecast_path(urls)
    if timestamp > last_timestamp:
        logger.info(f'New forcings available at timestamp {timestamp}')
        streamflow = download_nwm_streamflow(nwm_dir, forecast_hour=forecast_hour, comids=comids)
        logger.info(f'Streamflow downloaded')
        inputs = download_nwm_forcings(nwm_dir, forecast_hour=forecast_hour, comids=comids)
        logger.info(f'Forcings downloaded')
        simulation.load_states()
        simulation.inputs = simulation.load_inputs(inputs)
        # Download gage data
        gage_end_time = pd.to_datetime(datetime.now(timezone.utc))
        gage_start_time = pd.to_datetime(datetime.now(timezone.utc) 
                                         - timedelta(hours=GAGE_LOOKBACK_HOURS))
        # TODO: Check these start and end times
        gage_end_time = max(gage_end_time, timestamp)
        gage_start_time = min(gage_start_time, last_timestamp)
        app.ctx.all_ids['to'] = gage_end_time.isoformat()
        app.ctx.all_ids['from'] = gage_start_time.isoformat()
        logger.info('Downloading gage data...')
        measurements = await download_gage_data(app.ctx.all_ids.to_dict(orient='records'))
        # Ensure that we always use the same ordering and columns
        measurements = measurements.reindex(app.ctx.all_ids['comid'].values.astype(str), axis=1)
        # TODO: This needs to be fixed eventually
        measurements = measurements.fillna(0.)
        logger.info('Gage data downloaded')
        logger.info('Assigning gage data to subbasin models...')
        for model in simulation.model_collection.models.values():
            if hasattr(model, 'callbacks') and 'kf' in model.callbacks:
                measurements_columns = model.callbacks['kf'].measurements.columns
                basin_measurements = measurements[measurements_columns]
                #print(f'Removing duplicate columns')
                #basin_measurements = basin_measurements.loc[:, ~basin_measurements.columns.duplicated()].copy()
                model.callbacks['kf'].measurements = basin_measurements
        # Print current times
        logger.info(f'Gage start time: {measurements.index.min().isoformat()}\n'
                    f'Gage end time: {measurements.index.max().isoformat()}\n'
                    f'Input start time: {inputs.index.min().isoformat()}\n'
                    f'Input end time: {inputs.index.max().isoformat()}\n'
                    f'Model timestamp: {simulation.datetime.isoformat()}')
        # Step model forward in time
        logger.info('Beginning simulation...')
        outputs = await simulation.simulate()
        outputs = pd.concat([series for series in outputs.values()], axis=1)
        logger.info('Simulation finished')
        app.ctx.simulation = simulation
        app.ctx.outputs = outputs
        app.ctx.current_timestamp = timestamp
        app.ctx.streamflow = streamflow
    # Queue next loop
    app.add_task(tick, name='tick')

@app.before_server_start
async def start_model(app, loop):
    # Create model
    input_path = '/home/akagi/Documents/Git/tx-fast-hydrology/data/huc8.json'
    # Set app parameters
    app.ctx.tick_dt = 600.0
    # Download gage data
    app.ctx.all_ids = all_ids
    gage_end_time = pd.to_datetime(datetime.now(timezone.utc))
    gage_start_time = pd.to_datetime(datetime.now(timezone.utc) 
                                     - timedelta(hours=GAGE_LOOKBACK_HOURS))
    app.ctx.all_ids['to'] = gage_end_time.isoformat()
    app.ctx.all_ids['from'] = gage_start_time.isoformat()
    logger.info('Downloading gage data...')
    measurements = await download_gage_data(app.ctx.all_ids.to_dict(orient='records'))
    # Ensure that we always use the same ordering and columns
    measurements = measurements.reindex(app.ctx.all_ids['comid'].values.astype(str), axis=1)
    # TODO: This needs to be fixed eventually
    measurements = measurements.fillna(0.)
    logger.info('Gage data downloaded')
    # Create model collection
    model_collection = ModelCollection.from_file(input_path)
    logger.info('Model collection loaded')
    # Set checkpoint callbacks
    logger.info('Setting up checkpoints...')
    for model in model_collection.models.values():
        checkpoint = CheckPoint(model, timedelta=3600.)
        model.bind_callback(checkpoint, key='checkpoint')
    # Set up Kalman Filter
    logger.info('Setting up Kalman Filter...')
    for model in model_collection.models.values():
        model_sites = [reach_id for reach_id in model.reach_ids 
                       if reach_id in measurements.columns]
        if model_sites:
            basin_measurements = measurements[model_sites]
            # Remove duplicated COMIDs
            # basin_measurements = basin_measurements.loc[:, ~basin_measurements.columns.duplicated()].copy()
            Q_cov = 2 * np.eye(model.n)
            R_cov = 1e-2 * np.eye(basin_measurements.shape[1])
            P_t_init = Q_cov.copy()
            kf = KalmanFilter(model, basin_measurements, Q_cov, R_cov, P_t_init)
            model.bind_callback(kf, key='kf')
            try:
                assert model.o_t_next[kf.s].size == kf.measurements.columns.size
            except:
                raise
    # Download latest forcings and streamflows
    logger.info('Downloading initial NWM forcings and streamflows...')
    urls = get_forcing_directories()
    nwm_dir, forecast_hour, timestamp = get_forecast_path(urls)
    streamflow = download_nwm_streamflow(nwm_dir, forecast_hour=forecast_hour, comids=comids)
    logger.info('Initial streamflow downloaded')
    inputs = download_nwm_forcings(nwm_dir, forecast_hour=forecast_hour, comids=comids)
    logger.info('Initial forcings downloaded')
    # Create simulation class instance
    simulation = AsyncSimulation(model_collection, inputs)
    logger.info('Simulation instance created')
    timestamp = streamflow.index.item()
    streamflow_values = streamflow.loc[timestamp]
    # Set timestamp and initial states for all models
    simulation.set_datetime(timestamp)
    simulation.init_states(streamflow_values)
    simulation.save_states()
    logger.info('Model states initialized')
    # Run initial simulation
    logger.info('Beginning initial simulation...')
    outputs = await simulation.simulate()
    outputs = pd.concat([series for series in outputs.values()], axis=1)
    logger.info('Initial simulation finished')
    # Add persistent objects to global app context
    app.ctx.simulation = simulation
    app.ctx.outputs = outputs
    app.ctx.current_timestamp = timestamp
    app.ctx.streamflow = streamflow
    # Read geojson
    with open('/home/akagi/Documents/Git/tx-fast-hydrology/data/huc2_12_nhd_min.json') as basin:
        stream_network = jsonlib.load(basin)
        app.ctx.stream_network = stream_network
    # Start loop
    app.add_task(tick)

@app.get("/forecast/<reach_id>")
async def reach_forecast(request, reach_id):
    outputs = app.ctx.outputs
    reach_id = str(reach_id)
    timestamp_utc = [index.isoformat() for index in outputs.index]
    streamflow_cms = [value for value in outputs[reach_id].values]
    json_output = {'timestamp__utc' : timestamp_utc, 
                   'streamflow__cms' : streamflow_cms}
    return json(json_output)

@app.get("/diff")
async def reach_diff(request):
    outputs = app.ctx.outputs
    streamflow = app.ctx.streamflow
    time_index = streamflow.index.item()
    diff = outputs.loc[time_index, streamflow.columns] - streamflow
    pct_diff = (diff / streamflow)
    diff = diff.loc[time_index].fillna(0.).replace([np.inf, -np.inf], 0.).to_dict()
    pct_diff = pct_diff.loc[time_index].fillna(0.).replace([np.inf, -np.inf], 0.).to_dict()
    json_output = {'diff__cms' : diff, 
                   'pct_diff__pct' : pct_diff}
    return json(json_output)

@app.get("/map")
async def map_handler(request):
    stream_network = app.ctx.stream_network
    # Get differences
    outputs = app.ctx.outputs
    streamflow = app.ctx.streamflow
    time_index = streamflow.index.item()
    diff = outputs.loc[time_index, streamflow.columns] - streamflow
    pct_diff = (diff / streamflow)
    diff = diff.loc[time_index].fillna(0.).replace([np.inf, -np.inf], 0.)
    pct_diff = pct_diff.loc[time_index].fillna(0.).replace([np.inf, -np.inf], 0.)
    hi = float(diff.quantile(0.90))
    lo = float(diff.quantile(0.10))
    for feature in stream_network['features']:
        comid = str(feature['properties']['COMID'])
        path_diff = diff[comid]
        if path_diff > hi:
            c = 'positive'
        elif path_diff < lo:
            c = 'negative'
        else:
            c = 'zero'
        feature['properties']['change'] = c
    return await render("show_page.html", context={'streams_json' : stream_network})

if __name__ == "__main__":
    app.run()
