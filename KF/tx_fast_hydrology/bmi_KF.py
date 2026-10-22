"""
BMI wrapper for KF code base.

This class exposes Muskingum and KF model through BMI.

Written and tested by: Sonam Lama (slama@ua.edu)

"""


import asyncio
import threading
import itertools

import numpy as np
import pandas as pd

from tx_fast_hydrology.simulation import AsyncSimulation
from tx_fast_hydrology.da import KalmanFilter
from tx_fast_hydrology.muskingum import ModelCollection


#we will have remove this later. This adds streamflow in the upstream boundary
UPSTREAM_INPUTS = {
    "5779305": "5781161",
    "5785187": "5785899",
    "5786029": "5785351",
    }

#These are the gages for hold out assessment. Ideally, we would want to be this in the yaml file
#or elsewhere, where user can define what gages to remove. 
EXCLUDE_GAGES = [8155400,8158000]

class BmiKF:
    """BMI wrapper for the TX fast hydrology model collection with DA."""

    #these are the variables user can set value by using set value functions
    _INPUT_VAR_NAMES = ("k", "x", "initial_discharge")

    #these are the variables user can get value by using the get value functions
    _OUTPUT_VAR_NAMES = ("reach_list", "discharge")

    def load_config(self, config_path="KF.yaml"):
        """
        Loads the config file.

        Parameters:
        config_path: str
            path to the yaml file that helps to initialize the model.
        """
        config = {}
        with open(config_path, "r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    key, value = line.split(":", 1)
                    config[key.strip()] = value.strip().strip('"')
        return config

    def load_and_build_inputs(self, config):
        """
        Loads the input values for the function such
        as network, measurements, forcing (lateral flow)
        values.

        Parameters:
        config: dict
            dictionary that gets created by using the config file path.
        """

        model_collection = ModelCollection.from_file(config["network_file_path"])

        #this is the internal time step of the model
        dt = model_collection.timedelta.seconds

        inputs = None
        start_date = config["start_date"]
        end_date = config["end_date"]

        #build these inputs only if with_troute is false
        #otherwise lateral flow is not needed and just initiated to 0s
        if not self.with_troute:
            forcing = pd.read_csv(config["forcing_file_path"], index_col=0)
            streamflow = pd.read_csv(config["streamflow_file_path"], index_col=0)

            forcing.index = pd.to_datetime(forcing.index)
            streamflow.index = pd.to_datetime(streamflow.index)

            # Subset forcing and streamflow by the specified time window
            forcing = forcing.loc[start_date:end_date]
            streamflow = streamflow.loc[start_date:end_date]

            forcing.columns = forcing.columns.astype(str)
            streamflow.columns = streamflow.columns.astype(str)

            input_columns = list(
                itertools.chain.from_iterable(
                    [model.reach_ids for model in model_collection.models.values()]
                )
            )
            inputs = pd.DataFrame(0.0, index=forcing.index.copy(), columns=input_columns)

            for col in inputs.columns:
                if col in forcing.columns:
                    inputs[col] = forcing[col]

            inputs = inputs.resample(f"{dt}s").mean()
            inputs = inputs.interpolate().bfill().ffill()
            assert not inputs.isnull().any().any()

            for downstream_reach, upstream_reach in UPSTREAM_INPUTS.items():
                if upstream_reach in streamflow.columns:
                    inputs[downstream_reach] += streamflow[upstream_reach]

        else:
            time_index = pd.date_range(
            start=start_date, 
            end=end_date, 
            freq=f"{dt}s"
            )
            input_columns = list(
                itertools.chain.from_iterable(
                    [model.reach_ids for model in model_collection.models.values()]
                    )
                )
            inputs = pd.DataFrame(0.0, index=time_index, columns=input_columns)

        #create the observation dataframe
        measurements = pd.read_csv(config["measurement_file_path"], index_col=0, usecols=lambda col: col not in EXCLUDE_GAGES)
        measurements.index = pd.to_datetime(measurements.index)
        measurements.columns = measurements.columns.astype(str)

        usgs_to_comid = pd.read_csv(config["usgs_to_comid_file_path"], index_col=0)
        usgs_to_comid["gage_id"] = usgs_to_comid["gage_id"].astype(str)
        usgs_to_comid["comid"] = usgs_to_comid["comid"].astype(str)
        usgs_to_comid = usgs_to_comid[usgs_to_comid["comid"].isin(input_columns)]
        usgs_to_comid = pd.Series(
            usgs_to_comid["comid"].values,
            index=usgs_to_comid["gage_id"].values,
        )

        measurements = measurements[usgs_to_comid.index]
        measurements.columns = measurements.columns.map(usgs_to_comid)
        
        # Subset measurements by the specified time window
        measurements = measurements.loc[start_date : end_date]
        measurements = measurements.dropna(axis=1)
        measurements = measurements.loc[:, ~(measurements == 0.0).all(axis=0).values]
        measurements = measurements.loc[:, ~measurements.columns.duplicated()].copy()
        measurements = measurements.resample(f"{dt}s").mean().interpolate().ffill().bfill()

        return model_collection, measurements, inputs, dt

    def prepare_model(self, model_collection, measurements, dt, config):
        """
        Prepares the model_collection by binding callback

        Parameters:
            model_collection: dict
                model collection object that gets built using the network info
            measurements: df
                measurements dataframe
            dt: int | float
                internal time step of the model
            config: dict
        """
        for model in model_collection.models.values():
            model_sites = [
                reach_id for reach_id in model.reach_ids if reach_id in measurements.columns
            ]
            if model_sites:
                basin_measurements = measurements[model_sites]
                Q_cov = 1.0 * np.eye(model.n)
                R_cov = 0.0 * np.eye(basin_measurements.shape[1])
                P_t_init = Q_cov.copy()
                kf = KalmanFilter(model, basin_measurements, Q_cov, R_cov, P_t_init)
                model.bind_callback(kf, key="kf")

        for model in model_collection.models.values():
            outlet = model.startnodes[model.startnodes == model.endnodes].item()
            model.set_transmissive_boundary(outlet)

        timedelta = pd.to_timedelta(dt, unit="s")

        for model_name, model in model_collection.models.items():
            #if running KF conventionally, push back the start time by timedelta
            if not self.with_troute:
                model.datetime = pd.to_datetime(config["start_date"]) - timedelta
            else:
                model.datetime = pd.to_datetime(config["start_date"])
        return model_collection

    def initialize(self, config_file="KF.yaml"):
        """
        initializes the entire model by binding all the helper functions.
        
        Parameters:
            config_file: str
                path to the yaml file that helps to initialize the model.
        """
        self.config_file = config_file
        self.config = self.load_config(config_file)

        # Parse with_troute parameter (defaults to False if not provided)
        with_troute_str = str(self.config.get("with_troute", "false")).strip().lower()
        self.with_troute = with_troute_str in ["true", "1", "yes", "t"]

        (
            self.model_collection,
            self.measurements,
            self.inputs,
            self.dt,
        ) = self.load_and_build_inputs(self.config)
        
        self.model_collection = self.prepare_model(
            self.model_collection,
            self.measurements,
            self.dt,
            self.config
        )
        self.simulation = AsyncSimulation(self.model_collection, self.inputs, with_troute=self.with_troute)
        self.outputs_da = None
        self.input_var_store = {name: None for name in self._INPUT_VAR_NAMES}
        self.output_var_store = {name: None for name in self._OUTPUT_VAR_NAMES}

        self._start_datetime = self.model_collection.datetime
        self._start_time = 0.0
        self._current_time = self._start_time
        self._time_step = float(self.model_collection.timedelta.total_seconds())
        self._end_time = float("inf")
        self._initialized = True
        self._sorted_indices = None

    def update(self):
        #update the model for the internal time step of the model
        self.update_until(time_window=self.model_collection.timedelta.total_seconds())

    def update_until(self, time_window):
        """Advance KF by ``time_window`` seconds.
        Parameters
        ----------
        time_window : int
            time window you want the model to slide into future
        """
        self._require_initialized()
        model_timestep = self.model_collection.timedelta.total_seconds()

        #if the time window provided by the user is not the perfect multiple
        #of model_timestep (dt), raise error
        if time_window % model_timestep != 0:
            raise ValueError(
                "time_window must be expressible as a multiple of "
                f"model_collection.timedelta. {model_timestep}"
            )

        #if it is a perfect multiple, route that many time steps.
        n_steps = int(time_window / model_timestep)
        for _ in range(n_steps):
            outputs_da = self._run_async(
                self.simulation.simulate(only_one_time_step=True)
            )
            self.outputs_da = pd.concat(
                [series for series in outputs_da.values()],
                axis=1,
            )
            self.output_var_store["reach_list"] = self.outputs_da.columns

            self.output_var_store["discharge"] = self.outputs_da.loc[
                    self.model_collection.datetime
                ].values

        #sort the reach_list once and apply everytime when updating
        if self._sorted_indices is None:
            self._sorted_indices = np.argsort(self.output_var_store["reach_list"])
            
        self.output_var_store["reach_list"] = self.output_var_store["reach_list"][self._sorted_indices]
        self.output_var_store["discharge"] = self.output_var_store["discharge"][self._sorted_indices]

        self._current_time += time_window

    def finalize(self):
        self._require_initialized()
        self._initialized = False

    def get_component_name(self):
        return "TX Fast Hydrology BMI"

    def get_input_item_count(self):
        return len(self.input_var_store)

    def get_output_item_count(self):
        return len(self.output_var_store)

    def get_start_time(self):
        return self._start_time

    def get_end_time(self):
        return self._end_time

    def get_start_datetime(self):
        return self._start_datetime

    def get_current_datetime(self):
        return self.model_collection.datetime
    
    def get_current_time(self):
        return self._current_time

    def get_time_step(self):
        return self._time_step

    def get_time_units(self):
        return "s"

    def get_input_var_names(self):
        return tuple(self.input_var_store.keys())

    def get_output_var_names(self):
        return tuple(self.output_var_store.keys())

    def get_var_type(self, name):
        self._validate_name(name)
        return type(self.input_var_store[name])

    def get_var_units(self, name):
        return NotImplementedError

    def get_var_itemsize(self, name):
        return NotImplementedError

    def get_var_nbytes(self, name):
        return NotImplementedError

    def get_var_location(self, name):
        return NotImplementedError

    def get_var_grid(self, name):
        return NotImplementedError

    def get_grid_rank(self, grid):
        return NotImplementedError

    def get_grid_size(self, grid):
        return NotImplementedError

    def get_grid_type(self, grid):
        return NotImplementedError

    def get_grid_shape(self, grid, shape):
        return NotImplementedError

    def get_grid_spacing(self, grid, spacing):
        return NotImplementedError

    def get_grid_origin(self, grid, origin):
        return NotImplementedError

    def get_grid_node_count(self, grid):
        return NotImplementedError

    def get_grid_edge_count(self, grid):
        return NotImplementedError

    def get_grid_face_count(self, grid):
        return NotImplementedError

    def get_grid_x(self, grid, x):
        return NotImplementedError

    def get_grid_y(self, grid, y):
        return NotImplementedError

    def get_grid_z(self, grid, z):
        return NotImplementedError

    def get_grid_edge_nodes(self, grid, edge_nodes):
        return NotImplementedError

    def get_grid_face_edges(self, grid, face_edges):
        return NotImplementedError

    def get_grid_face_nodes(self, grid, face_nodes):
        return NotImplementedError

    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        return NotImplementedError

    def get_value(self, name, dest):
        self._validate_output_name(name)
        dest[:] = self.output_var_store[name]
        return dest

    def get_value_ptr(self, name):
        self._validate_output_name(name)
        return self.output_var_store[name]

    def get_value_at_indices(self, name, dest, inds):
        return NotImplementedError

    def set_value(self, name, src):
        self._validate_input_name(name)
        self.input_var_store[name] = src

    def update_k_x_discharge(self, index_crosswalk_cache):
        """
        This updates k, x, and discharge for the individual model of model collection
        with the k, x, and discharge set by the set_value function (primarily from t-route).

        This function should only be used if the self.with_troute is True because the index crosswalking
        is between the reach_list of BMI T-route and BMI KF.
        """

        if self.input_var_store["k"] is None or self.input_var_store["x"] is None or self.input_var_store["initial_discharge"] is None:
            raise ValueError("Cant update the value when one of the input variables is None")
        else:
            for model_name, model in self.model_collection.models.items():
                #the input_var_store stores information according to the
                #sorted reach_ids list. To map it back to how model accepts it
                #index_crosswalk_cache is required
                model.K = self.input_var_store["k"][index_crosswalk_cache[model_name]]
                model.X = self.input_var_store["x"][index_crosswalk_cache[model_name]]
                model.o_next = self.input_var_store["initial_discharge"][index_crosswalk_cache[model_name]]

                #update the msukingum coefficients
                model.compute_muskingum_coeffs(model.K, model.X)


    def set_value_at_indices(self, name, inds, src):
        return NotImplementedError

    @staticmethod
    def _run_async(coro):
        """
        Handles async computation
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result = {}
        error = {}

        def runner():
            try:
                result["value"] = asyncio.run(coro)
            except BaseException as exc:
                error["value"] = exc

        thread = threading.Thread(target=runner)
        thread.start()
        thread.join()
        if error:
            raise error["value"]
        return result["value"]

    def _require_initialized(self):
        if not getattr(self, "_initialized", False):
            raise RuntimeError("BmiKF has not been initialized.")

    def _validate_name(self, name):
        if name not in self.input_var_store and name not in self.output_var_store:
            raise KeyError(f"Unknown BMI variable: {name}")

    def _validate_input_name(self, name):
        if name not in self.input_var_store:
            raise KeyError(f"Unknown BMI input variable: {name}")

    def _validate_output_name(self, name):
        if name not in self.output_var_store:
            raise KeyError(f"Unknown BMI output variable: {name}")


BmiTxFastHydrology = BmiKF