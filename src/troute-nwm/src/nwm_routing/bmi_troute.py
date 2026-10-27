"""
BMI wrapper for the NWM t-route driver.

This class exposes t-route through BMI. Initialization performs the same
setup as ``main_v04``; each ``update_until`` call routes for the requested
time window and advances the BMI clock by that amount.

Written and tested by: Sonam Lama (slama@ua.edu)
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from troute.NHDNetwork import NHDNetwork
from troute.HYFeaturesNetwork import HYFeaturesNetwork
from troute.DataAssimilation import DataAssimilation
import troute.hyfeature_network_utilities as hnu
import troute.nhd_io as nhd_io

from .input import _input_handler_v04
from .output import nwm_output_generator
from .__main__ import _handle_args_v03, nwm_route


LOG = logging.getLogger("")


class BmiTroute:
    """Basic Model Interface adapter for t-route NWM routing."""

    _INPUT_VAR_NAMES = ("initial_discharge",)
    _OUTPUT_VAR_NAMES = ("reach_list", "discharge", "k", "x")

    def initialize(self, config_file=""):
        """
        Initialize t-route from the same YAML config file used by ``main_v04``.

        Parameters
        ----------
        config_file : str
            Path to the t-route v04 configuration file.
        """
        # if not config_file:
        #     raise ValueError("BmiTroute.initialize requires a t-route config file.")

        self.config_file = config_file
        self.args = _handle_args_v03(["-f", config_file])

        (
            self.log_parameters,
            self.preprocessing_parameters,
            self.supernetwork_parameters,
            self.waterbody_parameters,
            self.compute_parameters,
            self.forcing_parameters,
            self.restart_parameters,
            self.hybrid_parameters,
            self.output_parameters,
            self.parity_parameters,
            self.data_assimilation_parameters,
        ) = _input_handler_v04(self.args)

        self.output_parameters = self.output_parameters or {}
        self.parity_parameters = self.parity_parameters or {}
        self.data_assimilation_parameters = self.data_assimilation_parameters or {}
        self.compute_parameters = self.compute_parameters or {}

        # Force return_courant to True regardless of config file
        self.compute_parameters["return_courant"] = True

        self.run_parameters = {
            "dt": self.forcing_parameters.get("dt"),
            "nts": self.forcing_parameters.get("nts"),
            "cpu_pool": self.compute_parameters.get("cpu_pool"),
        }

        self.cpu_pool = self.compute_parameters.get("cpu_pool", None)
        self.giuh_node = self.compute_parameters.get("giuh_node")

        self._initialize_max_loop_size()
        self._initialize_network()
        self._initialize_run_sets()
        self._initialize_compute_fields()
        self._initialize_kernel_log()

        self.subnetwork_list = [None, None, None]
        self.firstRun = bool(self.kernelTalks)
        self.run_results = None
        self.input_var_store = {name: None for name in self._INPUT_VAR_NAMES}
        self.output_var_store = {name: None for name in self._OUTPUT_VAR_NAMES}

        self._start_datetime = self.network.t0
        self._start_time = 0.0
        self._current_time = self._start_time
        self._end_time = float("inf")
        self._time_step = 900.0 #every 15 minutes
        self._sorted_indices = None
        self._initialized = True

    def update(self):
        """Advance t-route by the default BMI update window (15 minutes)."""
        self.update_until(time_window=900)

    def update_until(self, time_window):
        """Advance t-route by ``time_window`` seconds."""
        self._require_initialized()
        t0 = self.network.t0
        dt = self.forcing_parameters.get("dt")
        
        # Check if time_window is a multiple of dt
        if time_window % dt != 0:
            raise ValueError(f"time_window ({time_window}) must be a multiple of dt ({dt}).")

        nts = int(time_window / dt)

        run_results = nwm_route(
            self.network.connections,
            self.network.reverse_network,
            self.network.waterbody_connections,
            self.network.reaches_by_tailwater,
            self.parallel_compute_method,
            self.compute_kernel,
            self.subnetwork_target_size,
            self.cpu_pool,
            self.network.t0,
            dt,
            nts,
            self.qts_subdivisions,
            self.network.independent_networks,
            self.network.dataframe,
            self.network.q0,
            self.network._qlateral,
            self.data_assimilation.usgs_df,
            self.data_assimilation.lastobs_df,
            self.data_assimilation.reservoir_usgs_df,
            self.data_assimilation.reservoir_usgs_param_df,
            self.data_assimilation.reservoir_usace_df,
            self.data_assimilation.reservoir_usace_param_df,
            self.data_assimilation.reservoir_rfc_df,
            self.data_assimilation.reservoir_rfc_param_df,
            self.data_assimilation.great_lakes_df,
            self.data_assimilation.great_lakes_param_df,
            self.network.great_lakes_climatology_df,
            self.data_assimilation.assimilation_parameters,
            self.assume_short_ts,
            self.return_courant,
            self.network.waterbody_dataframe,
            self.data_assimilation_parameters,
            self.network.waterbody_types_dataframe,
            self.network.waterbody_type_specified,
            self.network.diffusive_network_data,
            self.network.topobathy_df,
            self.network.refactored_diffusive_domain,
            self.network.refactored_reaches,
            self.subnetwork_list,
            self.network.coastal_boundary_depth_df,
            self.network.unrefactored_topobathy_df,
            self.firstRun,
            self.logFileName,
            giuh_node=self.giuh_node,
        )

        self.subnetwork_list = run_results[1]
        self.run_results = run_results[0]

        self.network.new_q0(self.run_results)
        self.network.update_waterbody_water_elevation()
        self.data_assimilation.update_after_compute(self.run_results, time_window)

        #this updates the output variable that can be referenced by the object
        self.update_output_var_store(self.run_results)

        if self.output_parameters.get("lite_restart") is not None:
            nhd_io.write_lite_restart(
                self.network.q0,
                self.network._waterbody_df,
                t0 + timedelta(seconds=time_window),
                self.output_parameters["lite_restart"],
            )

        if self.network.poi_nex_dict:
            self.poi_crosswalk = self.network.poi_nex_dict
        else:
            self.poi_crosswalk = dict()

        #this writes to a file, which is not needed for BMI, so commenting out for now
        # self._write_output(self._build_output_run(t0, nts))

        self.network.t0 = self.network.t0 + timedelta(seconds=time_window)
        self._current_time += time_window
        self.firstRun = False

    def finalize(self):
        """Finalize the BMI component."""
        self._require_initialized()
        self._initialized = False

    def get_component_name(self):
        return "t-route NWM BMI"

    def get_input_item_count(self):
        return len(self.input_var_store)

    def get_output_item_count(self):
        return len(self.output_var_store)

    def get_start_time(self):
        return self._start_time

    def get_end_time(self):
        return self._end_time

    def get_current_time(self):
        return self._current_time

    def get_time_step(self):
        return self._time_step

    def get_time_units(self):
        return "s"

    def get_start_datetime(self):
        return pd.Timestamp(self._start_datetime).tz_localize('UTC')

    def get_current_datetime(self):
        return pd.Timestamp(self.network.t0).tz_localize('UTC')
    
    def get_input_var_names(self):
        return tuple(self.input_var_store.keys())

    def get_output_var_names(self):
        return tuple(self.output_var_store.keys())

    def get_var_type(self, name):
        self._validate_name(name)
        return type(self.input_var_store[name])

    def get_var_units(self, name):
        self._validate_name(name)
        return NotImplementedError

    def get_var_itemsize(self, name):
        self._validate_name(name)
        return NotImplementedError

    def get_var_nbytes(self, name):
        return NotImplementedError

    def get_var_location(self, name):
        self._validate_name(name)
        return NotImplementedError

    def get_var_grid(self, name):
        self._validate_name(name)
        return NotImplementedError

    def get_grid_rank(self, grid):
        self._validate_grid(grid)
        return NotImplementedError

    def get_grid_size(self, grid):
        self._validate_grid(grid)
        return NotImplementedError

    def get_grid_type(self, grid):
        self._validate_grid(grid)
        return NotImplementedError

    def get_grid_shape(self, grid, shape):
        self._validate_grid(grid)
        shape[:] = [1]
        return NotImplementedError

    def get_grid_spacing(self, grid, spacing):
        self._validate_grid(grid)
        spacing[:] = [1.0]
        return NotImplementedError

    def get_grid_origin(self, grid, origin):
        self._validate_grid(grid)
        origin[:] = [0.0]
        return NotImplementedError

    def get_grid_node_count(self, grid):
        self._validate_grid(grid)
        return NotImplementedError

    def get_grid_edge_count(self, grid):
        self._validate_grid(grid)
        return NotImplementedError

    def get_grid_face_count(self, grid):
        self._validate_grid(grid)
        return NotImplementedError

    def get_grid_x(self, grid, x):
        self._validate_grid(grid)
        x[:] = [0.0]
        return NotImplementedError

    def get_grid_y(self, grid, y):
        self._validate_grid(grid)
        y[:] = [0.0]
        return NotImplementedError

    def get_grid_z(self, grid, z):
        self._validate_grid(grid)
        z[:] = [0.0]
        return NotImplementedError

    def get_grid_edge_nodes(self, grid, edge_nodes):
        self._validate_grid(grid)
        return NotImplementedError

    def get_grid_face_edges(self, grid, face_edges):
        self._validate_grid(grid)
        return NotImplementedError

    def get_grid_face_nodes(self, grid, face_nodes):
        self._validate_grid(grid)
        return NotImplementedError

    def get_grid_nodes_per_face(self, grid, nodes_per_face):
        self._validate_grid(grid)
        return NotImplementedError

    def get_value(self, name, dest):
        self._validate_output_name(name)
        dest[:] = self.output_var_store[name]
        return dest

    def get_value_ptr(self, name):
        self._validate_output_name(name)
        return (self.output_var_store[name])

    def get_value_at_indices(self, name, dest, inds):
        # self._validate_output_name(name)
        # dest[:] = self.output_var_store[name][inds]
        return NotImplementedError

    def update_q0(self):
        #update the qu0 and qd0 with improved discharge
        #associated with sorted reach_list
        if self.input_var_store["initial_discharge"] is not None:
            qu0 = self.input_var_store["initial_discharge"]
            qd0 = self.input_var_store["initial_discharge"]
        else:
            raise ValueError("Discharge has not been updated with improved discharge")

        #the dataframe also has height that needs to be sorted too
        if self._sorted_indices is not None:
            h0 = self.network.q0['h0'].values[self._sorted_indices]
        else:
            raise ValueError("The indices have not been sorted yet. This indicates routing has not been done ever before.")

        #sorted reach_list
        if self.output_var_store["reach_list"] is not None:
            reach_list = self.output_var_store["reach_list"]
        else:
            raise ValueError("This indicates routing has not been done ever before.")

        if self.network.q0 is not None:
            self.network.q0.index = reach_list
            self.network.q0["h0"] = h0
            self.network.q0["qu0"] = qu0
            self.network.q0["qd0"] = qd0


    def set_value(self, name, src):
        self._validate_input_name(name)
        self.input_var_store[name] = src

        #trigger the update of q0 dataframe 
        self.update_q0()

    def set_value_at_indices(self, name, inds, src):
        return NotImplementedError
        # self._validate_input_name(name)
        # value = self._as_object_array(self.input_var_store[name])
        # value[inds] = src
        # self.input_var_store[name] = value[0]

    def update_output_var_store(self, run_results):
        discharge = []
        k = []
        x = []
        reach_list = []

        discharge = np.concatenate([result[1][:, -3] for result in run_results])

        k = np.concatenate([result[11][:, -2] for result in run_results])

        x = np.concatenate([result[11][:, -1] for result in run_results])

        reach_list = np.concatenate([result[0] for result in run_results])

        #this gives us a cache to sort index, so we don't have to sort for every time stamp
        if self._sorted_indices is None:
            self._sorted_indices = np.argsort(reach_list)

        #since the change is internal, bypassing the check rule and sorting as well
        self.output_var_store["discharge"] = discharge[self._sorted_indices]
        self.output_var_store["k"] = k[self._sorted_indices]
        self.output_var_store["x"] = x[self._sorted_indices]
        self.output_var_store["reach_list"] = reach_list[self._sorted_indices]

    def _initialize_network(self):
        if self.supernetwork_parameters["network_type"] == "HYFeaturesNetwork":
            self.network = HYFeaturesNetwork(
                self.supernetwork_parameters,
                self.waterbody_parameters,
                self.data_assimilation_parameters,
                self.restart_parameters,
                self.compute_parameters,
                self.forcing_parameters,
                self.hybrid_parameters,
                self.preprocessing_parameters,
                self.output_parameters,
                verbose=True,
                showtiming=self.log_parameters.get("showtiming", None),
            )
            self.duplicate_ids_df = self.network._duplicate_ids_df
        elif self.supernetwork_parameters["network_type"] == "NHDNetwork":
            self.network = NHDNetwork(
                self.supernetwork_parameters,
                self.waterbody_parameters,
                self.restart_parameters,
                self.forcing_parameters,
                self.compute_parameters,
                self.data_assimilation_parameters,
                self.hybrid_parameters,
                self.output_parameters,
                verbose=True,
                showtiming=self.log_parameters.get("showtiming", None),
            )
            self.duplicate_ids_df = pd.DataFrame()
        else:
            raise ValueError(
                "Unsupported network_type: "
                f"{self.supernetwork_parameters['network_type']}"
            )

    def _initialize_max_loop_size(self):
        nts = self.forcing_parameters.get("nts")
        qts_subdivisions = self.forcing_parameters.get("qts_subdivisions", 1)
        self.forcing_parameters["max_loop_size"] = int(nts / qts_subdivisions)

    def _initialize_run_sets(self):
        self.run_sets = self.network.build_forcing_sets()
        self.da_sets = hnu.build_da_sets(
            self.data_assimilation_parameters,
            self.run_sets,
            self.network.t0,
        )

        self.network.assemble_forcings(self.run_sets[0])

        self.data_assimilation = DataAssimilation(
            self.network,
            self.data_assimilation_parameters,
            self.run_parameters,
            self.waterbody_parameters,
            from_files=True,
            value_dict=None,
            da_run=self.da_sets[0],
        )

    def _initialize_compute_fields(self):
        self.parallel_compute_method = self.compute_parameters.get(
            "parallel_compute_method", None
        )
        self.subnetwork_target_size = self.compute_parameters.get(
            "subnetwork_target_size", 1
        )
        self.qts_subdivisions = self.forcing_parameters.get("qts_subdivisions", 1)
        self.compute_kernel = self.compute_parameters.get("compute_kernel", "V02-caching")
        self.assume_short_ts = self.compute_parameters.get("assume_short_ts", False)
        self.return_courant = self.compute_parameters.get("return_courant", False)

    def _initialize_kernel_log(self):
        self.logFileName = "NONE"
        self.kernelTalks = self.log_parameters.get("log_directory", None)

        if not self.kernelTalks:
            return

        self.logFileName = self.kernelTalks / "kernelTalks.log"
        with open(self.logFileName, "w") as preRunLog:
            preRunLog.write("************************************************************\n")
            preRunLog.write("Pre- and post run parameter and run statistics output file. \n")
            preRunLog.write("************************************************************\n\n")
            preRunLog.write("-----\n")

            if self.restart_parameters["lite_channel_restart_file"] is None:
                output_string = "No channel restart file: cold start."
            else:
                output_string = (
                    "Warmstart - restart file: "
                    + str(self.restart_parameters["lite_channel_restart_file"])
                )
            preRunLog.write(output_string + "\n")
            LOG.info(output_string)

            if self.restart_parameters["lite_waterbody_restart_file"] is None:
                output_string = "No waterbody restart file."
            else:
                output_string = (
                    "Waterbody restart file: "
                    + str(self.restart_parameters["lite_waterbody_restart_file"])
                )
            preRunLog.write(output_string + "\n")
            LOG.info(output_string)

            preRunLog.write("-----\n\n")

    def _build_output_run(self, t0, nts):
        run = self.run_sets[0]
        run["t0"] = t0
        run["nts"] = nts
        return run

    def _write_output(self, run):
        nwm_output_generator(
            run,
            self.run_results,
            self.supernetwork_parameters,
            self.output_parameters,
            self.parity_parameters,
            self.restart_parameters,
            {},
            self.qts_subdivisions,
            self.compute_parameters.get("return_courant", False),
            self.cpu_pool,
            self.network.waterbody_dataframe,
            self.network.waterbody_types_dataframe,
            self.duplicate_ids_df,
            self.data_assimilation_parameters,
            self.data_assimilation.lastobs_df,
            self.network.link_gage_df,
            self.network.link_lake_crosswalk,
            self.network.nexus_dict,
            self.poi_crosswalk,
            self.logFileName,
        )

    def _require_initialized(self):
        if not getattr(self, "_initialized", False):
            raise RuntimeError("BmiTroute has not been initialized.")

    def _validate_name(self, name):
        if name not in self.input_var_store and name not in self.output_var_store:
            raise KeyError(f"Unknown BMI variable: {name}")

    def _validate_input_name(self, name):
        if name not in self.input_var_store:
            raise KeyError(f"Unknown BMI input variable: {name}")

    def _validate_output_name(self, name):
        if name not in self.output_var_store:
            raise KeyError(f"Unknown BMI output variable: {name}")

    @staticmethod
    def _validate_grid(grid):
        return NotImplementedError


BmiTrouteModel = BmiTroute