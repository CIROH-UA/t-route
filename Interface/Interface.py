import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import copy

# Setup system paths so Python can locate the modules
sys.path.append(os.path.abspath("../../src/troute-nwm/src"))

from nwm_routing.bmi_troute import BmiTroute
from tx_fast_hydrology.bmi_KF import BmiKF

class Interface:
    def __init__(self, yaml_file_path_troute, yaml_file_path_KF):
        self.yaml_file_path_troute = Path(yaml_file_path_troute)
        self.yaml_file_path_KF = Path(yaml_file_path_KF)

        self.troute_model = BmiTroute()
        self.KF_model = BmiKF()

        self.index_crosswalk_cache = None

        self._time_history = []
        self._discharge_history = []
        self._reach_columns = None

        self._initialize_models()
        self._verify_compatibility()

    def _initialize_models(self):
        original_dir = os.getcwd()

        try:
            os.chdir(self.yaml_file_path_troute.parent)
            self.troute_model.initialize(self.yaml_file_path_troute.name)
        finally:
            os.chdir(original_dir)

        init_measurements = self.troute_model.get_value_ptr("discharge_obs")
        if isinstance(init_measurements, pd.DataFrame):
            init_measurements = convert_df_for_kf(init_measurements)
            # first timeseries obs data, needed to initialize kf
            init_measurements = init_measurements.iloc[0:1]
        else:
            raise TypeError("Passed USGS data must be a pandas dataframe")

        try:
            os.chdir(self.yaml_file_path_KF.parent)
            self.KF_model.initialize(init_measurements, self.yaml_file_path_KF.name)
        finally:
            os.chdir(original_dir)

    def _verify_compatibility(self):
        dt_troute = self.troute_model.get_time_step()
        dt_kf = self.KF_model.get_time_step()
        if dt_troute != dt_kf:
            raise ValueError(f"Time step mismatch: Troute ({dt_troute}) vs KF ({dt_kf})")

        start_troute = self.troute_model.get_start_datetime()
        start_kf = self.KF_model.get_start_datetime()

        if start_troute != start_kf:
            raise ValueError(f"Start datetime mismatch: Troute ({start_troute}) vs KF ({start_kf})")

    def _build_cache(self):
        sorted_reach_list = self.troute_model.get_value_ptr("reach_list")
        self.index_crosswalk_cache = {}

        for model_name, model in self.KF_model.model_collection.models.items():
            idx_map = np.searchsorted(sorted_reach_list, model.reach_ids)
            self.index_crosswalk_cache[model_name] = idx_map

    def update(self):
        # 1. Update troute model first
        self.troute_model.update()

        # 2. Build the crosswalk cache if this is the very first update
        if self.index_crosswalk_cache is None:
            self._reach_columns = self.troute_model.get_value_ptr("reach_list")
            self._build_cache()


        # 3. Pass values from Troute -> KF
        self.KF_model.set_value("k", self.troute_model.get_value_ptr("k"))
        self.KF_model.set_value("x", self.troute_model.get_value_ptr("x"))
        self.KF_model.set_value("initial_discharge", self.troute_model.get_value_ptr("discharge"))

        discharge_obs = self.troute_model.get_value_ptr("discharge_obs")
        if isinstance(discharge_obs, pd.DataFrame):
            if not discharge_obs.empty:
                discharge_obs = convert_df_for_kf(discharge_obs)
                discharge_obs = discharge_obs.iloc[0:1] # just pass one timestep
            else:
                raise ValueError("Passed USGS dataframe from t-route cannot be empty")
        else:
            raise TypeError("Passed USGS data must be a pandas dataframe")
        self.KF_model.set_value("discharge_obs", discharge_obs)
        print("setting this measurements df in KF")
        print(discharge_obs.head())

        # 4. KF parses sorted arrays into unsorted clusters using the cache
        self.KF_model.update_k_x_discharge(self.index_crosswalk_cache)

        # 5. update measurements in KF
        self.KF_model.update_measurements()

        # 6. Update KF model
        self.KF_model.update()

        discharge = self.KF_model.get_value_ptr("discharge")

        # 7. Pass the computed discharge from KF -> Troute for the next time step
        self.troute_model.set_value("initial_discharge", discharge)

    def warmup_with_KF(self, num_steps=96):
        """Prepare the t-route model for simulation by exchanging information between t-route and KF for a specified number of warmup steps."""
        for _ in range(num_steps):
            self.update()

    def warmup_without_KF(self, num_steps=96):
        """Prepare the t-route model for simulation by updating it without involving the KF model for a specified number of warmup steps."""
        for _ in range(num_steps):
            self.troute_model.update()
            # 2. Build the crosswalk cache if this is the very first update
            if self.index_crosswalk_cache is None:
                self._reach_columns = self.troute_model.get_value_ptr("reach_list")
                self._build_cache()

    def forecast_only(self, num_steps=96):
        troute_model_forecast = copy.deepcopy(self.troute_model)

        #output the first time step as well
        self._time_history.append(troute_model_forecast.get_current_datetime())
        discharge = troute_model_forecast.get_value_ptr("discharge")
        self._discharge_history.append(discharge.copy())


        """Update the t-route model for a specified number of steps without involving the KF model."""
        for _ in range(num_steps):
            troute_model_forecast.update()
            # 6. Save the forecast outputs to the history lists for later retrieval
            current_time = troute_model_forecast.get_current_datetime()

            self._time_history.append(current_time)

            discharge = troute_model_forecast.get_value_ptr("discharge")
            self._discharge_history.append(discharge.copy())

        # discharge = troute_model_forecast.get_value_ptr("discharge")
        del troute_model_forecast  # Clean up the forecast model to free memory


    def get_results_dataframe(self):
        """Constructs and returns the DataFrame on-demand after the forecast simulation."""
        if not self._time_history:
            return pd.DataFrame()

        return pd.DataFrame(
            self._discharge_history,
            index=self._time_history,
            columns=self._reach_columns
        )

def convert_df_for_kf(discharge_obs):
    discharge_obs = discharge_obs.T
    discharge_obs = discharge_obs.astype("float64")
    discharge_obs.index = pd.to_datetime(discharge_obs.index, utc=True)
    discharge_obs.columns.name = None
    discharge_obs.columns = discharge_obs.columns.astype(str)
    discharge_obs = discharge_obs.dropna(axis=1, how="all")

    return discharge_obs