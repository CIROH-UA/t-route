import os
import uuid
import copy
import logging
import json
from scipy.sparse import lil_matrix, csgraph
import numpy as np
import pandas as pd
import datetime
from tx_fast_hydrology.nutils import interpolate_sample, _ax_bu
from tx_fast_hydrology.callbacks import BaseCallback
from tx_fast_hydrology.io import ModelDecoder, ModelEncoder
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL

MIN_SLOPE = 1e-8

DEFAULT_START_TIME = pd.to_datetime(0., utc=True)
DEFAULT_TIMEDELTA = pd.to_timedelta(3600, unit='s')

class Muskingum:
    """
    Base class for implementing Muskingum routing.

    Inputs:
    -------
    data : dict
        JSON-like object containing data needed to instantiate the Muskingum model.
        {
            'name' : (str)
                The name of the model instance.
            'datetime' : (pandas.DateTime)
                The starting timestamp of the model.
            'timedelta' : (pandas.TimeDelta)
                The timestep of the model.
            'reach_ids' : (list, dtype str)
                The identifiers for each reach (e.g. NWM COMIDS)
            'startnodes' : (numpy.ndarray, dtype int64)
                The numeric index of each reach.
            'endnodes' : (np.ndarray, dtype int64)
                The index of the reaches downstream of each reach in `startnodes`.
            'K' : (np.ndarray, dtype float64)
                Muskingum travel time parameters for each reach [s].
            'X' : (np.ndarray, dtype float64)
                Muskingum attenuation parameters for each reach [unitless].
            'o_t' : (np.ndarray, dtype float64) [optional]
                Starting outflows from each reach at time `datetime` [m^3 / s].
            'dx' : (np.ndarray, dtype float64) [optional]
                Length of each reach [m].
            'paths' : (list of lists) [optional]
                Geometric data for plotting the stream network
        }
    
    load_optional : bool
        If True, load optional paramters `o_t`, `dx`, and `paths`. 
    
    create_state_space : bool
        If True, populate matrices A and B for the state space equation:
            x_t+1 = A @ x_t + B @ u_t

    sparse: bool
        If True, use sparse matrices for the state space equation and covariance matrices.

    Attributes:
    -----------
    States:

    o_t_next : np.ndarray, dtype float64
        Outflows from each reach at current timestep [m^3 / s]
    i_t_next : np.ndarray, dtype float64
        Inflows to each reach at current timestep [m^3 / s]
    o_t_prev : np.ndarray, dtype float64
        Outflows from each reach at previous timestep [m^3 / s]
    i_t_prev : np.ndarray, dtype float64
        Inflows to each reach at previous timestep [m^3 / s]

    Parameters:

    n : int
        Number of reaches
    datetime : pd.DateTime
        Current timestamp of the model in UTC time.
    timedelta : pd.TimeDelta
        Timestep of the model as a timedelta object.
    dt : float
        Timestep of the model in seconds.
    K : np.ndarray, dtype float64
        Muskingum travel time paramters for each reach [s].
    X : np.ndarray, dtype float64
        Muskingum attenuation parameters for each reach [unitless].
    dx : (np.ndarray, dtype float64)
        Length of each reach [m].
    paths : (list of lists)
        Geometric data for plotting the stream network

    Structures:

    sinks : dict of dicts
        Describes downstream connectivity of model. Each internal dict is structed as follows:
        {
            model : Muskingum instance
                Downstream model that model instance feeds into.
            exit_node : int
                Index of the reach at which flow leaves the model. 
        }

    sources : dict of dicts
        Describes upstream connectivity of model. Each internal dict is structed as follows:
        {
            model : Muskingum instance
                Upstream model that feeds into model instance.
            entry_node : int
                Index of the reach at which flow enters the model. 
        }

    saved_states : dict
        Saved states from current or previous timestep.

    callbacks : dict
        A dictionary of callback functions that may be applied to methods like step and simulate.

    A : np.ndarray, dtype float64
        State-space state transition matrix A.

    B : np.ndarray, dtype float64
        State-space input matrix B.

    Methods:
    -----------
    step : Steps model forward in time by one timestep.
    simulate : Simulates model by stepping forward in time over a given time range.
    save_state : Saves current state to an internal dictionary.
    load_state : Loads state from internal dictionary.
    plot : Plots stream network using matplotlib.
    copy : Returns a copy of the model instance using copy.deepcopy.
    bind_callback : Bind a callback to the model.
    unbind_callback : Remove a callback from the model.
    split : Splits the model into multiple interconnected submodels.
    """
    def __init__(self, data, load_optional=True, create_state_space=False, sparse=False):
        self.sparse = sparse
        self.callbacks = {}
        self.saved_states = {}
        self.sinks = []
        self.sources = []
        # Read json input file
        if isinstance(data, dict):
            self.load_model(data, load_optional=load_optional)
        elif isinstance(data, str):
            self.load_model_file(data, load_optional=load_optional)
        else:
            raise TypeError('`data` must be a file path or dictionary.')
        # Create logger
        self.logger = logging.getLogger(self.name)
        # Create arrays
        n = self.n
        self.o_t_prev = np.zeros(n, dtype=np.float64)
        self.i_t_next = np.zeros(n, dtype=np.float64)
        self.i_t_prev = np.zeros(n, dtype=np.float64)
        self.init_states(o_t_next=self.o_t_next)
        self.o_t_prev[:] = self.o_t_next[:]
        self.i_t_prev[:] = self.i_t_next[:]
        # Initialize state-space matrices
        if sparse:
            self.A = lil_matrix((n, n))
            self.B = lil_matrix((n, n))
        else:
            self.A = np.zeros((n, n), dtype=np.float64)
            self.B = np.zeros((n, n), dtype=np.float64)
        # Compute alpha, beta, and gamma coefficients
        self.alpha = np.zeros(n, dtype=np.float64)
        self.beta = np.zeros(n, dtype=np.float64)
        self.chi = np.zeros(n, dtype=np.float64)
        self.gamma = np.zeros(n, dtype=np.float64)
        self.compute_muskingum_coeffs()
        if create_state_space:
            self.create_state_space()
        self.save_state()

    @property
    def info(self):
        info_dict = {
            'name' : self.name,
            'datetime' : self.datetime,
            'timedelta' : self.timedelta,
            'reach_ids' : self.reach_ids,
            'startnodes' : self.startnodes,
            'endnodes' : self.endnodes,
            'K' : self.K,
            'X' : self.X,
            'o_t' : self.o_t_next,
            'dx' : self.dx,
            'paths' : self.paths
        }
        return info_dict

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        if not isinstance(new_name, str):
            self.logger.warning(f'`name` is not a string, converting to {new_name}')
            new_name = str(new_name)
        self._name = new_name

    @property
    def datetime(self):
        return self._datetime

    @datetime.setter
    def datetime(self, new_datetime):
        try:
            assert isinstance(new_datetime, pd.Timestamp)
        except:
            raise TypeError('New datetime must be of type `pd.Timestamp`')
        try:
            assert new_datetime.tz == datetime.timezone.utc
        except:
            raise ValueError('New datetime must be UTC.')
        self._datetime = new_datetime

    @property
    def timedelta(self):
        return self._timedelta

    @timedelta.setter
    def timedelta(self, new_timedelta):
        try:
            assert isinstance(new_timedelta, pd.Timedelta)
        except:
            raise TypeError('New timedelta must be of type `pd.Timedelta`')
        self._timedelta = new_timedelta

    @property
    def o_t(self):
        return self.o_t_next

    @o_t.setter
    def o_t(self, new_o_t):
        try:
            new_o_t = np.asarray(new_o_t, dtype=np.float64)
        except:
            raise TypeError('New `o_t` must be convertible to float64 np.ndarray')
        self.o_t_next = new_o_t

    @property
    def dt(self):
        dt = float(self.timedelta.seconds)
        return dt

    def load_model(self, obj, load_optional=True):
        required_fields = {'name', 'datetime', 'timedelta', 'reach_ids',
                           'startnodes', 'endnodes', 'K', 'X', 'o_t'}
        optional_fields = {'paths', 'dx'}
        defaults = {'name' : str(uuid.uuid4()), 
                    'datetime' : DEFAULT_START_TIME,
                    'timedelta' : DEFAULT_TIMEDELTA,
                    'dx' : None,
                    'paths' : []}
        # Validate data
        try:
            assert required_fields.issubset(set(obj.keys()))
        except:
            raise ValueError(f'Model field must contain fields {required_fields}')
        try:
            assert isinstance(obj['startnodes'], np.ndarray)
            assert isinstance(obj['endnodes'], np.ndarray)
            assert isinstance(obj['K'], np.ndarray)
            assert isinstance(obj['X'], np.ndarray)
            assert isinstance(obj['o_t'], np.ndarray)
            assert obj['startnodes'].dtype == np.int64
            assert obj['endnodes'].dtype == np.int64
            assert obj['K'].dtype == np.float64
            assert obj['X'].dtype == np.float64
            assert obj['o_t'].dtype == np.float64
        except:
            raise TypeError('Typing of input arrays is incorrect.')
        try:
            assert (obj['startnodes'].size == obj['endnodes'].size 
                    == obj['K'].size == obj['X'].size == obj['o_t'].size)
        except:
            raise ValueError('Arrays are not the same length')
        # If optional fields are desired, add to the set of fields
        if load_optional:
            fields = required_fields.union(optional_fields)
        else:
            fields = required_fields
        # Iterate through fields and add as attributes to class instance
        for field in fields:
            if field in defaults:
                default_value = defaults[field]
                value = obj.setdefault(field, default_value)
            else:
                value = obj[field]
            setattr(self, field, value)
        # Add derived attributes
        startnodes, endnodes = self.startnodes, self.endnodes
        self.n = startnodes.size
        self.indegree = self.compute_indegree(startnodes, endnodes)

    def load_model_file(self, file_path, load_optional=True):
        obj = load_model_file(file_path, load_optional=load_optional)
        self.load_model(obj)
    
    def dump_model_file(self, file_path, dump_optional=True):
        obj = self.info
        return dump_model_file(obj, file_path, dump_optional=dump_optional)

    @classmethod
    def from_model_file(cls, file_path, load_optional=True, **kwargs):
        obj = load_model_file(file_path, load_optional=load_optional)
        newinstance = cls(obj, **kwargs)
        return newinstance

    @classmethod
    def from_nhd_geojson(cls, file_path, **kwargs):
        parsed_data = load_nhd_geojson(file_path)
        newinstance = cls(parsed_data, **kwargs)
        return newinstance

    def compute_indegree(self, startnodes, endnodes):
        self_loops = []
        for i in range(len(startnodes)):
            if endnodes[i] == startnodes[i]:
                self_loops.append(i)
        indegree = np.bincount(endnodes.ravel(), minlength=startnodes.size)
        for self_loop in self_loops:
            indegree[self_loop] -= 1
        return indegree

    def compute_alpha(self, K, X, dt):
        # TODO: Correct order in numerator?
        alpha = (dt - 2 * K * X) / (2 * K * (1 - X) + dt)
        return alpha

    def compute_beta(self, K, X, dt):
        beta = (dt + 2 * K * X) / (2 * K * (1 - X) + dt)
        return beta

    def compute_chi(self, K, X, dt):
        chi = (2 * K * (1 - X) - dt) / (2 * K * (1 - X) + dt)
        return chi

    def compute_gamma(self, K, X, dt):
        gamma = dt / (K * (1 - X) + dt / 2)
        return gamma

    def compute_muskingum_coeffs(self, K=None, X=None, dt=None):
        self.logger.info('Computing Muskingum coefficients...')
        if K is None:
            K = self.K
        if X is None:
            X = self.X
        if dt is None:
            dt = self.dt
        self.alpha[:] = self.compute_alpha(K, X, dt)
        self.beta[:] = self.compute_beta(K, X, dt)
        self.chi[:] = self.compute_chi(K, X, dt)
        self.gamma[:] = self.compute_gamma(K, X, dt)

    def create_state_space(self, overwrite_old=True):
        self.logger.info('Creating state-space system...')
        startnodes = self.startnodes
        endnodes = self.endnodes
        alpha = self.alpha
        beta = self.beta
        chi = self.chi
        gamma = self.gamma
        indegree = self.indegree
        A = self.A
        B = self.B
        if overwrite_old:
            self.A.fill(0.0)
            self.B.fill(0.0)
        startnodes = startnodes[(indegree == 0)]
        A, B = self.muskingum_matrix(
            A, B, startnodes, endnodes, alpha, beta, chi, gamma, indegree
        )
        self.A = A
        self.B = B

    def muskingum_matrix(
        self, A, B, startnodes, endnodes, alpha, beta, chi, gamma, indegree
    ):
        m = startnodes.size
        n = endnodes.size
        indegree_t = indegree.copy()
        for k in range(m):
            startnode = startnodes[k]
            endnode = endnodes[startnode]
            while indegree_t[startnode] == 0:
                alpha_i = alpha[startnode]
                beta_i = beta[startnode]
                chi_i = chi[startnode]
                gamma_i = gamma[startnode]
                alpha_j = alpha[endnode]
                beta_j = beta[endnode]
                A[startnode, startnode] = chi_i
                B[startnode, startnode] = gamma_i
                if startnode != endnode:
                    A[endnode, startnode] += beta_j
                    A[endnode] += alpha_j * A[startnode]
                    B[endnode] += alpha_j * B[startnode]
                indegree_t[endnode] -= 1
                startnode = endnode
                endnode = endnodes[startnode]
        return A, B

    def init_states(self, o_t_next=None, i_t_next=None):
        if o_t_next is None:
            self.o_t_next[:] = 0.
        else:
            self.o_t_next[:] = o_t_next[:]
        if i_t_next is None:
            self.i_t_next[:] = 0.
            np.add.at(self.i_t_next, self.endnodes, self.o_t_next[self.startnodes])
        else:
            self.i_t_next[:] = i_t_next[:]

    def step_matrix(self, p_t_next, num_iter=1, inc_t=False):
        raise NotImplementedError
        A = self.A
        B = self.B
        dt = self.dt
        timedelta = self.timedelta
        o_t_prev = self.o_t_next
        o_t_next = A @ o_t_prev + B @ p_t_next
        self.o_t_next = o_t_next
        self.o_t_prev = o_t_prev
        if inc_t:
            self.t += dt
            self.datetime += self.timedelta

    def step_iter(self, p_t_next, timedelta=None):
        if timedelta is None:
            timedelta = self.timedelta
            dt = self.dt
        else:
            dt = float(timedelta.seconds)
        startnodes = self.startnodes
        endnodes = self.endnodes
        indegree = self.indegree
        sub_startnodes = startnodes[(indegree == 0)]
        o_t_prev = self.o_t_next
        i_t_prev = self.i_t_next
        if dt != self.dt:
            self.logger.warning('Timestep has changed. Recomputing Muskingum coefficients.')
            self.compute_muskingum_coeffs(dt=dt)
        alpha = self.alpha
        beta = self.beta
        chi = self.chi
        gamma = self.gamma
        for _, callback in self.callbacks.items():
            callback.__on_step_start__()
        i_t_next, o_t_next = _ax_bu(sub_startnodes, endnodes, alpha, beta, chi, gamma,
                                    i_t_prev, o_t_prev, p_t_next, indegree)
        self.o_t_next = o_t_next
        self.o_t_prev = o_t_prev
        self.i_t_next = i_t_next
        self.i_t_prev = i_t_prev
        self.datetime += timedelta
        for _, callback in self.callbacks.items():
            callback.__on_step_end__()
        self.logger.debug(f'Stepped to time {self.datetime}')

    def step(self, p_t_next, timedelta=None):
        """
        Advances model forward in time by one timestep, producing new estimates of
        outflows `o_t_next`, and inflows `i_t_next`.

        Inputs
        ------
        p_t_next : np.ndarray, dtype float64
            Array of lateral inputs at each reach (e.g. runoff + recharge)
        timedelta : pd.TimeDelta
            Timestep of step. Defaults to self.timedelta

        Returns:
        --------
        None
        """
        return self.step_iter(p_t_next, timedelta=timedelta)

    def simulate_matrix(self, dataframe, **kwargs):
        raise NotImplementedError
        assert isinstance(dataframe.index, pd.core.indexes.datetimes.DatetimeIndex)
        # assert (dataframe.index.tz == datetime.timezone.utc)
        assert np.in1d(self.reach_ids, dataframe.columns).all()
        dataframe = dataframe[self.reach_ids]
        dataframe.index = dataframe.index.tz_convert("UTC")
        self.datetime = dataframe.index[0]
        self.t = self.datetime.timestamp()
        for index in dataframe.index:
            p_t_next = dataframe.loc[index, :].values
            self.step(p_t_next, **kwargs)
            yield self

    def simulate_iter(self, dataframe, start_time=None, end_time=None, o_t_init=None,
                      only_one_time_step=False, with_troute=False, **kwargs):
        assert isinstance(dataframe.index, pd.core.indexes.datetimes.DatetimeIndex)
        assert (dataframe.index.tz == datetime.timezone.utc)
        assert np.in1d(self.reach_ids, dataframe.columns).all()
        # Set start and end times
        # TODO: Put in checks here
        if end_time is None:
            end_time = dataframe.index.max()
        else:
            try:
                assert isinstance(end_time, pd.Timestamp)
            except:
                raise TypeError('`end_time` must be of type `pd.Timestamp`')
        if start_time is None:
            start_time = self.datetime
        else:
            self.datetime = start_time
            if o_t_init is None:
                raise ValueError('If `start_time` is specified, initial state `o_t_init` must be provided.')
        if o_t_init is not None:
            self.init_states(o_t_next=o_t_init)
            
        # Crop input data to model reaches
        dataframe = dataframe[self.reach_ids]

        if only_one_time_step:
            #if with_troute, only run pre simulation callbacks because that calls KF
            if with_troute:
                for _, callback in self.callbacks.items():
                    callback.__on_simulation_start__()
                return self

            else:
                # Execute pre-simulation callbacks
                if self.datetime < dataframe.index.min():
                    for _, callback in self.callbacks.items():
                        callback.__on_simulation_start__()
                next_timestep = self.datetime + self.timedelta
                p_t_next = interpolate_sample(float(next_timestep.value), 
                                            dataframe.index.astype(int).astype(float).values,
                                            dataframe.values) 
                self.step_iter(p_t_next, **kwargs)

                if self.datetime > end_time:
                    # Execute post-simulation callbacks
                    for _, callback in self.callbacks.items():
                        callback.__on_simulation_end__()
                return self

        def simulate_window():
            # Execute pre-simulation callbacks
            for _, callback in self.callbacks.items():
                callback.__on_simulation_start__()
            while self.datetime < end_time:
                next_timestep = self.datetime + self.timedelta
                p_t_next = interpolate_sample(float(next_timestep.value), 
                                              dataframe.index.astype(int).astype(float).values,
                                              dataframe.values) 
                self.step_iter(p_t_next, **kwargs)
                yield self
            # Execute post-simulation callbacks
            for _, callback in self.callbacks.items():
                callback.__on_simulation_end__()

        return simulate_window()

    def simulate(self, dataframe, start_time=None, end_time=None, o_t_init=None, only_one_time_step=False, with_troute=False, **kwargs):
        """
        Advances model forward in time over a specified time range, producing successive
        new estimates of `o_t_next` and `i_t_next` at each timestep.

        Inputs
        ------
        dataframe : pd.DataFrame, dtype float64
            Dataframe of lateral inputs at each reach (e.g. runoff + recharge).
            Column labels must contain each `reach_id` in the model.
            Index must be of type pd.DateTime with UTC timezone.
            
        start_time : pd.DateTime
            Starting time of the simulation. Defaults to self.datetime.

        end_time : pd.DateTime
            Ending time of the simulation. Defaults to the last timestamp in the provided dataframe.

        o_t_init : np.ndarray, dtype float64
            Initial outflow states at start of simulation. Defaults to self.o_t_next.

        Returns:
        --------
        self : Muskingum
            Yields model instance at each timestep for inspection.
        """
        return self.simulate_iter(dataframe, start_time=start_time, 
                                  end_time=end_time, o_t_init=o_t_init, only_one_time_step=only_one_time_step, with_troute=with_troute, **kwargs)

    def set_transmissive_boundary(self, index):
        self.alpha[index] = 1.
        self.beta[index] = 0.
        self.chi[index] = 0.
        self.gamma[index] = 0.

    def save_state(self):
        self.logger.info(f'Saving state for model {self.name} at time {self.datetime}...')
        self.saved_states["datetime"] = self.datetime
        # TODO: Don't need to store `i_t_next`
        self.saved_states["i_t_next"] = self.i_t_next.copy()
        self.saved_states["o_t_next"] = self.o_t_next.copy()
        for _, callback in self.callbacks.items():
            callback.__on_save_state__()

    def load_state(self):
        self.datetime = self.saved_states["datetime"]
        self.i_t_next = self.saved_states["i_t_next"]
        self.o_t_next = self.saved_states["o_t_next"]
        self.logger.info(f'Loading state for model {self.name} at time {self.datetime}...')
        for _, callback in self.callbacks.items():
            callback.__on_load_state__()

    def plot(self, ax, *args, **kwargs):
        paths = self.paths
        for path in paths:
            for subpath in path:
                plotting_path = np.asarray(subpath)
                ax.plot(plotting_path[:,0], plotting_path[:,1], *args, **kwargs)
    
    def bind_callback(self, callback, key='callback'):
        assert isinstance(callback, BaseCallback)
        self.callbacks[key] = callback

    def unbind_callback(self, key):
        return self.callbacks.pop(key)

    def copy(self):
        return copy.deepcopy(self)

    def split(self, indices, name=None, create_state_space=True):
        self = copy.deepcopy(self)
        startnode_indices = np.asarray([np.flatnonzero(self.startnodes == i).item()
                                        for i in indices])
        endnode_indices = self.endnodes[startnode_indices]
        # Cut watershed at indices
        self.endnodes[startnode_indices] = indices
        # Find connected components
        adj = lil_matrix((self.n, self.n), dtype=int)
        for i, j in zip(self.startnodes, self.endnodes):
            adj[j, i] = 1
        n_components, labels = csgraph.connected_components(adj)
        index_map = pd.Series(np.arange(self.n), index=self.startnodes)
        outer_startnodes = labels[startnode_indices].astype(int).tolist()
        outer_endnodes = labels[endnode_indices].astype(int).tolist()
        # Re-order
        new_outer_startnodes = []
        new_outer_endnodes = []
        new_startnode_indices = []
        new_endnode_indices = []
        for k in range(n_components):
            new_outer_startnodes.append(k)
            if k in outer_startnodes:
                pos = outer_startnodes.index(k)
                new_outer_endnodes.append(outer_endnodes[pos])
                new_startnode_indices.append(startnode_indices[pos])
                new_endnode_indices.append(endnode_indices[pos])
            else:
                new_outer_endnodes.append(k)
                new_startnode_indices.append(-1)
                new_endnode_indices.append(-1)
        outer_startnodes = np.asarray(new_outer_startnodes, dtype=int)
        outer_endnodes = np.asarray(new_outer_endnodes, dtype=int)
        startnode_indices = np.asarray(new_startnode_indices, dtype=int)
        endnode_indices = np.asarray(new_endnode_indices, dtype=int)
        # Create sub-watershed models
        components = {}
        for component in range(n_components):
            components[component] = {}
            selection = (labels == component)
            sub_startnodes = self.startnodes[selection]
            sub_endnodes = self.endnodes[selection]
            new_startnodes = np.arange(len(sub_startnodes))
            node_map = pd.Series(new_startnodes, index=sub_startnodes)
            new_endnodes = node_map.loc[sub_endnodes].values
            sub_index_map = index_map[sub_startnodes].values
            # Create sub-watershed model
            sub_model = copy.deepcopy(self)
            sub_model.name = str(component)
            sub_model.startnodes = new_startnodes
            sub_model.endnodes = new_endnodes
            sub_model.n = len(sub_model.startnodes)
            sub_model.indegree = self.indegree[sub_index_map]
            sub_model.reach_ids = np.asarray(self.reach_ids)[sub_index_map].tolist()
            sub_model.dx = self.dx[sub_index_map]
            sub_model.K = self.K[sub_index_map]
            sub_model.X = self.X[sub_index_map]
            sub_model.alpha = self.alpha[sub_index_map]
            sub_model.beta = self.beta[sub_index_map]
            sub_model.chi = self.chi[sub_index_map]
            sub_model.gamma = self.gamma[sub_index_map]
            sub_model.A = np.zeros((sub_model.n, sub_model.n))
            sub_model.B = np.zeros((sub_model.n, sub_model.n))
            sub_model.o_t_prev = self.o_t_prev[sub_index_map]
            sub_model.i_t_prev = self.i_t_prev[sub_index_map]
            sub_model.o_t_next = self.o_t_next[sub_index_map]
            sub_model.i_t_next = self.i_t_next[sub_index_map]
            sub_model.paths = [self.paths[i] for i in sub_index_map]
            if create_state_space:
                sub_model.create_state_space()
            sub_model.save_state()
            components[component]['model'] = sub_model
            components[component]['node_map'] = node_map
        # Create connections between sub-watersheds
        for component in range(n_components):
            startnode = outer_startnodes[component]
            endnode = outer_endnodes[component]
            upstream_node_map = components[startnode]['node_map']
            downstream_node_map = components[endnode]['node_map']
            startnode_index = startnode_indices[component]
            endnode_index = endnode_indices[component]
            components[startnode]['exit_node'] = startnode_index
            components[startnode]['entry_node'] = endnode_index
            if (startnode_index >= 0) and (endnode_index >= 0):
                upstream_model = components[startnode]['model']
                downstream_model = components[endnode]['model']
                upstream_index = upstream_node_map[startnode_index]
                downstream_index = downstream_node_map[endnode_index]
                downstream_model.indegree[downstream_index] -= 1
                downstream_model.i_t_next[downstream_index] -= upstream_model.o_t_next[upstream_index]
                # Fill in sink and source objects
                connection = Connection(upstream_model, downstream_model, 
                                        upstream_index, downstream_index)
                upstream_model.sinks.append(connection)
                downstream_model.sources.append(connection)
                #upstream_model.sinks.update({downstream_model.name : 
                #{
                #    'model' : downstream_model,
                #    'exit_node' : exit_index
                #}})
                #downstream_model.sources.update({upstream_model.name : 
                #{
                #    'model' : upstream_model,
                #    'entry_node' : entry_index
                #}})
        models = [components[component]['model'] for component in components]
        model_collection = ModelCollection(models, name=name)
        return model_collection

class Connection():
    def __init__(self, upstream_model, downstream_model, 
                 upstream_index, downstream_index, name=None):
        self.upstream_model = upstream_model
        self.downstream_model = downstream_model
        self.upstream_index = upstream_index
        self.downstream_index = downstream_index
        if name is None:
            self.name = str(uuid.uuid4())
        else:
            self.name = name

class ModelCollection():
    def __init__(self, models, name=None):
        self.models = {model.name : model for model in models}
        if name is None:
            self.name = str(uuid.uuid4())
        else:
            self.name = name

    @property
    def info(self):
        model_info = {}
        return model_info

    @property
    def datetime(self):
        # NOTE: Throws exception if models do not have same datetime
        collection_datetime = set([model.datetime for model
                                    in self.models.values()])
        if len(collection_datetime) == 1:
            collection_datetime = collection_datetime.pop()
            collection_datetime = pd.to_datetime(collection_datetime)
            return collection_datetime
        else:
            raise ValueError('Models must all have the same datetime')

    @property
    def timedelta(self):
        # NOTE: Throws exception if models do not have same datetime
        collection_timedelta = set([model.timedelta for model
                                    in self.models.values()])
        if len(collection_timedelta) == 1:
            collection_timedelta = collection_timedelta.pop()
            collection_timedelta = pd.to_timedelta(collection_timedelta)
            return collection_timedelta
        else:
            raise ValueError('Models must all have the same timedelta')

    def load_states(self):
        for key in self.models:
            self.models[key].load_state()

    def save_states(self):
        for key in self.models:
            self.models[key].save_state()

    def set_datetime(self, timestamp):
        for key in self.models:
            self.models[key].datetime = timestamp

    def init_states(self, streamflow):
        for key in self.models:
            model = self.models[key]
            reach_ids = model.reach_ids
            model_states = streamflow[reach_ids]
            model.init_states(o_t_next=model_states)

    def dump_model_collection(self, file_path, model_file_paths={}, dump_optional=True):
        connections = {}
        models = {}
        for model_name, model in self.models.items():
            # TODO: Path handling seems fragile
            #file_path_stem = os.path.splitext(file_path)[0]
            #default_file_path = f'{file_path_stem}.{model_name}.inp'
            #model_file_paths[model_name] = model_file_paths.setdefault(model_name,
            #                                                           default_file_path)
            for sink in model.sinks:
                name = sink.name
                if not name in connections:
                    connections.update({name : {
                        'upstream_model' : sink.upstream_model.name,
                        'downstream_model' : sink.downstream_model.name,
                        'upstream_index' : int(sink.upstream_index),
                        'downstream_index' : int(sink.downstream_index),
                    }})
            for source in model.sources:
                name = source.name
                if not name in connections:
                    connections.update({name : {
                        'upstream_model' : source.upstream_model.name,
                        'downstream_model' : source.downstream_model.name,
                        'upstream_index' : int(source.upstream_index),
                        'downstream_index' : int(source.downstream_index),
                    }})
        for model_name, model in self.models.items():
            #model_file_path = model_file_paths[model_name]
            #model.dump_model_file(model_file_path, dump_optional=dump_optional)
            #model_pointers.append({
            #    'model' : os.path.abspath(model_file_path),
            #    'sinks' : [sink.name for sink in model.sinks],
            #    'sources' : [source.name for source in model.sources]
            #                 })
            # TODO: Sinks and sources don't seem to be used
            # TODO: Doesn't allow ignoring optional fields
            models[model_name] = {'model' : model.info,
                                  'sinks' : [sink.name for sink in model.sinks],
                                  'sources' : [source.name for source in model.sources]}
        model_collection_info = {
            'models' : models,
            'connections' : connections 
        }
        with open(file_path, 'w') as f:
            #json.dump(model_collection_info, f)
            json.dump(model_collection_info, f, cls=ModelEncoder)

    @classmethod
    def from_file(cls, file_path, load_optional=True, **kwargs):
        models = load_model_collection(file_path, load_optional=load_optional)
        newinstance = cls(models, **kwargs)
        return newinstance


def load_model_file(file_path, load_optional=True):
    required_fields = {'name', 'datetime', 'timedelta', 'reach_ids',
     'startnodes', 'endnodes', 'K', 'X', 'o_t'}
    with open(file_path, 'r') as f:
        obj = json.load(f, cls=ModelDecoder)
    try:
        assert required_fields.issubset(set(obj.keys()))
    except:
        raise ValueError(f'Model field must contain fields {required_fields}')
    obj['startnodes'] = np.asarray(obj['startnodes'], dtype=np.int64)
    obj['endnodes'] = np.asarray(obj['endnodes'], dtype=np.int64)
    obj['K'] = np.asarray(obj['K'], dtype=np.float64)
    obj['X'] = np.asarray(obj['X'], dtype=np.float64)
    obj['o_t'] = np.asarray(obj['o_t'], dtype=np.float64)
    try:
        assert (obj['startnodes'].size == obj['endnodes'].size 
                == obj['K'].size == obj['X'].size == obj['o_t'].size)
        obj['n'] = obj['startnodes'].size
    except:
        raise ValueError('Arrays are not the same length')
    if load_optional:
        return obj
    else:
        obj = {k : v for k, v in obj.items()
               if not k in required_fields}
        return obj


def dump_model_file(obj, file_path, dump_optional=True):
    required_fields = {'name', 'datetime', 'timedelta', 'reach_ids',
     'startnodes', 'endnodes', 'K', 'X', 'o_t'}
    with open(file_path, 'w') as f:
        if dump_optional:
            to_dump = obj
        else:
            to_dump = {k : v for k, v in obj.items()
                        if k in required_fields}
        json.dump(to_dump, f, cls=ModelEncoder)


def load_nhd_geojson(file_path):
    with open(file_path, 'r') as f:
        d = json.load(f)
    node_ids = [i['attributes']['COMID'] for i in d['features']]
    link_ids = [i['attributes']['COMID'] for i in d['features']]
    paths = [np.asarray(i['geometry']['paths']) for i in d['features']]
    reach_ids = link_ids
    reach_ids = [str(x) for x in reach_ids]
    source_node_ids = [i['attributes']['COMID'] for i in d['features']]
    target_node_ids = [i['attributes']['toCOMID'] for i in d['features']]
    dx = np.asarray([i['attributes']['Shape_Length'] for i in d['features']]) #* 1000 # km to m
    # NOTE: node_ids and source_node_ids should always be the same for NHD
    # But not necessarily so for original json files
    node_ids = node_ids
    source_node_ids = source_node_ids
    target_node_ids = target_node_ids
    self_loops = []
    node_index_map = pd.Series(np.arange(len(node_ids)), index=node_ids)
    startnodes = node_index_map.reindex(source_node_ids, fill_value=-1).values
    endnodes = node_index_map.reindex(target_node_ids, fill_value=-1).values
    for i in range(len(startnodes)):
        if endnodes[i] == -1:
            self_loops.append(i)
            endnodes[i] = startnodes[i]
    n = startnodes.size
    obj = {
        'name' : str(uuid.uuid4()),
        'datetime' : DEFAULT_START_TIME,
        'timedelta' : DEFAULT_TIMEDELTA,
        'reach_ids' : reach_ids,
        'startnodes' : startnodes,
        'endnodes' : endnodes,
        'K' : 3600 * np.ones(n, dtype=np.float64),
        'X' : 0.29 * np.ones(n, dtype=np.float64),
        'o_t' : 1e-3 * np.ones(n, dtype=np.float64),
        'dx' : dx,
        'paths' : paths
    }
    return obj


def load_model_collection(file_path, load_optional=True):
    models = {}
    with open(file_path, 'r') as f:
        #model_collection_info = json.load(f)
        model_collection_info = json.load(f, cls=ModelDecoder)
    connections = model_collection_info['connections']
    #for model_info in model_collection_info['models']:
    for model_name, model_info in model_collection_info['models'].items():
        #model_file_path = model_info['model']
        #model = Muskingum.from_model_file(model_file_path, load_optional=load_optional)
        obj = model_info['model']
        model = Muskingum(obj, load_optional=load_optional)
        models[model.name] = model
    for connection_name, connection_dict in connections.items():
        upstream_model = models[connection_dict['upstream_model']]
        downstream_model = models[connection_dict['downstream_model']]
        upstream_index = connection_dict['upstream_index']
        downstream_index = connection_dict['downstream_index']
        connection = Connection(upstream_model, downstream_model, 
                                upstream_index, downstream_index, 
                                name=connection_name)
        upstream_model.sinks.append(connection)
        downstream_model.sources.append(connection)
    models = list(models.values())
    return models


def _solve_normal_depth(h, Q, B, z, mann_n, So):
    Q_computed = _Q(h, Q, B, z, mann_n, So)
    return Q_computed - Q


def _Q(h, Q, B, z, mann_n, So):
    A = h * (B + z * h)
    P = B + 2 * h * np.sqrt(1 + z**2)
    Q = (np.sqrt(So) / mann_n) * A ** (5 / 3) / P ** (2 / 3)
    return Q

def _dQ_dh(h, Q, B, z, mann_n, So):
    num_0 = 5 * np.sqrt(So) * (h * (B + h * z)) ** (2 / 3) * (B + 2 * h * z)
    den_0 = 3 * mann_n * (B + 2 * h * np.sqrt(z + 1)) ** (2 / 3)
    num_1 = 4 * np.sqrt(So) * np.sqrt(z + 1) * (h * (B + h * z)) ** (5 / 3)
    den_1 = 3 * mann_n * (B + 2 * h * np.sqrt(z + 1)) ** (5 / 3)
    t0 = num_0 / den_0
    t1 = num_1 / den_1
    return t0 - t1
