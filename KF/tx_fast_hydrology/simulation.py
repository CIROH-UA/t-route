import os
import datetime
import asyncio
import logging
import uuid
import json
import numpy as np
import pandas as pd
from tx_fast_hydrology.callbacks import BaseCallback

logger = logging.getLogger(__name__)

class Simulation():
    def __init__(self, model_collection, inputs, with_troute=False):
        self.model_collection = model_collection
        self.models = model_collection.models
        self.inputs = self.load_inputs(inputs)
        self.with_troute = with_troute
        self.outputs = {}

    def load_inputs(self, inputs):
        input_collection = {}
        for index, model in self.models.items():
            input_collection[index] = inputs[model.reach_ids].copy()
        return input_collection

    def simulate(self):
        raise NotImplementedError
        outer_startnodes = self.outer_startnodes
        outer_endnodes = self.outer_endnodes
        outer_indegree = self.outer_indegree
        multi_outputs = {}
        m = outer_startnodes.size
        outer_indegree_t = outer_indegree.copy()
        outer_indegree_t[outer_startnodes == outer_endnodes] -= 1

        for k in range(m):
            startnode = outer_startnodes[k]
            endnode = outer_endnodes[k]
            while (outer_indegree_t[startnode] == 0):
                model_start = self.models[startnode]
                p_t_start = self.inputs[startnode]
                outputs = {}
                outputs[model_start.datetime] = model_start.o_t_next.copy()
                for state in model_start.simulate_iter(p_t_start, inc_t=True):
                    o_t_next = state.o_t_next
                    outputs[state.datetime] = o_t_next.copy()
                outputs = pd.DataFrame.from_dict(outputs, orient='index')
                outputs.index = pd.to_datetime(outputs.index, utc=True)
                outputs.columns = p_t_start.columns
                multi_outputs[startnode] = outputs
                print(startnode)
                if startnode != endnode:
                    model_end = self.models[endnode]
                    terminal_node = self.terminal_nodes[startnode]
                    entry_node = self.entry_nodes[startnode]
                    index_out = self.node_map[startnode][terminal_node]
                    index_in = self.node_map[endnode][entry_node]
                    reach_id_out = model_start.reach_ids[index_out]
                    reach_id_in = model_end.reach_ids[index_in]
                    i_t_prev = outputs[reach_id_out].shift(1).iloc[1:].fillna(0.)
                    i_t_next = outputs[reach_id_out].iloc[1:]
                    gamma_in = model_end.gamma[index_in]
                    alpha_in = model_end.alpha[index_in]
                    beta_in = model_end.beta[index_in]
                    self.inputs[endnode].loc[:, reach_id_in] += (alpha_in
                                                                 * i_t_next.values
                                                                 / gamma_in
                                                               + beta_in
                                                                 * i_t_prev.values
                                                                 / gamma_in)
                    outer_indegree_t[endnode] -= 1
                    startnode = endnode
                    endnode = outer_endnodes[startnode]
                else:
                    break
        return multi_outputs
    
    @property
    def datetime(self):
        return self.model_collection.datetime

    def load_states(self):
        self.model_collection.load_states()

    def save_states(self):
        self.model_collection.save_states()

    def init_states(self, streamflow):
        self.model_collection.init_states(streamflow)

    def set_datetime(self, timestamp):
        self.model_collection.set_datetime(timestamp)

class AsyncSimulation(Simulation):
    def __init__(self, model_collection, inputs, with_troute=False):
        return super().__init__(model_collection, inputs, with_troute=with_troute)
    
    async def simulate(self, only_one_time_step=False):
        try:
            asyncio.get_running_loop()
            loop_running = True
        except RuntimeError:
            loop_running = False
        if loop_running:
            await self._main(only_one_time_step=only_one_time_step)
        else:
            asyncio.run(self._main(only_one_time_step=only_one_time_step))
        return self.outputs

    async def _main(self, only_one_time_step=False):
        indegree = {model.name : len(model.sources) for model 
                    in self.model_collection.models.values()}
        self._indegree = indegree
        async with asyncio.TaskGroup() as taskgroup:
            for name, predecessors in indegree.items():
                if predecessors == 0:
                    model = self.models[name]
                    inputs = self.inputs[name]
                    taskgroup.create_task(self._simulate(taskgroup, model, inputs,
                                                        name, only_one_time_step))

    async def _simulate(self, taskgroup, model, inputs, name, only_one_time_step=False):
        logger.debug(f'Started job for sub-watershed {name}')
        start_time = model.datetime
        
        # Grab the initial state directly from the model's current state
        o_t_init = model.o_t_next.copy()

        outputs = {}
        outputs[start_time] = o_t_init

        if only_one_time_step:
            #routing is already done, so we can update model's datetime to the next time step for kalman filter
            if self.with_troute:
                model.datetime += model.timedelta

            state = model.simulate_iter(
                inputs, 
                o_t_init=o_t_init, 
                only_one_time_step=only_one_time_step, 
                with_troute=self.with_troute
            )
            
            current_time = state.datetime
            o_t_next = state.o_t_next
            outputs[current_time] = o_t_next

        else:
            for state in model.simulate_iter(inputs, o_t_init=o_t_init, only_one_time_step=only_one_time_step):
                current_time = state.datetime
                o_t_next = state.o_t_next
                outputs[current_time] = o_t_next
                
        outputs = pd.DataFrame.from_dict(outputs, orient='index')
        outputs.index = pd.to_datetime(outputs.index, utc=True)
        outputs.columns = inputs.columns
        self.outputs[name] = outputs
        
        taskgroup.create_task(self._accumulate(taskgroup, outputs, name, only_one_time_step))

    async def _accumulate(self, taskgroup, outputs, name, only_one_time_step=False):
        indegree = self._indegree
        startnode = name
        upstream_model = self.models[startnode]
        connections = upstream_model.sinks
        #endnodes = model_start.sinks.keys()
        #for endnode in endnodes:
        for connection in connections:
            downstream_model = connection.downstream_model
            endnode = downstream_model.name
            if startnode != endnode:
                #model_end = self.models[endnode]
                inputs = self.inputs[endnode]
                upstream_index = connection.upstream_index
                downstream_index = connection.downstream_index
                reach_id_out = upstream_model.reach_ids[upstream_index]
                reach_id_in = downstream_model.reach_ids[downstream_index]
                # TODO: This seems fragile
                i_t_prev = outputs[reach_id_out].shift(1).iloc[1:].fillna(0.)
                i_t_next = outputs[reach_id_out].iloc[1:]
                gamma_in = downstream_model.gamma[downstream_index]
                alpha_in = downstream_model.alpha[downstream_index]
                beta_in = downstream_model.beta[downstream_index]
                inputs.loc[:, reach_id_in] += (alpha_in * i_t_next.values / gamma_in
                                            + beta_in * i_t_prev.values / gamma_in)
                indegree[endnode] -= 1
                if (indegree[endnode] == 0):
                    taskgroup.create_task(self._simulate(taskgroup, downstream_model,
                                                        inputs, endnode,
                                                        only_one_time_step))
        logger.debug(f'Finished job for sub-watershed {name}')


class CheckPoint(BaseCallback):
    def __init__(self, model, checkpoint_time=None, timedelta=None):
        self.model = model
        if checkpoint_time is None:
            if timedelta is None:
                raise ValueError('Either `checkpoint_time` or `timedelta` must not be `None`.')
            else:
                checkpoint_time = model.datetime + datetime.timedelta(seconds=timedelta)
        self.checkpoint_time = checkpoint_time
        self.timedelta = timedelta
        self.model_saved = False

    def __on_simulation_start__(self):
        timedelta = self.timedelta
        if timedelta is None:
            return None
        else:
            checkpoint_time = self.model.datetime + datetime.timedelta(seconds=timedelta)
            self.set_checkpoint(checkpoint_time)
    
    def __on_step_end__(self):
        checkpoint_time = self.checkpoint_time
        current_time = self.model.datetime
        if (current_time >= checkpoint_time) and (not self.model_saved):
            self.model.save_state()
            self.model_saved = True

    def __on_save_state__(self):
        model = self.model
        for _, callback in model.callbacks.items():
            if hasattr(callback, 'save_state'):
                callback.save_state()

    def __on_load_state__(self):
        model = self.model
        for _, callback in model.callbacks.items():
            if hasattr(callback, 'load_state'):
                callback.load_state()
    
    def set_checkpoint(self, checkpoint_time):
        logger.info(f'Setting checkpoint time to {checkpoint_time}')
        self.checkpoint_time = checkpoint_time
        self.model_saved = False
