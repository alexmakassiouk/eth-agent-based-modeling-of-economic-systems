import numpy as np
from mesa.model import Model
from mesa.space import SingleGrid
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector
from agent import PolyaAgent, UtilityPolyaAgent


class PolyaProcess(Model):

    MODEL_TYPES = ("linear", "nonlinear")

    def __init__(self, model_type, max_steps, preference_match=None, benefits=None):
        """ Initialize models based on different model types.

        For models "linear" and "nonlinear", we use majority voting rule.
        For model "utility", we assume every agent has the same benefits and match of preferences. 
        The corresponding utility functions are defined in slide 19 and slide 27.

        Args:
            model_type: "linear", "nonlinear", or "utility"
            num_agents: Number of agents in the model
            preference_match: a 2x2 array-like structure, denoted as u^c(a, b) in slides where a, b in {0, 1}.
            benefits: a 2d vector representing benefits for agents, denoted as A^0 or A^1 in the slides
        """
        assert model_type in PolyaProcess.MODEL_TYPES,\
            "model_types should be either \"linear\", \"nonlinear\""

        super().__init__()
        self.schedule = BaseScheduler(self)

        self.max_steps = max_steps
        self.model_type = model_type
        self.frequencies = []
        if model_type == "linear" or model_type =="nonlinear":
            # For "linear" or "nonlinear" models,
            # we initialize an agent with state 1 ..
            self.num_ones = 1
            self.num_zeros = 0
            agent_one = PolyaAgent(1, self)
            self.schedule.add(agent_one)
            # and an agent with state 0 at t=0
            self.num_ones = 0
            self.num_zeros = 1
            agent_zero = PolyaAgent(0, self)
            self.schedule.add(agent_zero)
            #
            self.num_ones = 1
            self.num_zeros = 1
            self._next_agent_id = 2

            self.datacollector = DataCollector(
            model_reporters={"f_1": self.collect_frequencies}
        )

            



    @property
    def global_frequency(self):
        # global_frequency represents N^1 / N
        if (self.num_ones + self.num_zeros == 0):
            return 0
        else:
            return self.num_ones/(self.num_ones + self.num_zeros)


    def step(self):
        '''Advance the model by one step.'''
        new_agent = PolyaAgent(self._next_agent_id, self)
        self.num_ones += new_agent.state
        self.num_zeros += (1-new_agent.state)
        self._next_agent_id +=1
        self.frequencies.append(self.global_frequency)
        self.datacollector.collect(self)

    def collect_frequencies(self):
         return self.frequencies
        

class UtilityPolyaProcess(PolyaProcess):
    UTILITY_MODELS = ("preference", "technology")
    MODEL_TYPES = ("linear", "nonlinear", "preference-utility", "technology-utility")
    def __init__(self, model_type, max_steps, utility_model_type, A, uc, seed, preference_match=None, benefits=None):
        assert utility_model_type in UtilityPolyaProcess.UTILITY_MODELS
        assert model_type in UtilityPolyaProcess.MODEL_TYPES

        super().__init__(model_type, max_steps, preference_match, benefits)
        self.utility_type = utility_model_type
        self.num_zeros = 0
        self.num_ones = 0
        self.seed = seed
        self._next_agent_id = seed
        self.A = A
        self.uc = uc
        self.datacollector = DataCollector(
            model_reporters={"f_1": self.collect_frequency, "delta_n": self.collect_delta, "n0": self.collect_n_0, "n1": self.collect_n_1}
        )

    # Override
    def step(self):
        new_agent = UtilityPolyaAgent(self._next_agent_id, self, self.utility_type, self.A, self.uc)
        self.num_ones += new_agent.state
        self.num_zeros += (1-new_agent.state)
        self._next_agent_id +=1
        self.datacollector.collect(self)


    @property
    def delta_n(self):
        return self.num_zeros-self.num_ones
    
    def collect_delta(self):
         return self.delta_n
    def collect_frequency(self):
         return self.global_frequency
    def collect_n_0(self):
        return self.num_zeros
    def collect_n_1(self):
        return self.num_ones