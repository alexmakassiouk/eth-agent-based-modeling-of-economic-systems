import numpy as np
from mesa.model import Model
from mesa.space import SingleGrid
from mesa.time import BaseScheduler
from agent import PolyaAgent

np.random.seed(42)


class PolyaProcess(Model):

    MODEL_TYPES = ("linear", "nonlinear", "preference", "technology")

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
        assert model_type in PolyaProcess.MODEL_TYPES, \
            "model_types should be either \"linear\", \"nonlinear\""

        super().__init__()
        self.schedule = BaseScheduler(self)

        self.max_steps = max_steps
        self.model_type = model_type
        self.frequencies = []
        self.deltas = []
        self.num_zero_list = []
        self.benefits = benefits
        self.preference_match = preference_match
        if model_type == "linear" or model_type == "nonlinear":
            # For "linear" or "nonlinear" models,
            # we initialize an agent with state 1 ..
            self.num_ones = 1
            self.num_zeros = 0
            agent_one = PolyaAgent(unique_id=0, model=self, preference=1)
            self.schedule.add(agent_one)
            # and an agent with state 0 at t=0
            self.num_ones = 1
            self.num_zeros = 1
            agent_zero = PolyaAgent(unique_id=1, model=self, preference=0)
            self.schedule.add(agent_zero)
            #
            self.num_ones = 1
            self.num_zeros = 1
            self._next_agent_id = 2
        if model_type == "preference" or model_type == "technology":
            self._next_agent_id = 0
            self.num_ones = 0
            self.num_zeros = 0

    @property
    def global_frequency(self):
        # global_frequency represents N^1 / N
        if self.N == 0:
            return 0
        else:
            return self.num_ones/self.N

    @property
    def N(self):
        return self.num_ones+self.num_zeros

    @property
    def delta_N(self):
        return self.num_zeros-self.num_ones

    def step(self):
        '''Advance the model by one step.'''
        new_agent = PolyaAgent(self._next_agent_id, self,
                               np.random.choice([0, 1]))
        self.schedule.add(new_agent)
        self._next_agent_id += 1
        if new_agent.state == 0:
            self.num_zeros += 1
        elif new_agent.state == 1:
            self.num_ones += 1
            self.num_zero_list.append(self.num_zeros)
        self.frequencies.append(self.global_frequency)
        self.deltas.append(self.delta_N)
