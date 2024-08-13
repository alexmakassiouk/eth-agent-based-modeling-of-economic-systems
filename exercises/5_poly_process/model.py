import numpy as np
from mesa.model import Model
from mesa.time import BaseScheduler
from agent import PolyaAgent

np.random.seed(42)


class PolyaProcess(Model):

    MODEL_TYPES = ("linear", "nonlinear", "benefits_to_pref", "benefits_to_tech")

    def __init__(self, model_type, max_steps, preference_match=None, benefits=None):
        """ Initialize models based on different model types.

        For models "linear" and "nonlinear", we use majority voting rule.
        For model "benefits_to_pref" or "benefits_to_tech", we assume every agent has the same benefits and match of preferences.
        The corresponding utility functions are defined in slide 19 and slide 27.

        Args:
            model_type: "linear", "nonlinear", "benefits_to_pref", or "benefits_to_tech"
            num_agents: Number of agents in the model
            preference_match: a 2x2 array-like structure, denoted as u^c(a, b) in slides where a, b \in {0, 1}.
            benefits: a 2d vector representing benefits for agents, denoted as A^0 or A^1 in the slides
        """
        assert model_type in PolyaProcess.MODEL_TYPES,\
            "model_types should be either \"linear\", \"nonlinear\", \"benefits_to_pref\", or \"benefits_to_tech\""

        super().__init__()
        self.schedule = BaseScheduler(self)

        self.max_steps = max_steps
        self.model_type = model_type
        if model_type == "linear" or model_type =="nonlinear":
            # For "linear" or "nonlinear" models,
            # we initialize an agent with state 1 ..
            self.num_ones = 1
            self.num_zeros = 0
            agent_one = PolyaAgent(unique_id=0, model=self)
            self.schedule.add(agent_one)
            # and an agent with state 0 at t=0
            self.num_ones = 0
            self.num_zeros = 1
            agent_zero = PolyaAgent(unique_id=1, model=self)
            self.schedule.add(agent_zero)
            #
            self.num_ones = 1
            self.num_zeros = 1
            self._next_agent_id = 2
        elif model_type == "benefits_to_pref" or model_type == "benefits_to_tech": # task 5
            assert benefits is not None and preference_match is not None,\
                "Please provide 2D array for the benefits and " +\
                "2x2 matrix for the preference_match"                              # task 5
                                                                                   # task 5
            self.benefits = benefits                                               # task 5
            self.preference_match = preference_match                               # task 5
            self._next_agent_id = 0                                                # task 5
            self.num_ones = 0                                                      # task 5
            self.num_zeros = 0                                                     # task 5

    @property
    def global_frequency(self):
        # global_frequency represents N^1 / N
        return self.num_ones / (self.num_ones + self.num_zeros)


    def step(self):
        '''Advance the model by one step.'''
        # Add new agent
        new_agent = PolyaAgent(self._next_agent_id, self)
        if new_agent.state == 0:
            self.num_zeros += 1
        else:
            self.num_ones += 1
        self.schedule.add(new_agent)

        # Advance time
        self.schedule.steps += 1
        self._next_agent_id += 1
        if self.schedule.steps >= self.max_steps:
            self.running = False
