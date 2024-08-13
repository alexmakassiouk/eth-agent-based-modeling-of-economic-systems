import numpy as np
from mesa.agent import Agent


class PolyaAgent(Agent):
    def __init__(self, unique_id, model):
        """ Initialize an agent for PolyaProcess model

        Args:
            STATE: 1 or 0, fixed once initialized
            PREFERENCES: 1 or 0, fixed once initialized
        """

        super().__init__(unique_id=unique_id, model=model)

        self.preference = None # initialized in choose_state below
        self.state = self.choose_state(model=model)

    def choose_state(self, model):
        # The agent choses its state depending on the model type
        model_type = model.model_type
        if model_type == "linear":
            return self.linear(model.global_frequency)
        elif model_type == "nonlinear":
            return self.nonlinear(model.global_frequency)
        else:                                                                          # task 5
            # Get new agent's preference                                               # task 5
            self.preference = np.random.choice([1, 0])                                 # task 5
                                                                                       # task 5
            # Get new agent's choice                                                   # task 5
            if model_type == "benefits_to_pref":                                       # task 5
                choice = self.benefits_to_pref(model.num_zeros, model.num_ones,        # task 5
                                               model.benefits, model.preference_match) # task 5
            elif model_type == "benefits_to_tech":                                     # task 5
                choice = self.benefits_to_tech(model.num_zeros, model.num_ones,        # task 5
                                               model.benefits, model.preference_match) # task 5
            return choice                                                              # task 5

    def linear(self, f_1):
        # Use slide 5
        p_1 = f_1
        return int(np.random.rand() < p_1) # 1 with probability p_1, and 0 otherwise

    def nonlinear(self, f_1):
        # Use equation and parameters on slide 13
        # parameters: theta = 0.5 and beta = 10
        z = f_1 - 0.5
        p_1 = 1 / (1 + np.exp(-10*z))
        return int(np.random.rand() < p_1) # 1 with probability p_1, and 0 otherwise

    def benefits_to_pref(self, n0, n1, a, uc):                                         # task 5
        # Use the equation form slide 19 with A assigned from preference               # task 5
        util = [None, None]                                                            # task 5
        util[0] = uc[0][self.preference] + a[self.preference]*n0                       # task 5
        util[1] = uc[1][self.preference] + a[self.preference]*n1                       # task 5
        if util[0] > util[1]:                                                          # task 5
            return 0                                                                   # task 5
        else:                                                                          # task 5
            return 1                                                                   # task 5

    def benefits_to_tech(self, n0, n1, a, uc):                                         # task 5
        # Use the equation form slide 19 with A assigned from technology               # task 5
        util = [None, None]                                                            # task 5
        util[0] = uc[0][self.preference] + a[0]*n0                                     # task 5
        util[1] = uc[1][self.preference] + a[1]*n1                                     # task 5
        if util[0] > util[1]:                                                          # task 5
            return 0                                                                   # task 5
        else:                                                                          # task 5
            return 1                                                                   # task 5
