import numpy as np
from mesa.agent import Agent


class PolyaAgent(Agent):
    def __init__(self, unique_id, model, preference=None):
        """ Initialize an agent for PolyaProcess model

        Args:
            STATE: 1 or 0, fixed once initialized
            PREFERENCES: 1 or 0, fixed once initialized
        """

        super().__init__(unique_id=unique_id, model=model)

        self.preference = preference
        self.state = self.choose_state(model=model)
        self.model = model

    def choose_state(self, model):
        # The agent choses its state depending on the model type
        model_type = model.model_type
        if model_type == "linear":
            return self.linear(model.global_frequency)
        elif model_type == "nonlinear":
            return self.nonlinear(model.global_frequency)
        elif model_type == "preference":
            return 1 if self.utility_preference(model.global_frequency) else 0
        elif model_type == "technology":
            return 1 if self.utility_technology(model.global_frequency) else 0

    def utility_preference(self, f_1):
        utility_0 = self.calculate_preference_utility(
            uc=self.model.preference_match[self.preference][0], s=0, f_1=f_1)
        utility_1 = self.calculate_preference_utility(
            uc=self.model.preference_match[self.preference][1], s=1, f_1=f_1)
        return utility_1 > utility_0

    def utility_technology(self, f_1):
        utility_0 = self.calculate_technology_utility(
            uc=self.model.preference_match[self.preference][0], s=0, f_1=f_1)
        utility_1 = self.calculate_technology_utility(
            uc=self.model.preference_match[self.preference][1], s=1, f_1=f_1)
        return utility_1 > utility_0

    def linear(self, f_1):
        # Use slide 5
        rnd = self.random.random()
        if rnd <= f_1:
            return 1
        else:
            return 0

    def nonlinear(self, f_1):
        # Use equation and parameters on slide 11
        beta = 10  # From slides
        theta = 0.5
        rnd = self.random.random()
        decision = self.logit(beta=beta, z=f_1-theta)
        if rnd <= decision:
            return 1
        else:
            return 0

    def logit(self, beta, z):
        return 1/(1+np.exp(-beta*z))

    def calculate_technology_utility(self, uc, s, f_1):
        benefit = self.model.benefits[s]
        if self.unique_id == 0:
            global_freq = 0
            n = 0
        else:
            global_freq = f_1 if s == 1 else 1-f_1
            n = self.model.num_ones if s == 1 else self.model.num_zeros
        return uc + benefit*n

    def calculate_preference_utility(self, uc, s, f_1):
        # individual_utility = 1 if s == self.preference else 0
        benefit = self.model.benefits[self.preference]
        if self.unique_id == 0:
            global_freq = 0
            n = 0
        else:
            global_freq = f_1 if s == 1 else 1-f_1
            n = self.model.num_ones if s == 1 else self.model.num_zeros
        return uc + benefit*n
