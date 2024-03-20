from typing import Literal
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
        self.state: Literal[1] | Literal[0] = self.choose_state(model)

    def choose_state(self, model) -> Literal[1] | Literal[0]:
        # The agent choses its state depending on the model type
        model_type = model.model_type
        if model_type == "linear":
            return self.linear(model.global_frequency)
        elif model_type == "nonlinear":
            return self.nonlinear(model.global_frequency)
        else:
            return 0


    def linear(self, f_1) -> Literal[1] | Literal[0]:
        # Use slide 5
        rnd = self.random.random()
        return 1 if rnd < f_1 else 0


    def nonlinear(self, f_1) -> Literal[1] | Literal[0]:
        # Use equation and parameters on slide 11
        beta = 10
        rnd = self.random.random()
        z = f_1-0.5
        decision = 1/(1+np.exp(-beta * z))
        # decision = expit(-beta*z)
        return 1 if rnd < decision else 0
    
    def step(self):
        pass
        
class UtilityPolyaAgent(PolyaAgent):
    def __init__(self, unique_id, model, utility_type, A, uc, preference=None):
        assert utility_type in ("preference", "technology")
        
        super().__init__(unique_id, model, preference)
        self.mu = self.random.choice([0,1])
        self.A: list[float] = A
        self.uc: list[list[float]] = uc
        self.utility_type = utility_type

        # self.random.seed(self.unique_id)
        self.state: int = self.choose_utility_state(model, self.utility_type)

    def calc_preference_utility(self, n):
        u0 = self.uc[0][self.mu] + self.A[self.mu]*n[0]
        u1 = self.uc[1][self.mu] + self.A[self.mu]*n[1]
        self.u = [u0, u1]

    def calc_technology_utility(self, n):
        u0 = self.uc[0][self.mu] + self.A[0]*n[0]
        u1 = self.uc[1][self.mu] + self.A[1]*n[1]
        self.u = [u0, u1]

    # Override
    def choose_utility_state(self, model, utility_type):
        n1 = model.num_ones
        n0 = model.num_zeros
        if utility_type == "preference":
            self.calc_preference_utility([n0, n1])
        elif utility_type == "technology":
            self.calc_technology_utility([n0, n1])
        return int(np.argmax(self.u))
    
