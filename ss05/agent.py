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
        
