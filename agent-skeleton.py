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

        self.preference = ...
        self.state = ...

    def choose_state(self, model):
        # The agent choses its state depending on the model type
        model_type = model.model_type
        if model_type == "linear":
                return self.linear(model.global_frequency)


    def linear(self, f_1):
        # Use slide 5
        ...


    def nonlinear(self, f_1):
        # Use equation and parameters on slide 11
        ...
