import numpy as np
from mesa.agent import Agent
import networkx as nx

class Node(Agent):
    def __init__(self, unique_id, model: "CascadeNetwork", load, capacity):
        """Initialize an agent for cascade network model

        "load" == "fragility" and "capacity" == "threshold" and their interpretations
        follow the ones in the paper.

        Args:
            state: failure state. True if fail and False if survive
            _fail_in_nodes: Set of in-neighbors of the node which is failing.
                            Defined as Equation 14 in the paper.
            _healthy_out_nodes: Set of out-neighbors of the node which remains healthy in the next time step.
                                Defined as Equation 14 in the paper.
        """
        super().__init__(unique_id=unique_id, model=model)
        self.failed = False
        self.load = load
        self.capacity = capacity
        self._next_load = load
        self._initial_load = load

        # this attribute is required by mesa.BatchRunner
        # refer to the official introductory tutorial of mesa for more
        self.running = True

    @property
    def failing(self):
        return not self.failed and self.load >= self.capacity

    @property
    def indegree(self):
        return self.model.network.in_degree(self.unique_id)

    @property
    def outdegree(self):
        return self.model.network.out_degree(self.unique_id)


    def _nodes_to_ids(self, nodes):
        ids = set()
        for node in nodes:
            ids.add(node.unique_id)

        return ids

    def _ids_to_nodes(self, ids):
        nodes = set()
        for id in ids:
            node = self.model.schedule.agents[id]
            nodes.add(node)

        return nodes

    def inneighbors(self, get_id=False):
        ids = list(self.model.network.predecessors(self.unique_id))
        return ids if get_id else self._ids_to_nodes(ids)


    def outneighbors(self, get_id=False):
        ids = list(self.model.network.successors(self.unique_id))
        return ids if get_id else self._ids_to_nodes(ids)


    # === YOUR CODE HERE ===
    def inward(self):
        failed_neighbors = 0
        # Iterate through incoming neighbors
        for neighbor in self.inneighbors():
            # Check if neighbor has failed or is failing
            if neighbor.failed or neighbor.failing:
                failed_neighbors += 1

        # Calculate the next load based on the number of failed neighbors and the node's indegree
        self._next_load = failed_neighbors / self.indegree

    def outward(self):
        self._next_load = 0
        # Iterate through incoming neighbors
        for neighbor in self.inneighbors():
            # Increase load based on the state of each neighbor and their outdegree
            if neighbor.failed or neighbor.failing:
                self._next_load += 1 / neighbor.outdegree

    # === END OF YOUR CODE ===


    def reach_in(self, get_id=False):
        reach_in = set()
        G = self.model.network
        # one nodes are failed nodes at t + 1
        one_nodes = self.model.failed_ids.union(self.model.failing_ids)
        in_neighbors = self.inneighbors(get_id=True)

        # Calculate reach_in_nodes
        for s in one_nodes:
            if s in reach_in:
                continue
            if s in in_neighbors:
                reach_in.add(s)
            else:
                for s, *path, t in nx.all_simple_paths(G, s, self.unique_id):
                    if set(path).issubset(one_nodes):
                        reach_in.add(s)
                        reach_in.union(path)
                        break
        # Return node ids or node objects
        return reach_in if get_id else self._ids_to_nodes(reach_in)

    def reach_out(self, get_id=False):
        reach_out = set()
        G = self.model.network
        # one nodes are failed nodes at t + 1
        one_nodes = self.model.failed_ids.union(self.model.failing_ids)
        # zero nodes are healthy nodes at t + 1
        zero_nodes = self.model.healthy_ids
        out_neighbors = self.outneighbors(get_id=True)

        # Calculate reach_out_nodes
        for t in zero_nodes:
            if t in reach_out:
                continue
            if t in out_neighbors:
                reach_out.add(t)
            else:
                for _, *path, _ in nx.all_simple_paths(G, self.unique_id, t):
                    if set(path).issubset(one_nodes):
                        reach_out.add(t)
                        break

         # Return node ids or node objects
        return reach_out if get_id else self._ids_to_nodes(reach_out)

    def llsc(self):
        if self.failed or self.failing:
            self._next_load = self.capacity if self.model.model_type == "overload" else 0
        else:
            self._next_load = self._initial_load
            reach_in = self.reach_in()

            # Calculate self._next_load
            for node in reach_in:
                num_reach_out = len(node.reach_out(get_id=True))
                if self.model.model_type == "load":
                    self._next_load += node._initial_load / num_reach_out
                else:
                    self._next_load += (node._initial_load -
                                        node.capacity) / num_reach_out

    def failed_in(self, get_id=False):
        failed_in = set()
        for in_node in self.inneighbors():
            if in_node.failing:
                failed_in.add(in_node)

        # Return node objects or node ids
        return self._nodes_to_ids(failed_in) if get_id else failed_in

    def healthy_out(self, get_id=False):
        healthy_out = set()
        for out_node in self.outneighbors():
            if not out_node.failed and not out_node.failing:
                healthy_out.add(out_node)

        # Return node objects or node ids
        return self._nodes_to_ids(healthy_out) if get_id else healthy_out

    def llss(self):
        if self.failed or self.failing:
            self._next_load = self.capacity if self.model.model_type == "overload" else 0
        else:
            failed_in = self.failed_in()

            # Calculate self._next_load
            self._next_load = self.load
            for node in failed_in:
                if len(node.healthy_out()) == 0:
                    continue
                else:
                    if self.model.model_type == "load":
                        self._next_load += node.load / \
                            len(node.healthy_out())
                    else:
                        self._next_load += (node.load - node.capacity) / \
                            len(node.healthy_out())

    def step(self):
        model_type = self.model.model_type
        load_type = self.model.load_type

        if model_type == "constant":
            if load_type == "in":
                self.inward()
            if load_type == "out":
                self.outward()

        if model_type == "load" or model_type == "overload":
            if load_type == "llsc":
                self.llsc()
            if load_type == "llss":
                self.llss()

    def advance(self):
        if self.model.test:
            self.test()

        # postprocess nodes that are failing at t
        if self.failing:
            self.model.failing_ids.remove(self.unique_id)
            self.failed = True
            self.model.failed_ids.add(self.unique_id)

        self.load = self._next_load

        # preprocess nodes that are failing at t + 1
        if self.failing:
            self.model.failing_ids.add(self.unique_id)
            self.model.healthy_ids.remove(self.unique_id)

    def test(self):
        if self.failing:
            state = "failing"
        elif self.failed:
            state = "failed"
        else:
            state = "non-failed"

        print("Node {} (Capacity = {}): Load {:.2f}, {}".format(self.unique_id, self.capacity,
                                                                self.load, state))
