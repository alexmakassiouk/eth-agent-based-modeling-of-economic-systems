import numpy as np
from mesa.model import Model
from mesa.time import SimultaneousActivation
from agent import Node
import networkx as nx

def form_edges_to_matrix(nnodes, edges):
    # we create an empty adjaceny matrix
    adj = np.zeros((nnodes, nnodes))

    # we fill in with 1 the adjacency matrix when we have an edge
    for i,j in edges:
        adj[i,j] = 1
        adj[j,i] = 1
    return adj


class CascadeNetwork(Model):

    MODEL_LOAD = (("constant", "in"), ("constant", "out"), (
        "load", "llss"), ("load", "llsc"), ("overload", "llss"), ("overload", "llsc"))

    def __init__(self, adj_matrix, load_vec, capacity_vec, model_type, load_type, edge_list =None, max_steps=5, test=False):
        """Implemented based on paper
        "Systemic Risk in a Unifying Framework for Cascading Processes on Network".

        Args:
            adj_matrix: A numpy matrix where each entry should be either 0 or 1 that denotes the existence
                of an edge between two nodes.
            edge_list: A list (or tuples) of edgee, e.g., [(1,0), (2,0), (1,2)] that creates a triangle.
                Note that this argument is ignored adj_matrix is not None.
            capacity_vec: A list of capacity values for all agents.
            model_type: "constant", "load", or "overload".
            load_type:
                When model_type == "constant", this value is either "in"
                for inward variant or "out" for outward variant.
                When model_type == "load" or "overload", this value is
                either "llsc" for local load sharing and load conservation or "llss" for local shedding.
        """
        if edge_list is not None and adj_matrix is None:
            nnodes = len(load_vec)
            adj_matrix = form_edges_to_matrix(nnodes, edge_list)

        assert adj_matrix.shape[0] == adj_matrix.shape[1],\
            "Adjacency matrix should be a square matrix"
        assert adj_matrix.shape[0] == len(load_vec) and len(load_vec) == len(capacity_vec), \
            "Shape of adjacency matrix and length of capacity/load vectors do not match"

        self.model_type = model_type.lower()
        self.load_type = load_type.lower()
        model_load = (self.model_type, self.load_type)
        assert model_load in CascadeNetwork.MODEL_LOAD, \
            "Supported values for model_type and load_type are one of the following: {}".format(
                CascadeNetwork.MODEL_LOAD)

        super().__init__()
        self.max_steps = max_steps
        self.test = test # used in testing, see the Node.advance and test.py
        self.schedule = SimultaneousActivation(self)
        self.num_nodes = adj_matrix.shape[0]

        # A failing node is healthy at t but failed at t+1
        # A healthy node will remain healthy at t+1
        self.healthy_ids = set()
        self.failing_ids = set()
        self.failed_ids = set()

        #initializing the network
        # self.network = pp.Network(directed=True)
        self.network = nx.DiGraph()

        # Initialize load and capacity for each agent/nodes
        for i in range(self.num_nodes):
            capacity = capacity_vec[i]
            load = load_vec[i]
            node = Node(i, self, load, capacity)
            self.schedule.add(node)
            self.network.add_node(i)
            if load >= capacity:
                self.failing_ids.add(node.unique_id)
            else:
                self.healthy_ids.add(node.unique_id)

        # Adding links in the network
        for source in range(self.num_nodes):
            for target in range(self.num_nodes):
                if adj_matrix[source,target] == 0:
                    continue
                self.network.add_edge(source, target)

    def step(self):
        '''Advance the model by one step.'''

        if self.test:
            print("Step {}:".format(self.schedule.steps))
        self.schedule.step()
        if self.test:
            print("-" * 60)

        if self.schedule.steps > self.max_steps:
            self.running = False
