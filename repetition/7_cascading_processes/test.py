import numpy as np
from model import CascadeNetwork

adj_matrix = np.matrix([[0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 1, 1, 1],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0]])

# Constant-load example from Figure 2, node_id([A-I])=[0-8]
load_vec_1 = [0] * 9
capacity_vec_1 = [0.7, 0.7, 0, 0.3, 0.5, 0.55, 0.55, 0.55, 0.55]

# Constant-load example from Figure 13, node_id([A-I])=[0-8]
load_vec_2 = [0] * 9
capacity_vec_2 = [0.7, 0.7, 0.3, 0.3, 0.5, 0.55, 0.55, 0.55, 0]

# Constant-load example from Figure 14, node_id([A-I])=[0-8]
load_vec_3 = [0] * 9
capacity_vec_3 = [0.7, 0.7, 0.3, 0.3, 0, 0.55, 0.55, 0.55, 0.55]

# Load redistribution example from Figure 3, node_id([A-I])=[0-8]
load_vec_4 = [1] * 9
capacity_vec_4 = [1.7, 1.7, 1, 1.3, 1.5, 1.55, 1.55, 1.55, 1.55]

# Load redistribution example from Figure 15, node_id([A-I])=[0-8]
load_vec_5 = [1] * 9
capacity_vec_5 = [1.7, 1.7, 1.3, 1, 1.5, 1.55, 1.55, 1.55, 1.55]

# Load redistribution example from Figure 16, node_id([A-I])=[0-8]
load_vec_6 = [1] * 9
capacity_vec_6 = [1.7, 1.7, 1.3, 1.3, 1.5, 1.55, 1.55, 1.55, 1]

# Overload distribution example from Figure 4, node_id([A-I])=[0-8]
load_vec_7 = [1, 1, 3.6, 1, 1, 1, 1, 1, 1]
capacity_vec_7 = [1.7, 1.7, 1.3, 1.3, 1.5, 1.55, 1.55, 1.55, 1.55]

model = CascadeNetwork(adj_matrix, load_vec_6, capacity_vec_6,
                       model_type="load", load_type="llss", test=True)
model.run_model()
