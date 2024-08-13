import mesa
import numpy as np
import networkx as nx
import random

from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np

import random
import networkx as nx
from scipy.sparse import coo_matrix
import numpy as np


class Agent(mesa.Agent):
    """
    An agent in the Koenig model, simulating decision making in a network.
    """

    def __init__(self, model:"World", unique_id):
        super().__init__(unique_id, model)

    # ==== Properties ====
    @property
    def uid(self):
        """unique agent identifier"""
        return self.unique_id

    @property
    def cost(self):
        """cost of sustaining a sustaining a link"""
        return self.model.cost

    @property
    def severance_cost(self):
        """Reduction in benefit from link severance"""
        return self.model.severance_cost

    def degree(self):
        """number of neighbors"""
        return self.model.net.degree(self.uid)

    def neighbors(self):
        """generate a sequence of Agents which are neighbors"""
        for uid in self.model.net.neighbors(self.uid):
            yield self.model.schedule.agents[uid]

    def non_neighbors(self):
        """generate a sequence of Agents which are not neighbor"""
        neighbors = set(a.uid for a in self.neighbors())
        all_nodes = set(self.model.uids()) - {self.uid}
        valid_uids = all_nodes - neighbors
        for uid in valid_uids:
            yield self.model.schedule.agents[uid]

    def random_neighbor(self):
        """Return a random neighbor, if no neighbors are available `None` is returned"""
        valid_set = list(self.neighbors())
        if valid_set:
            return random.choice(valid_set)
        return None

    def random_non_neighbors(self):
        """Return a random non-neighbor, if all agents are neighbors `None` is returned"""
        valid_set = list(self.non_neighbors())
        if valid_set:
            return random.choice(valid_set)
        return None

    def component_uids(self, network=None):
        """returns the uids of component in which this agent resides

        Args:
            network (Graph): the network from which the components should be extracted (optional)
                defaults to the current actual network.

        Returns:
            list
        """
        if network is None:
            network = self.model.net
        my_comp = [c for c in nx.connected_components(network) if self.uid in c][0]
        return my_comp

    def subgraph(self, network=None, copy=False):
        """The graph induced by selecting only the component to which the agent belongs.

        Args:
            network (Graph): the network from which the subgraph should be extracted (optional)
                defaults to the current actual network.
            copy (bool): True for a fully editable copy of the subgraph or False for a
                faster "View" of the graph

        Returns:
            Graph
        """
        if network is None:
            my_component = self.component_uids()
            network = self.model.net
        else:
            my_component = self.component_uids(network)

        sub_graph = nx.subgraph(network, my_component)
        if copy:
            return sub_graph.copy()
        return sub_graph

    # === Basic Actions ===
    def create_link_to(self, other):
        """Create a connection between this agent and the other agent.

        Args:
            other (Agent): Agent to link to
        """
        self.model.create_link(self, other)

    def delete_link_to(self, other):
        """Deleted link to other agent

        Args:
            other (Agent): Agent to remove link to.
        """
        self.model.delete_link(self, other)

    # === decision variables ===
    def utility(self, subgraph):
        """
        The utility for the agent given his position in the network.

        Args:
            subgraph (Graph): the subgraph for which the utility should be computed (optional)
                defaults to the current actual network.

        Returns:
            float
        """

        eigen_value = self.leading_eigen_value(subgraph)
        degree = subgraph.degree[self.uid]
        utility = eigen_value - self.cost * degree  # eq (6)

        return utility

    def leading_eigen_value(self, subgraph):
        """Compute the leading eigenvalue for the given agent. If no subgraph is specified then
        the current subgraph the agent is in will be used.

        Args:
            subgraph (Graph): the subgraph for which the leading eigen value should be computed
                (optional) defaults to the current actual network.

        Returns:
            float
        """

        n_subgraph = len(subgraph)

        if n_subgraph == 1:  # alone
            return 0.
        elif n_subgraph == 2:  # a pair (completely connected)
            return 1.

        adj_matrix = _adjacency_mat(subgraph, dense=True)
        eig_values = np.linalg.eigvals(adj_matrix)
        leading_ev = np.max(eig_values.real)
        return leading_ev

    # === potential outcomes ===
    def subgraph_if(self, add_link_to=None, remove_link_to=None):
        """Construct a hypothetical subgraph if the given changes were to be applied.

        Args:
            add_link_to (list): a list of agents
            remove_link_to (list): a list of agents

        Returns:
            Graph

        Examples:
            >>> w = World(n=2)
            >>> agent0, agent1 = w[0], w[1]
            >>> sub_g = agent0.subgraph()
            >>> list(sub_g.edges())  # no edges yet created
            []
            >>> sub_g = agent0.subgraph_if(add_link_to=agent1)
            >>> list(sub_g.edges())  # edges if the link were to be created
            [(0, 1)]
        """
        if add_link_to is None and remove_link_to is None:
            raise ValueError(
                "At least one of `add_connections` or `remove_connections` must be"
                "specified."
            )

        if add_link_to and not hasattr(add_link_to, "__iter__"):
            add_link_to = [add_link_to]

        if remove_link_to and not hasattr(remove_link_to, "__iter__"):
            remove_link_to = [remove_link_to]

        edges_to_remove_after = []
        if add_link_to:
            for other in add_link_to:
                if not self.model.link_exists(self, other):
                    self.create_link_to(other)
                    edges_to_remove_after.append(other)

        edges_to_add_after = []
        if remove_link_to:
            for other in remove_link_to:
                if self.model.link_exists(self, other):
                    self.delete_link_to(other)
                    edges_to_add_after.append(other)

        # save a copy of the focal subgraph
        net = self.model.net
        subgraph = self.subgraph(net, copy=True)

        # Revert Changes
        for other in edges_to_remove_after:
            self.delete_link_to(other)

        for other in edges_to_add_after:
            self.create_link_to(other)

        return subgraph

    def gain_from_link(self, other, create=True):
        """Compute the gain if the proposed link is created/deleted from the focal agents
        perspective

        Args:
            other (Agent):
            create (bool):

        Returns:
            float

        Examples:
            >>> w = World(n=2)
            >>> agent0, agent1 = w[0], w[1]
            >>> agent0.gain_from_link(agent1)
            1.0
        """
        links_exists_already = self.model.link_exists(self, other)
        if create is True and links_exists_already:
            return 0.0

        if create is False and not links_exists_already:
            return 0.0

        if create is True:
            potential_subgraph = self.subgraph_if(add_link_to=other)
        elif create is False:
            potential_subgraph = self.subgraph_if(remove_link_to=other)
        else:
            raise ValueError(f"`create` must be either True or False, not {create}")

        actual_ev = self.leading_eigen_value(self.subgraph())
        potential_ev = self.leading_eigen_value(potential_subgraph)

        if create is True:
            return potential_ev - actual_ev - self.cost
        else:
            return potential_ev - actual_ev + self.cost * (1 - self.severance_cost)

    def rank_agents(self, create=True, max_sample=None):
        """Create a ranking of `max_sample` agents for the focal agent to create/delete a link to

        Args:
            create (bool):
            max_sample (int): size of sample to compute the ranking on, if not set defaults to all
                agents.

        Returns:
            list of sorted [(gain, uid1), ...]

        Examples:
            >>> w = World(n=4)
            >>> agent0 = w[0]
            >>> agent0.create_beneficial_link(w[3])
            >>> agent0.rank_agents(create=True)
            [(0.4142135623730949, 2), (0.4142135623730949, 1)]
        """
        if create is True:
            valid_set = list(self.non_neighbors())
        elif create is False:
            valid_set = list(self.neighbors())
        else:
            raise ValueError(f"`create` must be either True or False, not {create}")

        if max_sample and max_sample < len(valid_set):
            chosen_uids = random.choices(valid_set, k=max_sample)
        else:
            random.shuffle(valid_set)
            chosen_uids = valid_set

        gains_from_agent = {}
        for other_agent in chosen_uids:
            gain = self.gain_from_link(other_agent, create=create)
            if gain > 0:
                gains_from_agent[other_agent.uid] = gain

        gains_and_uids = [(gain, agent) for agent, gain in gains_from_agent.items()]
        sorted_agents = sorted(gains_and_uids, reverse=True)
        return sorted_agents

    def best_simple_action(self, create=True, mutual=True, max_sample=None):
        """Return the agent with whom a link created/delete would yield the highest utility gain

        Args:
            create (bool): if True then the the best "link creation" target is chosen otherwise
                deletion.
            mutual (bool): default True, if `mutual` is True then then all possible candidates are
                ranked according to the gain they would provide if the action were to be executed.
                Then the first agent from this raking that would accept the offer (i.e. he does not
                lose) is chosen. This results in the highest possible payoff for the focal agent
                without reducing the gain of the other
            max_sample (int): number of agents to rank, if not specified all agents are considered
                valid targets. if a number is given the agent will randomly sample `max_samples`
                to compute his ranking.
        Returns:
            Agent or None, None indicates that the best action is to do nothing

        Examples:
            >>> w = World(n=2)
            >>> agent0 = w[0]
            >>> agent0.best_simple_action(create=True, mutual=True)
            (1.0, Agent(<World>, uid=1))
            >>> agent0.create_link_to(w[1])
            >>> agent0.best_simple_action(create=True, mutual=True)
            (0.0, None)
        """
        ranked_gains_uid = self.rank_agents(create, max_sample)

        if not ranked_gains_uid:
            return 0.0, None

        if not mutual:
            gain, chosen_uid = ranked_gains_uid[0]
            return gain, self.model.schedule.agents[chosen_uid]  # returns the first one

        for gain_from_other, other_uid in ranked_gains_uid:
            gain_other = self.model.schedule.agents[other_uid].gain_from_link(self, create)
            if gain_other >= 0:
                return gain_from_other, self.model.schedule.agents[other_uid]
        return 0.0, None

    def best_action(self, mutual_create=True, mutual_delete=True, max_sample=None):
        """Returns the type of action to perform and with which agent that would
        maximize the utility if the action were to be carried out.

        Args:
            mutual_create (bool): require that both agents do not suffer utility losses (True) or in
                case of False only consider the utility of the proposing agent.
            mutual_delete (bool): delete only if both gain or if False only if the focal gains, but
                not necessarily the other
            max_sample (int): how many agents to sample to make the decision

        Returns:
            bool, Agent: True -> create | False -> delete, Agent to perform the action on.

        Examples:
            >>> w = World(n=2)
            >>> agent0 = w[0]
            >>> agent0.best_action()
            (True, Agent(<World>, uid=1))
        """
        # best create action
        create_gain, top_create_target = self.best_simple_action(
            True, mutual_create, max_sample
        )
        # best delete action
        delete_gain, top_delete_target = self.best_simple_action(
            False, mutual_delete, max_sample
        )
        if top_create_target and top_delete_target:  # we have tow viable options
            if create_gain >= delete_gain:
                return True, top_create_target
            else:
                return False, top_delete_target
        elif top_create_target:  # only create is viable
            return True, top_create_target
        elif top_delete_target:  # only delete is viable
            return False, top_delete_target
        else:  # no viable option is available
            return None, None

    def link_would_be_beneficial(self, other, create=True, mutual=True):
        """Evaluate if a given link (creation/deletion) would be beneficial for the individual
        and if `mutual` is True also for the target agent.

        Args:
            other (Agent):
            create (bool): True -> Create link, False -> delete link
            mutual (bool): require that no agent is worse off

        Returns:
            bool
        """
        if self.uid == other.uid:  # ignore self-loops
            return False

        own_gain = self.gain_from_link(other, create=create)
        if own_gain < 0:
            return False
        elif not mutual:
            return own_gain > 0
        else:
            other_gain = other.gain_from_link(self, create=create)
            return (own_gain > 0) and (other_gain >= 0)

    # === Update Rules ===
    def best_action_update(
        self, mutual_creation=True, mutual_deletion=False, max_sample=None
    ):
        """Create/Delete or do nothing according to the action that maximizes the gain.

        Args:
            mutual_creation: require that neither agent is worse off when creating the link
            mutual_deletion: require that neither agent is worse off when deleting the link
            max_sample (int): sample of agents to evaluate create/delete action for,
                defaults to all agents

        """
        create, target_agent = self.best_action(
            mutual_creation, mutual_deletion, max_sample
        )
        if target_agent is not None:
            if create is True:
                self.create_link_to(target_agent)
            elif create is False:
                try:
                    self.delete_link_to(target_agent)
                except: # HACK: if the link does not exist
                    pass

    def create_beneficial_link(self, other=None, mutual=True):
        """If a beneficial link with the other agent can be created, create it.

        Args:
            other (Agent): specify Agent to create a link to if beneficial. If not agent is
                specified a random non-neighbor is picked.
            mutual (bool): require that neither agent is worse off
        """
        if other is None:
            other = self.random_non_neighbors()

        if other and self.link_would_be_beneficial(other, create=True, mutual=mutual):
            self.create_link_to(other)

    def delete_bad_link(self, other=None, mutual=True):
        """If a link with the other agent can be deleted while improving utility, delete it.

        Args:
            other (Agent): specify Agent to delete the link to if beneficial. If not agent is
                specified a random neighbor is picked.
            mutual (bool): require that neither agent is worse off
        """
        if other is None:
            other = self.random_neighbor()
        if other and self.link_would_be_beneficial(other, create=False, mutual=mutual):
            self.delete_link_to(other)

    def best_simple_action_update(self, create=True, mutual=True, max_sample=None):
        """Given the choice for this agent, which is the partner he would chose to create/delete a
        link to

        Args:
            create (bool): create (True) or delete (False) a link
            mutual (bool): require that neither agent is worse off when performing the action
            max_sample (int): sample of agents to evaluate create/delete action for, defaults to all
                agents
        """
        own_degree = self.degree()
        if create is True:
            if (
                own_degree < len(self.model) - 1
            ):  # is the agent already connected to everyone?
                _, top_partner = self.best_simple_action(True, mutual, max_sample)
                if top_partner:
                    self.create_link_to(top_partner)
        elif create is False:
            if self.degree() > 0:
                _, top_partner = self.best_simple_action(False, mutual, max_sample)
                if top_partner:
                    self.delete_link_to(top_partner)
        else:
            raise ValueError(f"`create` must be either True or False, not {create}")

    def step(self):
        max_sample = self.model.max_sample
        update_type = self.model.update_type
        mutual_create = self.model.mutual_create
        mutual_delete = self.model.mutual_delete

        create = random.random() < 1 - self.model.deletion_prob
        if update_type == "random":
            if create:
                self.create_beneficial_link(mutual=mutual_create)
            else:
                self.delete_bad_link(mutual=mutual_delete)
        elif update_type == "best_simple_choice":
            if create:
                self.best_simple_action_update(create, mutual_create, max_sample)
            else:
                self.best_simple_action_update(create, mutual_delete, max_sample)
        elif update_type == "best_choice":
            self.best_action_update(mutual_create, mutual_delete, max_sample)
        else:
            raise KeyError(f'The update_type "{update_type}" is not valid.')



class World(mesa.Model):
    """The World class represents the environment in which the Agents interact.

    The class keeps track of the Network on which the agents interact, the alterations to it
    over time as well as the model time
    """

    def __init__(self, num_agents, cost, severance_cost,
                 update_type,
                 deletion_prob,
                 mutual_create,
                 mutual_delete,
                 poling_interval=1,
                 check_stability=1,
                 max_steps=None,
                 max_sample=None
                 ):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.net = nx.Graph()
        self.max_sample = max_sample
        self.cost = cost
        self.severance_cost = severance_cost
        self._id_counter = 0
        self.poling_interval = poling_interval
        self.check_stability = check_stability
        self.update_type = update_type
        self.deletion_prob = deletion_prob
        self.mutual_create = mutual_create
        self.mutual_delete = mutual_delete
        self.max_steps = max_steps
        self.num_agents = num_agents

        self.add_agents(num_agents)

        self.datacollector = DataCollector(
            {
                "welfare": lambda m: m.aggregate_welfare(),
                "density": lambda m: m.density(),
                "clustering": lambda m: m.clustering(),
                "num_components": lambda m: len(list(m.connected_components())),
            }
        )

    # === Basic Actions ===
    def add_agents(self, num_agents):
        """Add the specified number of agents or ids in agent_list to the World.

        Args:
            num_agents (int):
            agent_ids (list):
        """
        for _ in range(num_agents):
            self.add_agent()

    def add_agent(self):
        """Add a single Agent to the simulation

        Args:
            uid:

        """
        uid = self._generate_uid() # generate a unique uid for the agent in the network
        if uid in self.net:
            raise KeyError(f"`uid={uid}` already taken")

        agent = Agent(self, uid)
        self.schedule.add(agent)
        self.net.add_node(uid)

    def remove_agent(self, agent_uid: int):
        """Remove the given agent uid from the World"""
        self.schedule.remove(self.agent(agent_uid))
        self.net.remove_node(agent_uid)

    def create_link(self, agent_1:Agent, agent_2:Agent):
        """Connect the two agents in the network"""
        self.net.add_edge(agent_1.uid, agent_2.uid)

    def delete_link(self, agent_1: Agent, agent_2: Agent):
        """Delete the link between the two agents in the network"""
        self.net.remove_edge(agent_1.uid, agent_2.uid)

    # === properties ===
    def uids(self):
        for agent in self.agents():
            yield agent.uid

    def agents(self):
        """A list of all agents currently active in the simulation

        Yields:
            Agent
        """
        for agent in self.schedule.agents:
            yield agent

    def agent(self, uid):
        """returns agent with with given uid"""
        return self.schedule._agents[uid]

    # === Network properties ===
    def aggregate_welfare(self, norm=True):
        """Compute aggregated welfare (i.e. the sum of all utilities)

        Args:
            norm (bool): if true then the welfare is in [0, 1] with 1 being the maximum attainable
            (i.e. fully connected graph)

        Returns:
            float
        """
        if not norm:
            welfare = sum(a.utility(a.subgraph()) for a in self.schedule.agents)
            return welfare
        else:
            num_agents = self.num_agents
            welfare = sum(a.leading_eigen_value(a.subgraph()) for a in self.schedule.agents)
            welfare /= num_agents * (num_agents - 1)
        return welfare

    def density(self):
        """Returns the proportion of possible edges which are actually present"""
        return nx.density(self.net)

    def clustering(self):
        """See networkx.average_clustering"""
        return nx.average_clustering(self.net)

    def degree(self):
        """Number of neighbors in the Graph"""
        return self.net.degree
    
    @property
    def measurements(self):
        df = self.datacollector.get_model_vars_dataframe()
        return df.T
    


    def connected_components(self):
        """Returns a generator containing the uids of the current network

        Yields:
            sets of uids in the same component
        """
        return nx.connected_components(self.net)

    def link_exists(self, agent_1: Agent, agent_2: Agent):
        """Return True if a link exists False otherwise"""
        edge = (agent_1.uid, agent_2.uid)
        return edge in self.net.edges()

    # === Stability Checks ===
    def is_pairwise_stable(self, agents=None, mutual_create=True, mutual_delete=True):
        """Return true if no agent would be willing to make a change (either create or delete)

        Args:
            agents (list of Agents): if the stability should only be checked for this subset pass a
                list of agents otherwise the stability is computed considering all agents
            mutual_create (bool): require that neither agent is worse off when creating the link
            mutual_delete (bool): require that neither agent is worse off when deleting the link

        Returns:
            Returns True if no agent wants to make a move, False otherwise
        """
        # short circuit logic to reduce the number of comparisons necessary to find that the
        # system is not stable (i.e. at least one agent wants to make a move).
        # If it is in fact stable all agents need to be tested
        deletion_stable = self.is_deletion_stable(agents, mutual=mutual_delete)

        if deletion_stable is False:  # the first test is False -> all is False
            return False

        # The first test has yielded True if the next test is False -> all is False
        creation_stable = self.is_creation_stable(agents, mutual=mutual_create)
        return creation_stable

    def is_deletion_stable(self, agents=None, mutual=True):
        """Returns True if no Agent would be willing to delete a link"""
        return self._is_simple_action_stable(agents, create=False, mutual=mutual)

    def is_creation_stable(self, agents=None, mutual=True):
        """Returns True if no Agent would be willing to create one more edge"""
        return self._is_simple_action_stable(agents, create=True, mutual=mutual)

    def _is_simple_action_stable(self, agents=None, create=True, mutual=True):
        """Given either create=True or create=False (i.e. delete) the function returns True if no
        Agent would be willing to perform an action of the given type."""
        if agents is None:
            agents = self.schedule.agents

        for agent in agents:
            _, other_agent = agent.best_simple_action(create, mutual)
            if other_agent is not None:
                return False
        return True

    def _generate_uid(self):
        """generate a unique uid"""
        valid_id_found = False
        uid = None
        while not valid_id_found:
            uid = self._id_counter
            valid_id_found = uid not in self.net
            self._id_counter += 1
        return uid

    def step(self):
        # at given interval report on the state of the system
        if self.schedule.time % self.poling_interval == 0:
            self.datacollector.collect(self)

        self.schedule.step()
        check_stability = self.check_stability
        update_type = self.update_type
        deletion_prob = self.deletion_prob
        mutual_create = self.mutual_create
        mutual_delete = self.mutual_delete


        if check_stability and (self.schedule.time % check_stability == 0):
            # only creation cases
            if (
                update_type in ["best_simple_choice", "random"]
                and deletion_prob <= 0
            ):
                if self.is_creation_stable(mutual=mutual_create):
                    # stop schedule
                    self.running = False
            elif update_type == "best_choice" or deletion_prob > 0:
                if self.is_pairwise_stable(
                    mutual_create=mutual_create, mutual_delete=mutual_delete
                ):
                    self.running = False
            else:
                raise ValueError(
                    f'update_type="{update_type}" is not a valid choice'
                )
        if self.max_steps and self.schedule.time >= self.max_steps:
            self.running = False

def _adjacency_mat(net, dense=True):
    """extract the adjacency matrix from the given networkx Graph.

    Args:
        net (Graph):
        dense (bool):

    Returns:

    """
    edges = net.edges()
    n_len = len(net)
    index = {n: i for i, n in enumerate(net.nodes)}
    coefficients = zip(*((index[s], index[t], 1) for s, t in edges))
    row, col, data = coefficients
    # for Symmetry
    s_data = data + data
    s_row = row + col
    s_col = col + row
    adj_mat = coo_matrix(
        (s_data, (s_row, s_col)), shape=(n_len, n_len), dtype=float
    )
    if not dense:
        return adj_mat
    return adj_mat.toarray()
