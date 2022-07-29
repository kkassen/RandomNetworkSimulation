"""
Authors: Kyle Kassen, Tate Teague
Dataset Source: https://networkrepository.com/rec-amazon.php
Cite: A First Course in Network Science By Clayton A. Davis, Filippo Menczer, and Santo Fortunato
      https://github.com/CambridgeUniversityPress/FirstCourseNetworkScience
"""

# import statements
import itertools
import random
import networkx as nx
import MonteCarlo as mc
import DrawGraphs as dg
import numpy as np
import numpy.random as rand
import time
from markov_clustering import get_clusters, run_mcl

class RandomNetworks(mc.MonteCarlo):
    def __init__(self, networks):
        self.networks = networks

    # Markov Clustering Algorithm
    # Cite: Data Science Bookcamp: Five Real-World Python Projects By Leonard Apeltsin
    # Cite: https://markov-clustering.readthedocs.io/en/latest/index.html
    # Cite: https://micans.org/mcl/
    def MarkovClustering(self, G):
        s = set()
        adjacency_matrix = nx.to_numpy_array(G)
        clusters = get_clusters(run_mcl(adjacency_matrix))
        for cluster in clusters:
            s.add(cluster)
        return frozenset(s)

    def erdosRenyi(self):
        # Takes combinations of different nodes to create new edges
        # and then randomly samples from list to get same number as before, modified from https://bit.ly/3w4n7jz
        G = nx.Graph()
        possible_edges = itertools.combinations(raw_network.nodes, 2)
        edges_to_add = random.sample(list(possible_edges), len(raw_network.edges))
        G.add_edges_from(edges_to_add)
        self.erdosRenyiGraph = G
        return G

    # Swtich statement for iterating through our Random Network Models
    def Switch(self, x):
        case = {
                 0: raw_network,
                 1: nx.barabasi_albert_graph(n=len(raw_network.nodes), m=rand.randint(1, 2), seed=rand.randint(1, 100)),
                 2: nx.stochastic_block_model(s, p, seed=rand.randint(1, 100)),
                 3: RandomNetworks(networks=networks).erdosRenyi()
               }
        return case.get(x)

    def createPartitionMap(self, partition):
        # PARAMS: partition RETURNS: partition map; Creates a partition map to break the network into clusters
        # Cite: https://bit.ly/3q4ob3k
        partition_map = {}
        for idx, cluster_nodes in enumerate(partition):
            for node in cluster_nodes:
                partition_map[node] = idx
        return partition_map

    def girvan_newman_partition(self, graph, numClusters):
        # PARAMS: graph object and # of clusters RETURNS clustered partitions that can be fed in as node colors
        # Method of generating partitions and returns clustered partitions of network
        partition = list(nx.community.girvan_newman(graph))[numClusters - 2]
        partitionMap = self.createPartitionMap(partition)
        return [partitionMap[i] for i in graph.nodes()]

    # SimulateOnce Function
    def SimulateOnce(self):
        results = []
        g = self.Switch(model)

        # Degree Centrality Metric
        degrees = [g.degree(j) for j in g.nodes]
        avgDegree = np.mean(degrees)
        results.append(avgDegree)

        # Betweenness Centrality Metric
        betweenness = list(nx.centrality.betweenness_centrality(g).values())
        avgBetweenness = np.mean(betweenness)
        results.append(avgBetweenness)

        # Connectedness Metric (Avg Shortest Path)
        avgConnectedness = []
        for i in (g.subgraph(i).copy() for i in nx.connected_components(g)):
            avgConnectedness.append(nx.average_shortest_path_length(i))
        avgConnectedness = np.mean(avgConnectedness)
        results.append(avgConnectedness)

        # print(results)
        return results

# Start time
start = time.time()

# List for our Random Network Models
networks = []

########################################################################################################################
# RAW NETWORK

# Retrieving our dataset
raw_network = nx.read_edgelist('amz.txt')
networks.append('Raw Network')
# Creating a visual of the Raw Network
dg.DrawGraphs.DrawRawGraph(raw_network)

########################################################################################################################
# BARABASI-ALBERT MODEL

BA_Random_Network = nx.barabasi_albert_graph(n=len(raw_network.nodes), m=rand.randint(1, 2), seed=rand.randint(1, 100))
networks.append('Barabasi-Albert Model')
# Creating a visual of the Barabasi-Albert Model
dg.DrawGraphs.DrawBarabasiAlbertGraph(BA_Random_Network)

# Barabasi-Albert with Girvan-Newman Clustering Algorithm
node_colors = RandomNetworks(networks=networks).girvan_newman_partition(graph=BA_Random_Network, numClusters=4)
# Creating a visual of the Barabasi-Albert Model with Girvan-Newman Clustering Algorithm
dg.DrawGraphs.DrawBAClustering(BA_Random_Network, colors=node_colors)

# Markov Clustering Algorithm applied to our Barabasi-Albert Model
markovClustering = RandomNetworks(networks=networks).MarkovClustering(G=BA_Random_Network)
partitionMap = RandomNetworks(networks=networks).createPartitionMap(markovClustering)
node_colors = [partitionMap[n] for n in BA_Random_Network.nodes()]
dg.DrawGraphs.DrawMarkovClustering(BA_Random_Network, node_colors)

########################################################################################################################
# STOCHASTIC BLOCK MODEL

s = [round(len(raw_network) * .30), round(len(raw_network) * .30), round(len(raw_network) * .40)]
p = [[0.25, 0.05, 0.02],
     [0.05, 0.35, 0.07],
     [0.02, 0.07, 0.40]]
SBM_Random_Network = nx.stochastic_block_model(s, p, seed=rand.randint(1, 100))
networks.append('Stochastic Block Model')
# Creating a visual of the Stochastic Block Model
dg.DrawGraphs.DrawStochasticBlockModel(SBM_Random_Network)

########################################################################################################################
# ERDOS-RENYI MODEL

ER_Random_Network = RandomNetworks(networks=networks).erdosRenyi()
networks.append('Erdos-Renyi Model')
# Creating a visual of the Erdos-Renyi Model
dg.DrawGraphs.DrawErdosRenyiGraph(ER_Random_Network)

# Erdos-Renyi with Girvan-Newman Clustering Algorithm
node_colors = RandomNetworks(networks=networks).girvan_newman_partition(graph=ER_Random_Network, numClusters=4)
# Creating a visual of the Erdos-Renyi Model with Girvan-Newman Clustering Algorithm
dg.DrawGraphs.DrawERClustering(ER_Random_Network, node_colors)

########################################################################################################################
# Looping through our list of Random Network Models

for model in range(len(networks)):
    # Creating RandomNetworks object
    rn = RandomNetworks(networks)
    # Title of Model for logging our results to output terminal
    print(networks[model])
    # Calling RunSimulation on SimCount=10
    result = rn.RunSimulation(simCount=10)
    # Logging our results to output terminal
    print("result: " + str(result))
    print("Average Degree: " + str(result[0]))
    print("Average Betweenness: " + str(result[1]))
    print("Average Connectedness: " + str(result[2]))
    print()

# How long did it take our program to run?
print("Time: " + str(time.time() - start))
