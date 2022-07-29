"""
Authors: Kyle Kassen, Tate Teague
Dataset Source: https://networkrepository.com/rec-amazon.php
Cite: A First Course in Network Science By Clayton A. Davis, Filippo Menczer, and Santo Fortunato
      https://github.com/CambridgeUniversityPress/FirstCourseNetworkScience
"""

import matplotlib.pyplot as plt
import networkx as nx

class DrawGraphs:

    # generates a visual of our Raw Network dataset
    def DrawRawGraph(network):
        layout = nx.spring_layout(network)
        plt.figure(figsize=(15, 15))
        plt.axes().set_facecolor('#FAF9F6')
        nx.draw_networkx_nodes(network, layout, node_size=80, node_color='#BF00FF', edgecolors='#3A242B', linewidths=1.5)
        nx.draw_networkx_edges(network, layout, alpha=0.6)
        plt.title('Figure 1.0: Raw Network', fontsize=30)
        plt.show()

    # generates a visual of our Barabasi-Albert Model
    def DrawBarabasiAlbertGraph(network):
        layout = nx.spring_layout(network)
        plt.figure(figsize=(15, 15))
        plt.axes().set_facecolor('#FAF9F6')
        nx.draw_networkx_nodes(network, layout, node_size=80, node_color='#0070BB', edgecolors='#002E63', linewidths=1.5)
        nx.draw_networkx_edges(network, layout, alpha=0.4)
        plt.title('Figure 1.1: Barabasi-Albert Model', fontsize=30)
        plt.show()

    # generates a visual of the Girvan-Newman Clustering Algorithm applied to our Barabasi-Albert Model
    def DrawBAClustering(network, colors):
        layout = nx.kamada_kawai_layout(network)
        plt.figure(figsize=(15, 15))
        plt.axes().set_facecolor('#FAF9F6')
        nx.draw_networkx_nodes(network, layout, node_size=80, node_color=colors, linewidths=1.5)
        nx.draw_networkx_edges(network, layout, alpha=0.4)
        plt.title('Figure 1.2: Barabasi-Albert Model: Girvan-Newman Clustering Algorithm', fontsize=25)
        plt.show()

    # generates a visual of the Markov Clustering Algorithm applied to our Barabasi-Albert Model
    def DrawMarkovClustering(network, colors):
        layout = nx.kamada_kawai_layout(network)
        plt.figure(figsize=(15, 15))
        plt.axes().set_facecolor('#FAF9F6')
        nx.draw_networkx_nodes(network, layout, node_size=80, node_color=colors, linewidths=1.5)
        nx.draw_networkx_edges(network, layout, alpha=0.4)
        plt.title('Figure 1.3: Barabasi-Albert Model: Markov Clustering Algorithm', fontsize=30)
        plt.show()

    # generates a visual of our Stochastic Block Model
    def DrawStochasticBlockModel(network):
        layout = nx.spring_layout(network)
        plt.figure(figsize=(15, 15))
        plt.axes().set_facecolor('#FAF9F6')
        nx.draw_networkx_nodes(network, layout, node_size=80, node_color='#00ff00', edgecolors='#21421E', linewidths=1.5)
        nx.draw_networkx_edges(network, layout, alpha=0.2)
        plt.title('Figure 1.4: Stochastic-Block Model', fontsize=30)
        plt.show()

    # generates a visual of our Erdos-Renyi Model
    def DrawErdosRenyiGraph(network):
        layout = nx.spring_layout(network)
        plt.figure(figsize=(15, 15))
        plt.axes().set_facecolor('#FAF9F6')
        nx.draw_networkx_nodes(network, layout, node_size=80, node_color='#FF9966', edgecolors='#FF7538', linewidths=1.5)
        nx.draw_networkx_edges(network, layout, alpha=0.4)
        plt.title('Figure 1.5: Erdos-Renyi Model', fontsize=30)
        plt.show()

    # generates a visual of the Girvan-Newman Clustering Algorithm applied to our Erdos-Renyi Model
    def DrawERClustering(network, colors):
        layout = nx.spring_layout(network)
        plt.figure(figsize=(15, 15))
        plt.axes().set_facecolor('#FAF9F6')
        nx.draw_networkx_nodes(network, layout, node_size=80, node_color=colors, linewidths=1.5)
        nx.draw_networkx_edges(network, layout, alpha=0.4)
        plt.title('Figure 1.6: Erdos-Renyi Model: Girvan-Newman Clustering Algorithm', fontsize=30)
        plt.show()

