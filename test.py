#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Patrice Bechard
@email: bechardpatrice@gmail.com
Created on Wed Nov 15 08:28:36 2017

PowerNet

"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

G_SIZE = 100


#initialize graph
G = nx.Graph()
G.add_node(0)
G.add_node(1)
G.add_edge(0, 1)



for i in range(G_SIZE):

    #growth
    G.add_node(G.number_of_nodes())

    for j in range(G.number_of_nodes() - 1):
        #preferential attachment
        prob = np.random.uniform()
        if prob <= G.degree(j) / (G.number_of_edges() * 2):
            G.add_edge(j, G.number_of_nodes() - 1)

    if G.degree(G.number_of_nodes() - 1) == 0:
        #node was not linked to graph, we delete it
        G.remove_node(G.number_of_nodes() - 1)


print(G.number_of_nodes())
print(G.number_of_edges())

list_degrees = list(G.degree([i for i in range(G.number_of_nodes())]))

print(list_degrees)


#show graph
nx.draw(G, with_labels=True)