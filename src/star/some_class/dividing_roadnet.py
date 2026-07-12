#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import threading
import os
import time
from collections import defaultdict
import math
from matplotlib.lines import Line2D

class roadnet:
    def __init__(self):
        # Prevent duplicate reception.
        self.net_version = 0
        # Data used to visualize the road network.
        self.nodes = None
        self.edges = None
        # Data used to visualize different edge types.
        self.delete_edge_1 = None
        self.delete_edge_2 = None
        # Data used to visualize the path.
        self.path_point = None

    # Initialize an offline road network.
    def init_graph(self,graph_file):
        # Read node and edge data.
        with open(graph_file, 'r') as f:
            lines = f.readlines()
        # Parse node data.
        nodes = []
        for line in lines:
            line = line.strip()
            if not line:
                break
            parts = line.split()
            x, y,z = float(parts[0]), float(parts[1]),5
            nodes.append((x,y,z))
        # Parse edge data.
        edges = []
        for line in lines[len(nodes) + 1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            u, v,w = int(parts[0]), int(parts[1]),1
            edges.append((u, v,w))
        self.nodes = nodes
        self.edges = edges

    # Calculate the angle formed by three points, in the range [0, 180] degrees.
    def calculate_angle(self,point1, point2, point3):
        # Calculate vector 1.
        vector1 = (point1[0] - point2[0], point1[1] - point2[1])
        # Calculate vector 2.
        vector2 = (point3[0] - point2[0], point3[1] - point2[1])
        # Calculate the dot product of vectors 1 and 2.
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        # Calculate the magnitudes of vectors 1 and 2.
        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
        # Calculate the angle in radians.
        angle_rad = math.acos(dot_product / (magnitude1 * magnitude2))
        # Convert radians to degrees.
        angle_deg = math.degrees(angle_rad)
        return angle_deg


    def get_other_vertex(self,u, v, vertex_connections):
        # Get another connected vertex other than vertices u and v.
        for vertex in vertex_connections[u]:
            if vertex != v:
                return vertex
        for vertex in vertex_connections[v]:
            if vertex != u:
                return vertex

    def dividing(self):
        # Count the vertices connected to each vertex.
        vertex_connections = defaultdict(set)
        for edge in self.edges:
            u, v, _ = edge
            vertex_connections[u] .add(v)
            vertex_connections[v] .add(u)

        # Set edge weights according to the number of connected vertices.
        for i, edge in enumerate(self.edges):
            u, v, _ = edge
            connections_max =max(len(vertex_connections[u]), len(vertex_connections[v]))
            if connections_max == 2:
                other_vertex = self.get_other_vertex(u, v, vertex_connections)
                x = [self.nodes[u][0], self.nodes[u][1]]
                y = [self.nodes[v][0], self.nodes[v][1]]
                z=  [self.nodes[other_vertex][0], self.nodes[other_vertex][1]]
                angle = self.calculate_angle(x,y,z)
                if angle < 50 and angle >10:
                    self.edges[i] = (u, v, 2)
            elif connections_max == 3:
                self.edges[i] = (u, v, 3)
            elif connections_max == 4:
                self.edges[i] = (u, v, 4)
            else:
                self.edges[i] = (u, v, 1)

            # Keep the weight consistent in both directions of an edge.
            for i, edge in enumerate(self.edges):
                u, v, _ = edge
                if (v, u, 2) in self.edges and (u, v, 1) in self.edges:
                    self.edges[i] = (u, v, 2)

    def save3D(self, graph_file):
     with open(graph_file, 'w') as f:
        # Write node data.
        for node in self.nodes:
            x, y, z = node
            f.write(f"{x} {y} {z}\n")
        f.write("\n")
        # Write edge data.
        for edge in self.edges:
            u, v, w = edge
            f.write(f"{u} {v} {w}\n")

    def drawing(self):
        '''Draw the road-network image.'''
        # Clear the current figure.
        plt.clf()
        # Draw the road network and path after receiving road-network data.
        if self.nodes is not None:
            # Do not draw nodes because there are too many.
            # Draw edges.
            for i, (u, v, w) in enumerate(self.edges):
                x = [self.nodes[u][0], self.nodes[v][0]]
                y = [self.nodes[u][1], self.nodes[v][1]]
                # L-shaped intersection.
                if w == 2:
                    plt.plot(x, y, linewidth=10, color='darkblue')
                # T-shaped intersection.
                elif w == 3:
                    plt.plot(x, y, linewidth=10, color='darkred')
                # Four-way intersection.
                elif w == 4:
                    plt.plot(x, y, linewidth=10, color='darkorange')
                # Normal straight road.
                else:
                    plt.plot(x, y, linewidth=10, color='darkgreen')
            # Set the image title and axis labels.
            plt.title('Road_Net', fontsize=20)
            plt.xlabel('x')
            plt.ylabel('y')
            # Use the same scale for both axes.
            plt.axis('equal')
            # Add the legend.
            plt.legend()
            plt.savefig('test/05_2D.png')
             # Refresh the figure.
            plt.pause(1)




if __name__ == '__main__':
    # Instantiate the class.
    okk=roadnet()
    # Read the road-network data file.
    current_dir = os.path.dirname(__file__)
    graph_file = os.path.join(current_dir,  '05.graph')
    okk.init_graph(graph_file)
    # Divide the road network.
    okk.dividing()
    # Save the road-network file for 3D visualization.
    graph_file_3D = "test/05_3D.graph"
    okk.save3D(graph_file_3D)
    # Visualize the road network in 2D.
    okk.drawing()


