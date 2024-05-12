from __future__ import print_function

import math
from sys import maxsize
from itertools import permutations

"""Simple travelling salesman problem between cities."""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model(adjmat):
    """Stores the data for the problem."""
    data = {'distance_matrix': adjmat, 'num_vehicles': 1, 'depot': 0}
    return data


def print_solution(manager, routing, solution):
    """Prints solution on console."""
    print('Objective: {} units'.format(solution.ObjectiveValue()))
    index = routing.Start(0)
    plan_output = 'Route for UAV:\n'
    route_distance = 0
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    print(plan_output)
    plan_output += 'Route distance: {} units \n'.format(route_distance)


def main(adjmat):
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model(adjmat)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(manager, routing, solution)


file1 = open('centroid.data', 'r')
handle = file1.readlines()
coords = list()
for line in handle:
    a, b = line.strip().split()
    x = int(a)
    y = int(b)
    coords.append([x, y])

adjmat = list()
n = len(coords)
for i in range(n):
    row = list()
    for j in range(n):
        if i == j:
            row.append(0.0)
        else:
            dist = int(
                math.sqrt(math.pow((coords[i][0] - coords[j][0]), 2) + math.pow((coords[i][1] - coords[j][1]), 2)))
            # d = round(dist, 2)
            row.append(dist)
    adjmat.append(row)

main(adjmat)
