"""
This script implements heuristics for the AGV assisted warehouse routing problem under the assumption of a DEDICATED
storage policy
"""

# example to use
import numpy as np
import xml.etree.cElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import datetime
import pickle
import traceback
import os
import os.path
from os import path
import math
import json
import time
from operator import itemgetter, attrgetter
from xml.dom import minidom
import rafs_instance as instance
from collections import defaultdict, Iterable
from utils import *
import copy
import random

# ------------------------------------------- CONFIG -------------------------------------------------------------------
SKU = "24"  # options: 24 and 360
SUBSCRIPT = ""
NUM_ORDERS = 20

layoutFile = r"data/layout/1-1-1-2-1.xlayo"
podInfoFile = "data/sku{}/pods_infos.txt".format(SKU)

instances = {}
instances[24, 2] = r"data/sku{}/layout_sku_{}_2.xml".format(SKU, SKU)

storagePolicies = {}
storagePolicies["dedicated"] = "data/sku{}/pods_items_dedicated_1.txt".format(SKU)
# storagePolicies['mixed'] = 'data/sku{}/pods_items_mixed_shevels_1-5.txt'.format(SKU)

orders = {}
orders[
    "{}_5".format(str(NUM_ORDERS))
] = r"data/sku{}/orders_{}_mean_5_sku_{}{}.xml".format(
    SKU, str(NUM_ORDERS), SKU, SUBSCRIPT
)
# orders['10_5'] = r'data/sku{}/orders_10_mean_1x6_sku_{}.xml'.format(SKU, SKU)

# ----------------------------------------------------------------------------------------------------------------------


class WarehouseDateProcessing:
    def __init__(self, warehouseInstance, batch_size=None):
        self.Warehouse = warehouseInstance
        self._InitSets(warehouseInstance, batch_size)

    def preprocessingFilterPods(self, warehouseInstance):
        resize_pods = {}
        print("preprocessingFilterPods")
        item_id_list = []
        for order in warehouseInstance.Orders:
            for pos in order.Positions.values():
                item = (
                    warehouseInstance.ItemDescriptions[pos.ItemDescID].Color.lower()
                    + "/"
                    + warehouseInstance.ItemDescriptions[pos.ItemDescID].Letter
                )
                # item_id = pos.ItemDescID
                if item not in item_id_list:
                    item_id_list.append(item)

        # for dedicated
        for pod in warehouseInstance.Pods.values():
            for item in pod.Items:
                # print("item in pod.Items:", item.ID)
                if item.ID in item_id_list:
                    print("item.ID in item_id_list:", item.ID)
                    resize_pods[pod.ID] = pod

        print(resize_pods)
        return resize_pods

    # Initialize sets and parameters
    def _InitSets(self, warehouseInstance, batch_size):
        # V Set of nodes, including shelves V^S and stations (depots)
        # V^D (V=V^S U V^D)
        # Add output and input depots
        self.V__D__C = warehouseInstance.OutputStations

        self.V__D__F = {}
        self.V__D = {**self.V__D__C, **self.V__D__F}
        # hli
        # self.V__S = warehouseInstance.Pods
        self.V__S = self.preprocessingFilterPods(warehouseInstance)
        # Merge dictionaries
        self.V = {**self.V__D, **self.V__S}

    def CalculateDistance(self):

        file_path = (
            r"data/distances/"
            + os.path.splitext(os.path.basename(self.Warehouse.InstanceFile))[0]
            + ".json"
        )

        if not path.exists(file_path):
            # Create d_ij
            # d_ij = tupledict()
            d_ij = {}

            # Loop over all nodes
            for key_i, node_i in self.V.items():
                for key_j, node_j in self.V.items():
                    source = "w" + node_i.GetPickWaypoint().ID
                    target = "w" + node_j.GetPickWaypoint().ID

                    # Calc distance with weighted shortest path
                    d_ij[(key_i, key_j)] = nx.shortest_path_length(
                        self.Graph, source=source, target=target, weight="weight"
                    )

            # Parse and save
            d_ij_dict = {}
            for key, value in d_ij.items():
                i, j = key
                if i not in d_ij_dict:
                    d_ij_dict[i] = {}
                d_ij_dict[i][j] = value

            with open(file_path, "w") as fp:
                json.dump(d_ij_dict, fp)

        else:
            # Load and deparse
            with open(file_path, "r") as fp:
                d_ij_dict = json.load(fp)
            print("d_ij file %s loaded" % file_path)
            d_ij = {}
            for i, values in d_ij_dict.items():
                for j, dist in values.items():
                    d_ij[i, j] = dist

        return d_ij


class Demo:
    def __init__(self, splitOrders=False):

        self.batch_weight = 18
        self.item_id_pod_id_dict = {}
        self.station_id_bot_id_dict = dict()
        # [0]
        self.warehouseInstance = self.prepareData()
        self.distance_ij = self.initData()
        # [2]
        if storagePolicies.get("dedicated"):
            self.is_storage_dedicated = True
        else:
            self.is_storage_dedicated = False
        # self.batches = None

    # warehouse instance
    def prepareData(self):
        print("[0] preparing all data with the standard format: ")
        # Every instance
        for key, instanceFile in instances.items():
            podAmount = key[0]
            depotAmount = key[1]
            # For different orders
            for key, orderFile in orders.items():
                orderAmount = key
                # For storage policies
                for storagePolicy, storagePolicyFile in storagePolicies.items():
                    warehouseInstance = instance.Warehouse(
                        layoutFile,
                        instanceFile,
                        podInfoFile,
                        storagePolicyFile,
                        orderFile,
                    )
        return warehouseInstance

    # distance
    def initData(self):
        print("[1] changing data format for the algorithm we used here: ")
        warehouse_data_processing = WarehouseDateProcessing(self.warehouseInstance)
        # Distance d_ij between two nodes i,j \in V
        d_ij = warehouse_data_processing.CalculateDistance()
        return d_ij

    # get pod for each item
    def build_trie_for_items(self):
        trie = defaultdict(dict)
        for key, item in self.warehouseInstance.ItemDescriptions.items():
            trie[item.Letter][item.Color] = item.ID
        return trie

    def fill_item_id_pod_id_dict(self):
        """
        This function build a dictionary with item IDs as keys and the shelf the respective item is stored in as value
        """
        trie = self.build_trie_for_items()
        for key, pod in self.warehouseInstance.Pods.items():
            for item in pod.Items:
                ids = item.ID
                color, product = ids.split("/")
                item_id = trie[product][color]
                self.item_id_pod_id_dict[item_id] = key

    def fill_station_id_bot_id_dict(self):
        """
        This function build a dictionary with pack stations as keys and their respective cobot as value
        """
        for key, bot in self.warehouseInstance.Bots.items():
            self.station_id_bot_id_dict[bot.OutputStation] = key

    # define get functions for better readability and accessibility of data
    def get_items_by_order(self, order_idx):
        """
        Retrieves all item IDs that are contained in an order. Here, only distinct items are retrieved, as the
        order quantity does not really matter (we assume shelves have enough items to satisfy demand)
        :param order_idx: ID of the order (as stored in self.warehouseInstance.Orders)
        :return: Item IDs
        """
        item_ids = []
        for item_id, item in self.warehouseInstance.Orders[order_idx].Positions.items():
            item_ids.append(item.ItemDescID)
        return item_ids

    def get_weight_by_item(self, item_idx):
        """
        getter for the weight of an item
        """
        return self.warehouseInstance.ItemDescriptions[item_idx].Weight

    def get_total_weight_by_order(self, order_idx):
        """
        calculates the total weight of an order by summing up the weight of the respective items multiplied with
        their order quantity.
        :param order_idx: ID of order
        :return: weight of order
        """
        weights = []
        items_of_order = self.get_items_by_order(order_idx)
        for item in items_of_order:
            quant_of_item = int(
                self.warehouseInstance.Orders[order_idx].Positions[item].Count
            )
            weights.append(self.get_weight_by_item(item) * quant_of_item)
        return np.sum(weights)

    def get_distance_between_items(self, item1, item2):
        """
        getter for the distance between items. Must first retrieve the shelf the items are stored in
        """
        shelf1 = self.item_id_pod_id_dict[item1]
        shelf2 = self.item_id_pod_id_dict[item2]
        return self.distance_ij[shelf1, shelf2]

    def get_fitness_of_batch(self, batch: Batch):
        """
        calculates the fitness of a batch. Batch must be passed as Batch object as defined in utils.py
        Fitness is defined as the total distance a cobot has to move along the warehouse in order to complete the batch
        """
        distances = []
        distances.append(
            self.distance_ij[
                batch.pack_station, self.item_id_pod_id_dict[batch.route[0]]
            ]
        )
        for first, second in zip(batch.route, batch.route[1:]):
            distances.append(
                self.distance_ij[
                    self.item_id_pod_id_dict[first], self.item_id_pod_id_dict[second]
                ]
            )
        distances.append(
            self.distance_ij[
                batch.pack_station, self.item_id_pod_id_dict[batch.route[-1]]
            ]
        )
        return np.sum(distances)

    def get_fitness_of_tour(self, batch_route, pack_station):
        """
        just like get_fitness_of_batch, but a route can be passed directly
        """
        distances = []
        distances.append(
            self.distance_ij[pack_station, self.item_id_pod_id_dict[batch_route[0]]]
        )
        for first, second in zip(batch_route, batch_route[1:]):
            distances.append(
                self.distance_ij[
                    self.item_id_pod_id_dict[first], self.item_id_pod_id_dict[second]
                ]
            )
        distances.append(
            self.distance_ij[pack_station, self.item_id_pod_id_dict[batch_route[-1]]]
        )
        return np.sum(distances)

    def get_fitness_of_solution(self):
        """
        calculates the fitness of a solution by summing up all the distances of the batches
        """
        distances = []
        for batch in self.batches.values():
            distances.append(self.get_fitness_of_batch(batch))
        return np.round(np.sum(distances), 1)

    def get_weight_per_batch(self):
        """
        calculates the total weight of all the items already assigned to the batches
        :return: dictionary with batch ID as key and its respective weight as value
        """
        weight_dict = dict()
        for batch_id, batch in self.batches.items():
            weight_dict[batch_id] = batch.weight
        return weight_dict

    def write_solution_to_xml(self, path):
        """
        function to write the solution in .xml format
        :param path: path + filename of the xml to be written
        """
        # create the file structure
        root = ET.Element("root")
        collecting = ET.SubElement(root, "Collecting")
        split = ET.SubElement(collecting, "Split")
        pack_stations = ET.SubElement(collecting, "PackStations")
        for pack_station in self.warehouseInstance.OutputStations.keys():
            ps_sum = ET.SubElement(split, "PackStation")
            ps_sum.set("ID", pack_station)
            ps = ET.SubElement(pack_stations, "PackStation")
            ps.set("ID", pack_station)
            for batch in self.batches.values():
                if batch.pack_station == pack_station:
                    # set up the batches for summary
                    bt_sum = ET.SubElement(ps_sum, "Batch")
                    bt_sum.set("ID", batch.ID)
                    # set up batches for detailed solution
                    bt = ET.SubElement(ps, "Batch")
                    bt.set("BatchNumber", batch.ID)
                    bt.set("Distance", str(self.get_fitness_of_batch(batch)))
                    items_data = ET.SubElement(bt, "ItemsData")
                    Ord = ET.SubElement(items_data, "Orders")
                    for order_id, order in self.warehouseInstance.Orders.items():
                        order_element = ET.SubElement(Ord, "Order")
                        order_element.set("ID", order_id)
                        if order_id in batch.orders:
                            # write orders to batches in summary of solution
                            order_sum = ET.SubElement(bt_sum, "Order")
                            order_sum.text = order_id
                            for item_id, item in order.Positions.items():
                                for i in range(int(item.Count)):
                                    # write items their pod and type to the orders
                                    item_element = ET.SubElement(order_element, "Item")
                                    item_element.set("ID", item_id)
                                    item_element.set(
                                        "Pod", self.item_id_pod_id_dict[item_id]
                                    )
                                    item_desc = self.warehouseInstance.ItemDescriptions[
                                        item_id
                                    ]
                                    item_element.set(
                                        "Type", item_desc.Color + "/" + item_desc.Letter
                                    )
                    edges = ET.SubElement(bt, "Edges")
                    for edge in batch.edges:
                        edge_element = ET.SubElement(edges, "Edge")
                        edge_element.set("StartNode", edge.StartNode)
                        edge_element.set("EndNode", edge.EndNode)
        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
        with open(path, "w") as f:
            f.write(xmlstr)


class GreedyHeuristic(Demo):
    """
    This class implements a greedy heuristic for the AGV assisted warehouse routing problem under the assumption of
    a dedicated storage policy. This class inherits from Demo class to access the functions and attributes defined
    there
    """

    def __init__(self):
        self.solution = None
        self.orders_of_batches = dict()
        self.batches = dict()
        self.batch_count = 0
        super(GreedyHeuristic, self).__init__()
        self.fill_item_id_pod_id_dict()
        self.fill_station_id_bot_id_dict()
        self.warehouseInstance.Orders = {
            str(key): value for key, value in enumerate(self.warehouseInstance.Orders)
        }
        self.warehouseInstance.Orders = {
            key: value
            for key, value in self.warehouseInstance.Orders.items()
            if self.get_total_weight_by_order(key) <= 18
        }  # exclude  too big orders

    def get_station_with_min_total_distance(self, order_idx):
        """
        function to retrieve the station which minimizes the total distance of the items of the
        specified order to the station
        :param order_idx: index or the order
        :return: the index of the pack station minimizing the total distance to all the items of the order
        """
        items = self.get_items_by_order(order_idx)
        order_pack_dists = dict()
        for pack_station in list(self.warehouseInstance.OutputStations.keys()):
            item_dists = []
            for item in items:
                item_dists.append(
                    self.distance_ij[pack_station, self.item_id_pod_id_dict[item]]
                )
            order_pack_dists[pack_station] = np.sum(item_dists)
        return min(order_pack_dists.keys(), key=(lambda k: order_pack_dists[k]))

    def get_new_batch_id(self):
        """
        after destroying a batch (important for ILS), new IDs have to be given to new batches. This function looks, if
        IDs are available within a sequence of numbers, or otherwise adds one to the highest ID number
        """
        if not self.batches:
            return 0
        curr_batch_ids = [int(batch.ID) for batch in self.batches.values()]
        sequence = np.arange(0, max(curr_batch_ids) + 1, 1)
        gaps = np.setdiff1d(sequence, curr_batch_ids)
        if len(gaps) == 0:
            return max(curr_batch_ids) + 1
        else:
            return min(gaps)

    def assign_orders_to_stations(self):
        """
        assigns an order to a station according to the minimum total distance. May lead to unbalanced
        assignments, e.g. that all orders are assigned to only one station.
        :return: dictionary containing the pack stations as keys and their respective orders (list) as value
        """
        orders_of_stations = defaultdict(list)
        for order_id, order in self.warehouseInstance.Orders.items():
            station_of_order = self.get_station_with_min_total_distance(order_id)
            orders_of_stations[station_of_order].append(order_id)
        return orders_of_stations

    def greedy_next_order_to_batch(
        self, batch: Batch, already_assigned, orders_of_station, forbidden=None
    ):
        """
        determines which order out of the not assigned orders will be assigned to a given batch. The order which
        minimizes the minimum distance of all items of the order to any item in the batch is chosen. The reasoning here
        is, that a order which is stored close to some item that is already in the batch will not add much distance to
        the tour itself.
        :param batch:
        :param already_assigned: list of orders which are already assigned to batches
        :param orders_of_station: list of orders that are assigned to the packing station of the batch
        :param forbidden: orders which have investigated already but would add to much weight to the batch
        :return:
        """
        if forbidden is None:
            forbidden = (
                []
            )  # to exclude orders that don´t fit in the batch anymore due to limited capacity
        other_orders_of_station = np.setdiff1d(
            orders_of_station, already_assigned + forbidden
        )
        print(
            "other unassigned orders in the same station ({}) as this batch: ".format(
                batch.pack_station
            ),
            other_orders_of_station,
        )
        sum_min_dist_to_item = dict()
        for order in other_orders_of_station:
            min_distances = []
            for item_in_batch in batch.items:
                min_distances.append(
                    min(
                        [
                            self.distance_ij[
                                self.item_id_pod_id_dict[item_in_batch],
                                self.item_id_pod_id_dict[item],
                            ]
                            for item in self.get_items_by_order(order)
                        ]
                    )
                )
            sum_min_dist_to_item[order] = np.sum(
                min_distances
            )  # average instead? otherwise we always pick small orders
        return min(sum_min_dist_to_item.keys(), key=(lambda k: sum_min_dist_to_item[k]))

    def assign_orders_to_batches_greedy(self, pack_station):
        """
        function to assign all orders of a given packing station. Goes on until all orders are assigned to a batch.
        For a given batch, orders are assigned to it until no order can be assigned to it without violating the
        capacity restriction. To assign an order to a given batch the function defined above is used
        :param pack_station: ID of the pack station
        :return:
        """
        orders_of_station = self.assign_orders_to_stations()[pack_station]
        already_assigned = list()
        while len(already_assigned) != len(orders_of_station):
            batch_id = str(self.get_new_batch_id())
            batch = Batch(
                batch_id,
                pack_station,
                self.station_id_bot_id_dict[pack_station],
                self.item_id_pod_id_dict,
            )
            print("initialized new batch with ID: ", batch.ID)
            # initialize the batch with a random order
            init_order = np.random.choice(
                np.setdiff1d(orders_of_station, already_assigned)
            )
            batch.orders.append(init_order)
            batch.items.extend(self.get_items_by_order(init_order))
            already_assigned.append(init_order)
            batch.weight += self.get_total_weight_by_order(init_order)
            forbidden_for_batch = []
            while batch.weight < self.batch_weight and not len(
                np.union1d(already_assigned, forbidden_for_batch)
            ) == len(orders_of_station):
                new_order = self.greedy_next_order_to_batch(
                    batch, already_assigned, orders_of_station, forbidden_for_batch
                )
                weight_of_order = self.get_total_weight_by_order(new_order)
                print("Chosen order: ", new_order)
                if (batch.weight + weight_of_order) <= self.batch_weight:
                    print("and it also fits in the current batch ({})".format(batch_id))
                    already_assigned.append(new_order)
                    batch.orders.append(new_order)
                    batch.items.extend(self.get_items_by_order(new_order))
                    batch.weight += weight_of_order
                else:
                    print(
                        "but it would add too much weight to the batch, go on to next..."
                    )
                    forbidden_for_batch.append(new_order)
                print(
                    "the current batch ({}) looks as follows: {}".format(
                        batch.ID, batch.orders
                    )
                )
            self.batches[batch_id] = batch

    def greedy_cobot_tour(self, batch: Batch):
        """
        for a given batch this function determines the tour in a greedy manner. Starting from the pack station of the
        batch, this function determines the node among the nodes not visited yet which is closest to the current node
        and moves to this node. Repeates this behaviour until all nodes are visited.
        :param batch:
        :return:
        """
        try:
            assert len(batch.route) == 0
        except AssertionError:
            batch.route = []
        items = np.unique(batch.items)
        distances_to_station = {
            item: self.distance_ij[batch.pack_station, self.item_id_pod_id_dict[item]]
            for item in items
        }
        min_item = min(
            distances_to_station.keys(), key=(lambda k: distances_to_station[k])
        )
        batch.route.append(min_item)
        while len(batch.route) != len(items):
            distances_to_current_item = {
                item: self.distance_ij[
                    self.item_id_pod_id_dict[batch.route[-1]],
                    self.item_id_pod_id_dict[item],
                ]
                for item in np.setdiff1d(items, batch.route)
            }
            min_item = min(
                distances_to_current_item.keys(),
                key=(lambda k: distances_to_current_item[k]),
            )
            batch.route.append(min_item)

    def apply_greedy_heuristic(self):
        """
        applies the functions defined above to determine a greedy solution to the problem
        """
        if self.batches:
            self.batches = dict()
        for pack_station in self.warehouseInstance.OutputStations.keys():
            self.assign_orders_to_batches_greedy(pack_station)
        for batch in self.batches.values():
            self.greedy_cobot_tour(batch)


class SimulatedAnnealing(GreedyHeuristic):
    def __init__(self):
        super(SimulatedAnnealing, self).__init__()

    def accept(self, curr_batch: Batch, candidate_batch: Batch, T):
        """
        determines whether a candidate solution is to be accepted or not
        :param curr_batch: batch from current solution which is to be improved
        :param candidate_batch: batch from candidate solution
        :param T: current temperature
        """
        currentFitness = self.get_fitness_of_tour(
            curr_batch.route, curr_batch.pack_station
        )
        candidateFitness = self.get_fitness_of_tour(
            candidate_batch.route, candidate_batch.pack_station
        )
        if candidateFitness < currentFitness:
            return True
        else:
            if np.random.random() <= self.acceptWithProbability(
                candidateFitness, currentFitness, T
            ):
                return True

    def acceptWithProbability(self, candidateFitness, currentFitness, T):
        # Accept the new tour for all cases where fitness of candidate => fitness current with a probability
        return math.exp(-abs(candidateFitness - currentFitness) / T)

    def two_opt(self, batch: Batch, currentSolution, batch_id, T):
        """
        implements the two opt heuristic. The algorithm iterates over every pair of edges of the tour and interchanges
        them.
        :param batch: the batch of the neighborhood solution
        :param currentSolution: the current solution to be improved
        :param batch_id: ID of the batch
        :param T: current temperature
        """
        curr_tour = batch.route[:]
        curr_tour.insert(0, batch.pack_station)
        for i in range(len(curr_tour) - 2):
            for j in range(i + 2, len(curr_tour)):
                tour = curr_tour[:]
                tour[i + 1] = tour[j]  # , tour[j] = , tour[i]
                reverse_order = [curr_tour[m] for m in reversed(range(i + 1, j))]
                tour[i + 2 : j + 1] = reverse_order
                tour.remove(batch.pack_station)
                batch.route = tour
                if self.accept(
                    curr_batch=currentSolution[batch_id], candidate_batch=batch, T=T
                ):
                    currentSolution[batch_id] = batch
                    return True
                else:
                    old_tour = curr_tour[:]
                    old_tour.remove(batch.pack_station)
                    batch.route = old_tour
                    continue
        return False

    def two_opt_randomized(self, batch: Batch, currentSolution, batch_id, T):
        """
        randomized version of two-opt in order to speed the local search up. Randomly selects to edges
        """
        curr_tour = batch.route[:]
        tour = batch.route
        length = range(len(tour))
        i, j = random.sample(length, 2)
        tour = tour[:]
        tour[i], tour[j] = tour[j], tour[i]
        reverse_order = [tour[m] for m in reversed(range(i + 1, j))]
        tour[i + 1 : j] = reverse_order
        if self.accept(
            curr_batch=currentSolution[batch_id], candidate_batch=batch, T=T
        ):
            currentSolution[batch_id] = batch
            return True
        else:
            batch.route = curr_tour
            return False

    def switch_stations(self, batch: Batch):
        """
        vary the pack station of batches
        """
        curr_ps = batch.pack_station
        new_ps = np.random.choice(
            np.setdiff1d(list(self.warehouseInstance.OutputStations.keys()), curr_ps)
        )
        return new_ps

    def simulatedAnnealing(
        self,
        alpha=0.95,
        maxIteration=1000,
        minTemperature=0.1,
        mutation_prob=0.2,
        max_it_without_change=30,
        batch_id=None,
    ):
        """
        implements a simulated annealing to optimize the tour of a given batch or all batches at once. Neighborhood
        solutions are generated using two opt, either randomized or full.
        :param alpha: cooling parameter
        :param maxIteration: maximum iterations
        :param minTemperature: temperature at which to terminate the algorithm if reached
        :param mutation_prob: probability for switching the pack station of a batch
        :param max_it_without_change: maximum number of iterations allowed without changes in the solution
        :param batch_id: if only a single batch shall be optimized, specify the ID here
        """
        # Simulated Annealing Parameters
        if not self.batches:
            # Run the greedy heuristic and get the current solution with the current fitness value
            self.apply_greedy_heuristic()
            self.greedyFitness = self.get_fitness_of_solution()
            print("Fitness of greedy solution: ", self.greedyFitness)
            currentSolution = copy.deepcopy(self.batches)
        else:
            currentSolution = copy.deepcopy(self.batches)

        # if we have only changed one batch, then the optimization of the route is only performed for that
        # particular batch
        if batch_id:
            currentSolution = {batch_id: currentSolution[batch_id]}
        for batch_id in currentSolution.keys():
            iteration = 0
            it_without_change = 0
            T = max(self.distance_ij.values()) - min(
                list(self.distance_ij.values())
            )  # determine initial temperature
            while (
                T >= minTemperature
                and iteration < maxIteration
                and it_without_change < max_it_without_change
            ):
                batch = copy.deepcopy(currentSolution[batch_id])
                if np.random.random() <= mutation_prob:
                    batch.pack_station = self.switch_stations(batch)
                    self.greedy_cobot_tour(
                        batch
                    )  # recalculate the greedy tour with new pack station
                made_change = self.two_opt_randomized(
                    batch, currentSolution, batch_id, T
                )

                it_without_change += 1 if not made_change else 0
                T *= alpha
                iteration += 1
            self.batches[batch_id] = currentSolution[batch_id]


class IteratedLocalSearch(SimulatedAnnealing):
    def __init__(self):
        super(IteratedLocalSearch, self).__init__()
        self.simulatedAnnealing()

    def update_batches_from_new_order(self, new_order, batch_id):
        """
        does all the necessary updating when a order is added to a batch. This includes optimizing the tour through
        greedy and simulated annealing heuristics
        :param new_order:
        :param batch_id:
        :return:
        """
        update_batch = self.batches[batch_id]
        update_batch.orders.append(new_order)
        update_batch.items.extend(self.get_items_by_order(new_order))
        update_batch.weight += self.get_total_weight_by_order(new_order)
        update_batch.route = []
        self.greedy_cobot_tour(update_batch)
        # print("solution before sim ann: ", self.get_fitness_of_solution())
        self.simulatedAnnealing(batch_id=batch_id, mutation_prob=0.2)

    def perturbation(self, history):
        """
        performs the perturbation. The perturbation operation implements a destroy and repair strategy. A random
        batch is destroyed and its orders are distributed among the other batches. If an order does not fit in any
        of the other batches, a new batch is created.
        :param history:
        :return:
        """
        print("Performing perturbation")
        weight_dict = self.get_weight_per_batch()
        probabilities = []
        keys = []
        # calculate probabilities for destroying batches
        for key, weight in weight_dict.items():
            if key == history:
                continue
            else:
                probabilities.append(1 - weight / self.batch_weight)
                keys.append(key)
        probabilities = [float(i) / sum(probabilities) for i in probabilities]
        # determine batches to be destroyed
        # destroy_batch = np.random.choice(keys, p=probabilities)
        destroy_batch = np.random.choice(keys)
        weight_dict.pop(destroy_batch)
        orders = self.batches[destroy_batch].orders
        self.batches.pop(destroy_batch)
        # sort orders with respect to their weight in descending order (try to assign the biggest orders first)
        weight_of_orders = {}
        for order in orders:
            weight_of_orders[order] = self.get_total_weight_by_order(order)
        weight_of_orders = {
            k: v
            for k, v in sorted(
                weight_of_orders.items(), key=lambda item: item[1], reverse=True
            )
        }
        for order, weight_of_order in weight_of_orders.items():
            candidate_batches = {}
            for key, weight in weight_dict.items():
                if weight_of_order + weight <= self.batch_weight:
                    candidate_batches[key] = weight_of_order + weight
            if not candidate_batches:
                pack_station = self.get_station_with_min_total_distance(order)
                batch_id = str(self.get_new_batch_id())
                self.batches[batch_id] = Batch(
                    batch_id,
                    pack_station,
                    self.station_id_bot_id_dict[pack_station],
                    self.item_id_pod_id_dict,
                )
                self.update_batches_from_new_order(order, batch_id)
                history = batch_id
            else:
                # choose the batch with maximum workload after assignment
                new_batch_of_order = np.random.choice(
                    list(candidate_batches.keys())
                )  # max(candidate_batches.keys(), key=(lambda k: candidate_batches[k]))
                # new_batch_of_order = max(candidate_batches.keys(), key=(lambda k: candidate_batches[k]))
                self.update_batches_from_new_order(
                    new_order=order, batch_id=new_batch_of_order
                )
            # update weight_dict
            weight_dict = self.get_weight_per_batch()
        print("fitness after perturbation: ", self.get_fitness_of_solution())
        return history

    def replace_order(self, batch: Batch, add_order, remove_order):
        """
        does all the necessary updating when a order is replaced with another. This includes optimizing the tour through
        greedy and simulated annealing heuristics
        """
        batch.orders.remove(remove_order)
        for item in self.get_items_by_order(remove_order):
            del batch.items[batch.items.index(item)]

        batch.orders.append(add_order)
        batch.items.extend(self.get_items_by_order(add_order))

        batch.weight += self.get_total_weight_by_order(
            add_order
        ) - self.get_total_weight_by_order(remove_order)

        self.greedy_cobot_tour(
            batch
        )  # optimize tour to accurately validate the change in fitness
        self.simulatedAnnealing(batch_id=batch.ID, mutation_prob=0.2)

    def determine_switchable_orders_randomized(self, batch_i, batch_j):
        """
        given two batches (batch ids), this function determines one order of each batch which can be exchanged between
        the batches (wrt to capacity restrictions).
        :param batch_i:
        :param batch_j:
        :return:
        """
        batch_i_orders = batch_i.orders
        batch_j_order = batch_j.orders
        np.random.shuffle(batch_i_orders)
        np.random.shuffle(batch_j_order)
        for order_i in batch_i_orders:
            for order_j in batch_j_order:
                weight_diff = self.get_total_weight_by_order(
                    order_i
                ) - self.get_total_weight_by_order(order_j)
                if weight_diff < 0:
                    if abs(weight_diff) <= self.batch_weight - batch_i.weight:
                        return order_i, order_j
                elif weight_diff > 0:
                    if weight_diff <= self.batch_weight - batch_j.weight:
                        return order_i, order_j
        return None, None

    def randomized_local_search(self, max_changes=20, history=None):
        """
        perform simple randomized swap of orders to batches. This is done for a given number of iterations on
        randomly drawn orders.
        Neighborhood solution is accepted if it results in a better fitness value
        :return:
        """
        iters = 0
        curr_fit = self.get_fitness_of_solution()
        curr_sol = copy.deepcopy(self.batches)
        while iters < max_changes:
            print("do local search")
            batch_i, batch_j = np.random.choice(
                [
                    batch
                    for key, batch in self.batches.items()
                    if key != history and len(batch.orders) > 1
                ],
                2,
                replace=False,
            )
            order_i, order_j = self.determine_switchable_orders_randomized(
                batch_i, batch_j
            )
            if order_i:
                self.replace_order(batch_i, add_order=order_j, remove_order=order_i)
                self.replace_order(batch_j, add_order=order_i, remove_order=order_j)
                if self.get_fitness_of_solution() < curr_fit:
                    curr_sol = copy.deepcopy(self.batches)
                    curr_fit = self.get_fitness_of_solution()
                else:
                    self.batches = copy.deepcopy(curr_sol)
            iters += 1

    def determine_switchable_orders(self, batch_i, batch_j, memory, curr_fit, curr_sol):
        """
        given two batches (batch ids), this function determines one order of each batch which can be exchanged between
        the batches (wrt to capacity restrictions).
        :param batch_i:
        :param batch_j:
        :return:
        """
        batch_i_orders = batch_i.orders
        batch_j_order = batch_j.orders
        for order_i in batch_i_orders:
            for order_j in batch_j_order:
                if {order_i, order_j} in memory:
                    continue
                memory.append({order_i, order_j})
                weight_diff = self.get_total_weight_by_order(
                    order_i
                ) - self.get_total_weight_by_order(order_j)
                if weight_diff <= 0:
                    if abs(weight_diff) <= self.batch_weight - batch_i.weight:
                        self.replace_order(
                            batch_i, add_order=order_j, remove_order=order_i
                        )
                        self.replace_order(
                            batch_j, add_order=order_i, remove_order=order_j
                        )

                        if self.get_fitness_of_solution() < curr_fit:
                            curr_sol = copy.deepcopy(self.batches)
                            curr_fit = self.get_fitness_of_solution()

                        else:
                            self.batches = copy.deepcopy(curr_sol)

                elif weight_diff > 0:
                    if weight_diff <= self.batch_weight - batch_j.weight:
                        self.replace_order(
                            batch_i, add_order=order_j, remove_order=order_i
                        )
                        self.replace_order(
                            batch_j, add_order=order_i, remove_order=order_j
                        )
                        if self.get_fitness_of_solution() < curr_fit:
                            curr_sol = copy.deepcopy(self.batches)
                            curr_fit = self.get_fitness_of_solution()
                        else:
                            self.batches = copy.deepcopy(curr_sol)

                return self.determine_switchable_orders(
                    self.batches[batch_i.ID],
                    self.batches[batch_j.ID],
                    memory,
                    curr_fit,
                    curr_sol,
                )

    def local_search(self, processed=None):
        """
        perform simple swap of orders to batches. This is done for any two orders.
        Neighborhood solution is accepted if it results in a better fitness value
        :return:
        """
        processed = [] if not processed else processed
        curr_fit = self.get_fitness_of_solution()
        curr_sol = copy.deepcopy(self.batches)
        memory = []
        for batch_i in self.batches.values():
            for batch_j in self.batches.values():
                if batch_i.ID == batch_j.ID or {batch_i.ID, batch_j.ID} in processed:
                    continue
                print("do local search")
                self.determine_switchable_orders(
                    batch_i, batch_j, memory, curr_fit, curr_sol
                )
                processed.append({batch_i.ID, batch_j.ID})
                return self.local_search(processed)

    def perform_ils(self, num_iters, t_max, threshold=50):
        """
        implements the Iterated Local Search
        """
        iter = 0
        iter_last = 0
        starttime = time.time()
        t = time.time() - starttime
        curr_fit = self.get_fitness_of_solution()
        best_fit = curr_fit
        curr_sol = copy.deepcopy(self.batches)
        best_sol = copy.deepcopy(curr_sol)
        history = None
        T = self.get_fitness_of_solution() * 0.04
        while iter < num_iters and t < t_max:
            iter += 1
            history = self.perturbation(history)
            self.local_search()
            neighbor_fit = self.get_fitness_of_solution()
            if neighbor_fit < curr_fit or self.acceptWithProbability(
                neighbor_fit, curr_fit, T
            ):
                curr_fit = neighbor_fit  # accept new solution
                curr_sol = copy.deepcopy(self.batches)
                iter_last = 0
                if curr_fit < best_fit:
                    best_sol = copy.deepcopy(self.batches)
                    best_fit = self.get_fitness_of_solution()
            elif (
                iter - iter_last
            ) > threshold:  # restart if no new solution has been found in a while
                print("RESTART")
                self.apply_greedy_heuristic()
                curr_sol = self.get_fitness_of_solution()
                curr_fit = self.get_fitness_of_solution()
                iter, iter_last = 0, 0
            else:
                self.batches = copy.deepcopy(
                    curr_sol
                )  # don´t accept new solution and stick with the current one
                iter_last += 1
            iter += 1
            T *= 0.875
            t = time.time() - starttime
            print("Fitness after iteration of ILS: ", self.get_fitness_of_solution())
        self.batches = best_sol


if __name__ == "__main__":
    SKUS = ["360"]  # options: 24 and 360
    SUBSCRIPTS = [""]
    NUM_ORDERSS = [20]
    MEANS = ["5"]
    instance_sols = {}
    for SKU in SKUS:
        for SUBSCRIPT in SUBSCRIPTS:
            for NUM_ORDERS in NUM_ORDERSS:
                for MEAN in MEANS:
                    # CAUTION: SCRIPT WONT RUN IF ALL SOLUTIONS ARE WRITTEN AND THIS IS NOT PUT IN COMMENTS
                    # if os.path.isfile('solutions/final/orders_{}_mean_{}_sku_{}{}_{}.xml'.format(str(NUM_ORDERS), MEAN, SKU, SUBSCRIPT, "dedicated")):
                    #    continue  # skip iteration if the instance has been solved already

                    try:
                        layoutFile = r"data/layout/1-1-1-2-1.xlayo"
                        podInfoFile = "data/sku{}/pods_infos.txt".format(SKU)
                        instances = {}
                        instances[24, 2] = r"data/sku{}/layout_sku_{}_2.xml".format(
                            SKU, SKU
                        )

                        storagePolicies = {}
                        storagePolicies[
                            "dedicated"
                        ] = "data/sku{}/pods_items_dedicated_1.txt".format(SKU)
                        # storagePolicies['mixed'] = 'data/sku{}/pods_items_mixed_shevels_1-5.txt'.format(SKU)

                        orders = {}
                        orders[
                            "{}_5".format(str(NUM_ORDERS))
                        ] = r"data/sku{}/orders_{}_mean_{}_sku_{}{}.xml".format(
                            SKU, str(NUM_ORDERS), MEAN, SKU, SUBSCRIPT
                        )
                        sols_and_runtimes = {}
                        runtimes = [5, 10, 20, 40, 80, 140]
                        runtimes = [40]  # for debugging
                        for runtime in runtimes:
                            np.random.seed(52302381)
                            if runtime == 0:
                                ils = GreedyHeuristic()
                                ils.apply_greedy_heuristic()
                            else:
                                ils = IteratedLocalSearch()
                                ils.perform_ils(1500, runtime)
                            STORAGE_STRATEGY = (
                                "dedicated" if ils.is_storage_dedicated else "mixed"
                            )
                            ils.write_solution_to_xml(
                                "solutions/orders_{}_mean_{}_sku_{}{}_{}.xml".format(
                                    str(NUM_ORDERS),
                                    MEAN,
                                    SKU,
                                    SUBSCRIPT,
                                    STORAGE_STRATEGY,
                                )
                            )
                            sols_and_runtimes[runtime] = (
                                ils.get_fitness_of_solution(),
                                {
                                    batch.ID: batch.determine_edges(
                                        ils.item_id_pod_id_dict
                                    )
                                    for batch in ils.batches.values()
                                },
                            )
                        print(sols_and_runtimes)
                    except Exception as e:
                        print(e)
                        continue
                    instance_sols[(SKU, SUBSCRIPT, NUM_ORDERS)] = sols_and_runtimes

    with open("../analyse_solution/solutions/dedicated.pickle", "wb") as handle:
        pickle.dump(instance_sols, handle, protocol=pickle.HIGHEST_PROTOCOL)
