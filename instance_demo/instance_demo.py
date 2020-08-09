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
from collections import defaultdict, Iterable
from utils import Batch
import copy
import random

layoutFile = r'data/layout/1-1-1-2-1.xlayo'
podInfoFile = 'data/sku24/pods_infos.txt'

instances = {}
instances[24, 2] = r'data/sku24/layout_sku_24_2.xml'

storagePolicies = {}
storagePolicies['dedicated'] = 'data/sku24/pods_items_dedicated_1.txt'
# storagePolicies['mixed'] = 'data/sku24/pods_items_mixed_shevels_1-5.txt'

orders = {}
orders['10_5'] = r'data/sku24/orders_10_mean_5_sku_24.xml'
# orders['20_5'] = r'data/sku24/orders_20_mean_5_sku_24.xml'

class WarehouseDateProcessing():
    def __init__(self, warehouseInstance, batch_size=None):
        self.Warehouse = warehouseInstance
        self._InitSets(warehouseInstance, batch_size)

    def preprocessingFilterPods(self, warehouseInstance):
        resize_pods = {}
        print("preprocessingFilterPods")
        item_id_list = []
        for order in warehouseInstance.Orders:
            for pos in order.Positions.values():
                item = warehouseInstance.ItemDescriptions[pos.ItemDescID].Color.lower() + '/' + \
                       warehouseInstance.ItemDescriptions[pos.ItemDescID].Letter
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

        file_path = r'data/distances/' + os.path.splitext(os.path.basename(self.Warehouse.InstanceFile))[0] + '.json'

        if not path.exists(file_path):
            # Create d_ij
            # d_ij = tupledict()
            d_ij = {}

            # Loop over all nodes
            for key_i, node_i in self.V.items():
                for key_j, node_j in self.V.items():
                    source = 'w' + node_i.GetPickWaypoint().ID
                    target = 'w' + node_j.GetPickWaypoint().ID

                    # Calc distance with weighted shortest path
                    d_ij[(key_i, key_j)] = nx.shortest_path_length(self.Graph, source=source, target=target,
                                                                   weight='weight')

            # Parse and save
            d_ij_dict = {}
            for key, value in d_ij.items():
                i, j = key
                if i not in d_ij_dict:
                    d_ij_dict[i] = {}
                d_ij_dict[i][j] = value

            with open(file_path, 'w') as fp:
                json.dump(d_ij_dict, fp)

        else:
            # Load and deparse
            with open(file_path, 'r') as fp:
                d_ij_dict = json.load(fp)
            print('d_ij file %s loaded' % file_path)
            d_ij = {}
            for i, values in d_ij_dict.items():
                for j, dist in values.items():
                    d_ij[i, j] = dist

        return d_ij


class Demo():
    def __init__(self, splitOrders=False):

        self.batch_weight = 18
        self.item_id_pod_id_dict = {}
        # [0]
        self.warehouseInstance = self.prepareData()
        self.distance_ij = self.initData()
        # [2]
        if storagePolicies.get('dedicated'):
            self.is_storage_dedicated = True
        else:
            self.is_storage_dedicated = False

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
                    warehouseInstance = instance.Warehouse(layoutFile, instanceFile, podInfoFile, storagePolicyFile,
                                                           orderFile)
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
        trie = self.build_trie_for_items()
        for key, pod in self.warehouseInstance.Pods.items():
            for item in pod.Items:
                ids = item.ID
                color, product = ids.split("/")
                item_id = trie[product][color]
                self.item_id_pod_id_dict[item_id] = key

    # define get functions for better readability and accessibility of data
    def get_items_by_order(self, order_idx):
        item_ids = []
        item_idx_keys = list(self.warehouseInstance.Orders[order_idx].Positions.keys())
        for key in item_idx_keys:
            item_ids.append(self.warehouseInstance.Orders[order_idx].Positions[key].ItemDescID)
        return item_ids

    def get_weight_by_item(self, item_idx):
        return self.warehouseInstance.ItemDescriptions[item_idx].Weight

    def get_total_weight_by_order(self, order_idx):
        weights = []
        items_of_order = self.get_items_by_order(order_idx)
        for item in items_of_order:
            quant_of_item = int(self.warehouseInstance.Orders[order_idx].Positions[item].Count)
            weights.append(self.get_weight_by_item(item) * quant_of_item)
        return np.sum(weights)

    def get_distance_between_items(self, item1, item2):
        shelf1 = self.item_id_pod_id_dict[item1]
        shelf2 = self.item_id_pod_id_dict[item2]
        return self.distance_ij[shelf1, shelf2]

    def get_fitness_of_batch(self, batch: Batch):
        distances = []
        distances.append(self.distance_ij[batch.pack_station, self.item_id_pod_id_dict[batch.route[0]]])
        for first, second in zip(batch.route, batch.route[1:]):
            distances.append(self.distance_ij[self.item_id_pod_id_dict[first], self.item_id_pod_id_dict[second]])
        distances.append(self.distance_ij[batch.pack_station, self.item_id_pod_id_dict[batch.route[-1]]])
        return np.sum(distances)

    def get_fitness_of_tour(self, batch_route, pack_station):
        distances = []
        distances.append(self.distance_ij[pack_station, self.item_id_pod_id_dict[batch_route[0]]])
        for first, second in zip(batch_route, batch_route[1:]):
            distances.append(self.distance_ij[self.item_id_pod_id_dict[first], self.item_id_pod_id_dict[second]])
        distances.append(self.distance_ij[pack_station, self.item_id_pod_id_dict[batch_route[-1]]])
        return np.sum(distances)

    def get_fitness_of_solution(self):
        distances = []
        for batch in self.batches.values():
            distances.append(self.get_fitness_of_batch(batch))
        return np.sum(distances)


class GreedyHeuristic(Demo):
    def __init__(self):
        self.solution = None
        self.orders_of_batches = dict()
        self.batches = dict()
        super(GreedyHeuristic, self).__init__()
        self.fill_item_id_pod_id_dict()
        self.warehouseInstance.Orders = {str(key): value for key, value in enumerate(self.warehouseInstance.Orders)}

    def get_station_with_min_total_distance(self, order_idx, retrieve_station_only=False):
        """
        function to retrieve the station which minimizes the total distance of the items of the
        specified order
        :param order_idx: index or the order
        :param retrieve_station_only: indicates whether to return the station with min distance only or
        a together with the distance which can be used for further decision making.
        :return:
        """
        items = self.get_items_by_order(order_idx)
        order_pack_dists = dict()
        for pack_station in list(self.warehouseInstance.OutputStations.keys()):
            item_dists = []
            for item in items:
                item_dists.append(self.distance_ij[pack_station, self.item_id_pod_id_dict[item]])
            order_pack_dists[pack_station] = np.sum(item_dists)
        if retrieve_station_only:
            return min(order_pack_dists.keys(), key=(lambda k: order_pack_dists[k]))
        else:
            # eventually we want the assignment of orders to stations to be balanced,
            # then the distance can be used to make the assignment in a greedy manner until
            # the max number of orders that may be assigned to a station is reached
            station = min(order_pack_dists.keys(), key=(lambda k: order_pack_dists[k]))
            dictio = {station: order_pack_dists[station]}
            return dictio

    def assign_orders_to_stations(self):
        """
        assigns an order to a given station according to the minimum total distance. May lead to unbalanced
        assignments, e.g. that all orders are assigned to only one station.
        :return:
        """
        orders_of_stations = defaultdict(list)
        for order_id, order in self.warehouseInstance.Orders.items():  # todo: give orders an ID instead of enumeration
            station_of_order = self.get_station_with_min_total_distance(order_id, True)
            orders_of_stations[station_of_order].append(order_id)
        return orders_of_stations

    def get_all_items_of_batch(self, batch):
        items = []
        for order in batch:
            items.extend(self.get_items_by_order(order))
        return items

    def greedy_next_order_to_batch(self, batch: Batch, already_assigned, orders_of_station, forbidden=None):
        if forbidden is None:
            forbidden = []  # to exclude orders that don´t fit in the batch anymore due to limited capacity # todo Order auch als eigene Klasse (child von Order mit neuen attributen)
        other_orders_of_station = np.setdiff1d(orders_of_station, already_assigned+forbidden)
        print("other unassigned orders in the same station ({}) as this batch: ".format(batch.pack_station), other_orders_of_station)
        sum_min_dist_to_item = dict()
        for order in other_orders_of_station:
            min_distances = []
            for item_in_batch in batch.items:
                min_distances.append(min([self.distance_ij[self.item_id_pod_id_dict[item_in_batch],
                                                           self.item_id_pod_id_dict[item]]
                                          for item in self.get_items_by_order(order)]))
            sum_min_dist_to_item[order] = np.sum(min_distances)  # average instead? otherwise we always pick small orders
        return min(sum_min_dist_to_item.keys(), key=(lambda k: sum_min_dist_to_item[k]))

    def assign_orders_to_batches_greedy(self, pack_station_idx):
        orders_of_station = self.assign_orders_to_stations()[pack_station_idx]
        already_assigned = list()
        batch_count = 0
        while len(already_assigned) != len(orders_of_station):
            batch_id = pack_station_idx + "_" + str(batch_count)
            batch = Batch(batch_id, pack_station_idx)
            print("initialized new batch with ID: ", batch.ID)
            # initialize the batch with a random order
            init_order = np.random.choice(np.setdiff1d(orders_of_station, already_assigned))
            batch.orders.append(init_order)
            batch.items.extend(self.get_items_by_order(init_order))
            already_assigned.append(init_order)
            batch.weight += self.get_total_weight_by_order(init_order)
            forbidden_for_batch = []
            while batch.weight < self.batch_weight and not len(
                    np.union1d(already_assigned, forbidden_for_batch)) == len(orders_of_station):
                new_order = self.greedy_next_order_to_batch(batch, already_assigned, orders_of_station, forbidden_for_batch)
                weight_of_order = self.get_total_weight_by_order(new_order)
                print("Chosen order: ", new_order)
                if (batch.weight + weight_of_order) <= self.batch_weight:
                    print("and it also fits in the current batch ({})".format(batch_id))
                    already_assigned.append(new_order)
                    batch.orders.append(new_order)
                    batch.items.extend(self.get_items_by_order(new_order))
                    batch.weight += weight_of_order
                else:
                    print("but it would add too much weight to the batch, go on to next...")
                    forbidden_for_batch.append(new_order)
                print("the current batch ({}) looks as follows: {}".format(batch.ID, batch.orders))
            self.batches[batch_id] = batch
            batch_count += 1

    def greedy_cobot_tour(self, batch: Batch):
        items = np.unique(batch.items)
        distances_to_station = {item: self.distance_ij[batch.pack_station, self.item_id_pod_id_dict[item]]
                                for item in items}
        min_item = min(distances_to_station.keys(), key=(lambda k: distances_to_station[k]))
        batch.route.append(min_item)  # TODO in items tauchen doppelte einträge auf. Wie geht man damit um?
        while len(batch.route) != len(items):
            distances_to_current_item = {
                item: self.distance_ij[self.item_id_pod_id_dict[batch.route[-1]], self.item_id_pod_id_dict[item]]
                for item in np.setdiff1d(items, batch.route)
            }
            min_item = min(distances_to_current_item.keys(), key=(lambda k: distances_to_current_item[k]))
            batch.route.append(min_item)

    def apply_greedy_heuristic(self):
        for pack_station in self.warehouseInstance.OutputStations.keys():
            self.assign_orders_to_batches_greedy(pack_station)
        for batch in self.batches.values():
            self.greedy_cobot_tour(batch)
        merge_batches = True
        while merge_batches:
            merge_batches = self.merge_batches_of_stations()
            self.update_solution()

    def get_weight_per_batch(self):
        weight_dict = dict()
        for batch_id, batch in self.batches.items():
            weight_dict[batch_id] = batch.weight
        return weight_dict

    def merge_batches_of_stations(self):
        weight_dict = self.get_weight_per_batch()
        # cartesian product of all batches and their corresponding weights
        weight_of_merged_batches = np.zeros(shape=(len(weight_dict.keys()), len(weight_dict.keys())))
        for i, weight_i in enumerate(weight_dict.values()):
            for j, weight_j in enumerate(weight_dict.values()):
                weight_of_merged_batches[i, j] = weight_i + weight_j
        # exclude similar elements, i.e. (i,j) = (j,i) --> drop (i,j) by using upper triangular of matrix
        merged_batches_df = pd.DataFrame(np.triu(weight_of_merged_batches, k=1),
                                         columns=list(weight_dict.keys()), index=list(weight_dict.keys()))\
            .reset_index().melt(id_vars="index")
        merged_batches_df.columns = ["index_x", "index_y", "total_weight"]
        merged_batches_df = merged_batches_df[(merged_batches_df.total_weight > 0) &
                                              (merged_batches_df.total_weight < self.batch_weight)]
        if merged_batches_df.empty:
            return False
        # get the merge that results in the highest workload of the cobot (in terms of capacity)
        merged_batches_df = merged_batches_df.loc[merged_batches_df.groupby("index_x")["total_weight"].idxmax()]
        merged_batches_df = merged_batches_df.loc[merged_batches_df.groupby("index_y")["total_weight"].idxmax()]
        # update the batches
        self.update_batches(merged_batches_df)
        return True

    def update_batches(self, merged_batches_df):
        for index, row in merged_batches_df.iterrows():
            batch1 = self.batches[row.index_x]
            batch2 = self.batches[row.index_y]
            if len(batch1.orders) >= len(batch2.orders):
                update_batch = batch1
                delete_batch = batch2
            else:
                update_batch = batch2
                delete_batch = batch1
            update_batch.orders.extend(delete_batch.orders)
            update_batch.items.extend(delete_batch.items)
            update_batch.weight += delete_batch.weight
            self.greedy_cobot_tour(update_batch)
            self.batches[delete_batch.ID].pop()
            del delete_batch

    def update_solution(self):
        """
        function to update the solution after changing the assignment of orders to batches
        --> lösung des tsp muss ausgelagert werden. am ende dann eine funktion die alles aufruft aber selber keine
        heurisitk mehr implementiert
        :return:
        """
        pass

class SimulatedAnnealing(Demo):

    def __init__(self):
        super(SimulatedAnnealing, self).__init__()
        self.fill_item_id_pod_id_dict()
        # Simulated Annealing Parameters
        self.alpha = 0.99
        self.T = 0.6
        self.iteration = 1
        self.maxIteration = 1000
        self.minTemperature = 1e-10

        # Run the greedy heuristic and get the current solution with the current fitness value
        greedy = GreedyHeuristic()
        greedy.apply_greedy_heuristic()

        self.greedyFitness = greedy.get_fitness_of_solution()
        print("Fitness of greedy solution: ", self.greedyFitness)
        self.currentSolution, self.currentFitness = greedy.batches, self.greedyFitness

        # Accept the new tour for all cases where fitness of candidate < fitness current

    def accept(self, current_route, candidate_route, pack_station):

        currentFitness = self.get_fitness_of_tour(current_route, pack_station)
        candidateFitness = self.get_fitness_of_tour(candidate_route, pack_station)

        # print("currentfit:",currentFitness)
        # print("candidatefit:",candidateFitness)

        if candidateFitness < currentFitness:
            return True
        else:
            if np.random.random() < self.acceptWithProbability(candidateFitness, currentFitness):
                return True

        # Accept the new tour for all cases where fitness of candidate => fitness current with a probability

    def acceptWithProbability(self, candidateFitness, currentFitness):

        return math.exp(-abs(candidateFitness - currentFitness) / self.T)

    # Swaps the position of two points in the solution of each batch
    def swap(self, seq):
        # length is the length of the tour of one batch
        length = range(len(seq))
        i1, i2 = random.sample(length, 2)
        seq[i1], seq[i2] = seq[i2], seq[i1]
        return seq

    def simulatedAnnealing(self):

        currenSolutionCopy = copy.deepcopy(self.currentSolution)

        print("Starting simulated annealing")

        while self.T >= self.minTemperature and self.iteration < self.maxIteration:

            for batch_id, batch in self.currentSolution.items():

                if len(batch.route) > 1:

                    pointCopy = batch.route[:]

                    candidate = self.swap(pointCopy)
                    # print("Current",point)
                    # print("Candidate",candidate)

                    if self.accept(batch.route, candidate, batch.pack_station):
                        currenSolutionCopy[batch_id].route = candidate

            self.currentSolution = copy.deepcopy(currenSolutionCopy)
            self.T *= self.alpha
            self.iteration += 1

        self.batches = self.currentSolution
        print("Fitness of Simulated Annealing: ", self.get_fitness_of_solution())

if __name__ == "__main__":
    np.random.seed(123)
    greedy = GreedyHeuristic()
    starttime = time.time()
    # solution = greedy.apply_greedy_heuristic()
    greedy.apply_greedy_heuristic()
    # print("runtime: ", time.time() - starttime)
    # print("fitness: ", greedy.get_fitness_of_solution(solution))
    # print(solution)
    # print(greedy.orders_of_batches)
    # print(greedy.get_weight_per_batch())
    print(greedy.get_fitness_of_solution())
    greedy.merge_batches_of_stations()

    simAnn = SimulatedAnnealing()
    simAnn.simulatedAnnealing()
