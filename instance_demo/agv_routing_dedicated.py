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

layoutFile = r'data/layout/1-1-1-2-1.xlayo'
podInfoFile = 'data/sku{}/pods_infos.txt'.format(SKU)

instances = {}
instances[24, 2] = r'data/sku{}/layout_sku_{}_2.xml'.format(SKU, SKU)

storagePolicies = {}
storagePolicies['dedicated'] = 'data/sku{}/pods_items_dedicated_1.txt'.format(SKU)
#storagePolicies['mixed'] = 'data/sku{}/pods_items_mixed_shevels_1-5.txt'.format(SKU)

orders = {}
orders['{}_5'.format(str(NUM_ORDERS))] = r'data/sku{}/orders_{}_mean_5_sku_{}{}.xml'.format(SKU, str(NUM_ORDERS), SKU, SUBSCRIPT)
#orders['10_5'] = r'data/sku{}/orders_10_mean_1x6_sku_{}.xml'.format(SKU, SKU)

# ----------------------------------------------------------------------------------------------------------------------


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
        self.station_id_bot_id_dict = dict()
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

    def fill_station_id_bot_id_dict(self):
        for key, bot in self.warehouseInstance.Bots.items():
            self.station_id_bot_id_dict[bot.OutputStation] = key

    # define get functions for better readability and accessibility of data
    def get_items_by_order(self, order_idx):
        item_ids = []
        for item_id, item in self.warehouseInstance.Orders[order_idx].Positions.items():
            item_ids.append(item.ItemDescID)
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

    def get_weight_per_batch(self):
        weight_dict = dict()
        for batch_id, batch in self.batches.items():
            weight_dict[batch_id] = batch.weight
        return weight_dict


class GreedyHeuristic(Demo):
    def __init__(self):
        self.solution = None
        self.orders_of_batches = dict()
        self.batches = dict()
        self.batch_count = 0
        super(GreedyHeuristic, self).__init__()
        self.fill_item_id_pod_id_dict()
        self.fill_station_id_bot_id_dict()
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
        for order_id, order in self.warehouseInstance.Orders.items():
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
            forbidden = []  # to exclude orders that don´t fit in the batch anymore due to limited capacity
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

    def get_new_batch_id(self):
        """after destroying a batch, new IDs have to be given to new batches. This function looks, if IDs are
        available within a sequence of numbers, or otherwise adds one to the highest ID number"""
        if not self.batches:
            return 0
        curr_batch_ids = [int(batch.ID.split("_")[1]) for batch in self.batches.values()]
        sequence = np.arange(min(curr_batch_ids), max(curr_batch_ids)+1, 1)
        gaps = np.setdiff1d(sequence, curr_batch_ids)
        if len(gaps) == 0:
            return max(curr_batch_ids) + 1
        else:
            return min(gaps)

    def assign_orders_to_batches_greedy(self, pack_station):
        orders_of_station = self.assign_orders_to_stations()[pack_station]
        already_assigned = list()
        while len(already_assigned) != len(orders_of_station):
            batch_id = pack_station + "_" + str(self.get_new_batch_id())
            batch = Batch(batch_id, pack_station, self.station_id_bot_id_dict[pack_station])
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

    def greedy_cobot_tour(self, batch: Batch):
        try:
            assert len(batch.route) == 0
        except AssertionError:
            batch.route = []
        items = np.unique(batch.items)
        distances_to_station = {item: self.distance_ij[batch.pack_station, self.item_id_pod_id_dict[item]]
                                for item in items}
        min_item = min(distances_to_station.keys(), key=(lambda k: distances_to_station[k]))
        batch.route.append(min_item)
        while len(batch.route) != len(items):
            distances_to_current_item = {
                item: self.distance_ij[self.item_id_pod_id_dict[batch.route[-1]], self.item_id_pod_id_dict[item]]
                for item in np.setdiff1d(items, batch.route)
            }
            min_item = min(distances_to_current_item.keys(), key=(lambda k: distances_to_current_item[k]))
            batch.route.append(min_item)

    def apply_greedy_heuristic(self):
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
        currentFitness = self.get_fitness_of_tour(curr_batch.route, curr_batch.pack_station)
        candidateFitness = self.get_fitness_of_tour(candidate_batch.route, candidate_batch.pack_station)
        # print("currentfit:",currentFitness)
        # print("candidatefit:",candidateFitness)
        if candidateFitness < currentFitness:
            return True
        else:
            if np.random.random() <= self.acceptWithProbability(candidateFitness, currentFitness, T):
                return True

    def acceptWithProbability(self, candidateFitness, currentFitness, T):
        # Accept the new tour for all cases where fitness of candidate => fitness current with a probability
        return math.exp(-abs(candidateFitness - currentFitness) / T)

    def two_opt(self, batch: Batch, currentSolution, batch_id, T):
        curr_tour = batch.route[:]
        curr_tour.insert(0, batch.pack_station)
        for i in range(len(curr_tour) - 2):
            for j in range(i + 2, len(curr_tour)):
                tour = curr_tour[:]
                tour[i + 1] = tour[j]  # , tour[j] = , tour[i]
                reverse_order = [curr_tour[m] for m in reversed(range(i + 1, j))]
                tour[i + 2: j + 1] = reverse_order
                tour.remove(batch.pack_station)
                batch.route = tour
                if self.accept(curr_batch=currentSolution[batch_id], candidate_batch=batch, T=T):
                    currentSolution[batch_id] = batch
                    return True
                else:
                    old_tour = curr_tour[:]
                    old_tour.remove(batch.pack_station)
                    batch.route = old_tour
                    continue
        return False

    def two_opt_randomized(self, batch: Batch, currentSolution, batch_id, T):
        curr_tour = batch.route[:]
        tour = batch.route
        length = range(len(tour))
        i, j = random.sample(length, 2)
        tour = tour[:]
        tour[i], tour[j] = tour[j], tour[i]
        reverse_order = [tour[m] for m in reversed(range(i + 1, j))]
        tour[i + 1: j] = reverse_order
        if self.accept(curr_batch=currentSolution[batch_id], candidate_batch=batch, T=T):
            currentSolution[batch_id] = batch
            return True
        else:
            batch.route = curr_tour
            return False

    def switch_stations(self, batch: Batch):
        """
        vary the pack station of batches
        :return:
        """
        curr_ps = batch.pack_station
        new_ps = np.random.choice(np.setdiff1d(list(self.warehouseInstance.OutputStations.keys()), curr_ps))
        return new_ps

    def simulatedAnnealing(self, alpha=0.975, maxIteration=1000, minTemperature=0.1, mutation_prob=0.2,
                           max_it_without_change=30, batch_id=None):
        # Simulated Annealing Parameters
        if not self.batches:
            # Run the greedy heuristic and get the current solution with the current fitness value
            self.apply_greedy_heuristic()
            self.greedyFitness = self.get_fitness_of_solution()
            print("Fitness of greedy solution: ", self.greedyFitness)
            currentSolution, currentFitness = copy.deepcopy(self.batches), self.greedyFitness
        else:
            currentSolution, currentFitness = copy.deepcopy(self.batches), self.get_fitness_of_solution()

        # if we have only changed one batch, then the optimization of the route is only performed for that
        # particular batch
        if batch_id:
            currentSolution = {batch_id: currentSolution[batch_id]}
        print("Starting simulated annealing with current fitness: ", currentFitness)
        for batch_id in currentSolution.keys():
            iteration = 0
            it_without_change = 0
            T = max(self.distance_ij.values()) - min(list(self.distance_ij.values()))
            while T >= minTemperature and iteration < maxIteration and it_without_change < max_it_without_change:
                batch = copy.deepcopy(currentSolution[batch_id])
                if np.random.random() <= mutation_prob:
                    batch.pack_station = self.switch_stations(batch)
                    self.greedy_cobot_tour(batch)  # recalculate the greedy tour with new pack station
                made_change = self.two_opt(batch, currentSolution, batch_id, T)
                it_without_change += 1 if not made_change else 0
                T *= alpha
                iteration += 1
            self.batches[batch_id] = currentSolution[batch_id]
            self.batches = {value.ID: value for value in
                            self.batches.values()}  # update keys in accordence to possible new batch ids
        print("Fitness of Simulated Annealing: ", self.get_fitness_of_solution())


class IteratedLocalSearch(SimulatedAnnealing):
    def __init__(self):
        super(IteratedLocalSearch, self).__init__()
        self.simulatedAnnealing()

    def update_batches_from_new_order(self, new_order, batch_id):
        update_batch = self.batches[batch_id]
        update_batch.orders.append(new_order)
        update_batch.items.extend(self.get_items_by_order(new_order))
        update_batch.weight += self.get_total_weight_by_order(new_order)
        update_batch.route = []
        self.greedy_cobot_tour(update_batch)
        #print("solution before sim ann: ", self.get_fitness_of_solution())
        self.simulatedAnnealing(batch_id=batch_id, mutation_prob=0.2)

    def perturbation(self, history):
        print("Performing perturbation")
        weight_dict = self.get_weight_per_batch()
        probabilities = []
        keys = []
        for key, weight in weight_dict.items():
            if key == history:
                continue
            else:
                probabilities.append(1 - weight / self.batch_weight)
                keys.append(key)
        probabilities = [float(i) / sum(probabilities) for i in probabilities]
        destroy_batch = np.random.choice(keys, p=probabilities)
        destroy_batch = np.random.choice(keys)
        weight_dict.pop(destroy_batch)
        orders = self.batches[destroy_batch].orders
        self.batches.pop(destroy_batch)
        weight_of_orders = {}
        for order in orders:
            weight_of_orders[order] = self.get_total_weight_by_order(order)
        weight_of_orders = {k: v for k, v in sorted(weight_of_orders.items(), key=lambda item: item[1], reverse=True)}
        for order, weight_of_order in weight_of_orders.items():
            candidate_batches = {}
            for key, weight in weight_dict.items():
                if weight_of_order+weight <= self.batch_weight:
                    candidate_batches[key] = weight_of_order+weight
            if not candidate_batches:
                pack_station = self.get_station_with_min_total_distance(order, True)
                batch_id = pack_station + "_" + str(self.get_new_batch_id())
                self.batches[batch_id] = Batch(batch_id, pack_station, self.station_id_bot_id_dict[pack_station])
                self.update_batches_from_new_order(order, batch_id)
                history = batch_id
            else:
                # choose the batch with maximum workload after assignment
                new_batch_of_order = np.random.choice(list(candidate_batches.keys()))  # max(candidate_batches.keys(), key=(lambda k: candidate_batches[k]))
                #new_batch_of_order = max(candidate_batches.keys(), key=(lambda k: candidate_batches[k]))
                self.update_batches_from_new_order(new_order=order, batch_id=new_batch_of_order)
            # update weight_dict
            weight_dict = self.get_weight_per_batch()
        print("fitness after perturbation: ", self.get_fitness_of_solution())
        return history

    def replace_order(self, batch: Batch, add_order, remove_order):
        batch.orders.remove(remove_order)
        for item in self.get_items_by_order(remove_order):
            del batch.items[batch.items.index(item)]

        batch.orders.append(add_order)
        batch.items.extend(self.get_items_by_order(add_order))

        batch.weight += (self.get_total_weight_by_order(add_order) - self.get_total_weight_by_order(remove_order))

        self.greedy_cobot_tour(batch)  # optimize tour to accurately validate the change in fitness
        self.simulatedAnnealing(batch_id=batch.ID, mutation_prob=0.2)

    def determine_switchable_orders(self, batch_i, batch_j):
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
                weight_diff = self.get_total_weight_by_order(order_i) - self.get_total_weight_by_order(order_j)
                if weight_diff < 0:
                    if abs(weight_diff) <= self.batch_weight - batch_i.weight:
                        return order_i, order_j
                elif weight_diff > 0:
                    if weight_diff <= self.batch_weight - batch_j.weight:
                        return order_i, order_j
        return None, None

    def local_search(self, max_changes=10, history=None):
        """
        perform simple randomized swap of orders to batches
        :return:
        """
        iters = 0
        curr_fit = self.get_fitness_of_solution()
        curr_sol = copy.deepcopy(self.batches)
        while iters < max_changes:
            print("do local search")
            batch_i, batch_j = np.random.choice([batch for key, batch in self.batches.items() if key != history and len(batch.orders)>1], 2, replace=False)
            order_i, order_j = self.determine_switchable_orders(batch_i, batch_j)
            if order_i:
                self.replace_order(batch_i, add_order=order_j, remove_order=order_i)
                self.replace_order(batch_j, add_order=order_i, remove_order=order_j)
                if self.get_fitness_of_solution() < curr_fit:
                    curr_sol = copy.deepcopy(self.batches)
                    curr_fit = self.get_fitness_of_solution()
                else:
                    self.batches = copy.deepcopy(curr_sol)
            iters += 1

    def perform_ils(self, num_iters, t_max, threshold=10):
        iter = 0
        iter_last = 0
        starttime = time.time()
        t = time.time() - starttime
        curr_fit = self.get_fitness_of_solution()
        best_fit = curr_fit
        curr_sol = copy.deepcopy(self.batches)
        best_sol = copy.deepcopy(curr_sol)
        history = None
        T = self.get_fitness_of_solution() * 0.03
        while iter < num_iters and t < t_max:
            iter += 1
            history = self.perturbation(history)
            self.local_search(20)
            neighbor_fit = self.get_fitness_of_solution()
            if neighbor_fit < curr_fit or self.acceptWithProbability(neighbor_fit, curr_fit, T):
                curr_fit = neighbor_fit  # accept new solution
                curr_sol = copy.deepcopy(self.batches)
                iter_last = 0
                if curr_fit < best_fit:
                    best_sol = copy.deepcopy(self.batches)
                    best_fit = self.get_fitness_of_solution()
            elif (iter - iter_last) > threshold:
                print("RESTART")
                self.apply_greedy_heuristic()
                curr_sol = self.get_fitness_of_solution()
                curr_fit = self.get_fitness_of_solution()
                iter, iter_last = 0, 0
            else:
                self.batches = copy.deepcopy(curr_sol)  # don´t accept new solution and stick with the current one
                iter_last += 1
            iter += 1
            T *= 0.85
            t = time.time() - starttime
            print("Fitness after iteration of ILS: ", self.get_fitness_of_solution())
        self.batches = best_sol


if __name__ == "__main__":
    # start = time.time()
    # ils = GreedyHeuristic()
    # ils.apply_greedy_heuristic()
    # print("runtime:, ", time.time() - start)
    SKUS = ["24", "360"]  # options: 24 and 360
    SUBSCRIPTS = ["", "_a", "_b"]
    NUM_ORDERSS = [20]  #[10,


    instance_sols = {}
    for SKU in SKUS:
        for SUBSCRIPT in SUBSCRIPTS:
            for NUM_ORDERS in NUM_ORDERSS:
                try:
                    layoutFile = r'data/layout/1-1-1-2-1.xlayo'
                    podInfoFile = 'data/sku{}/pods_infos.txt'.format(SKU)
                    instances = {}
                    instances[24, 2] = r'data/sku{}/layout_sku_{}_2.xml'.format(SKU, SKU)

                    storagePolicies = {}
                    storagePolicies['dedicated'] = 'data/sku{}/pods_items_dedicated_1.txt'.format(SKU)
                    # storagePolicies['mixed'] = 'data/sku{}/pods_items_mixed_shevels_1-5.txt'.format(SKU)

                    orders = {}
                    orders['{}_5'.format(str(NUM_ORDERS))] = r'data/sku{}/orders_{}_mean_5_sku_{}{}.xml'.format(SKU, str(NUM_ORDERS),
                                                                                                                SKU, SUBSCRIPT)
                    #orders['10_5'] = r'data/sku{}/orders_10_mean_1x6_sku_{}.xml'.format(SKU, SKU)
                    sols_and_runtimes = {}
                    runtimes=[5, 20, 40, 80, 140, 300]
                    #runtimes=[25]
                    for runtime in runtimes:
                        np.random.seed(52320381)
                        if runtime == 0:
                            ils = GreedyHeuristic()
                            ils.apply_greedy_heuristic()
                        else:
                            ils = IteratedLocalSearch()
                            ils.perform_ils(1500, runtime)
                        sols_and_runtimes[runtime] = (ils.get_fitness_of_solution(), {batch.ID: batch.determine_edges(ils.item_id_pod_id_dict) for batch in ils.batches.values()})
                    print(sols_and_runtimes)
                except:
                    continue
                instance_sols[(SKU, SUBSCRIPT, NUM_ORDERS)] = sols_and_runtimes

    with open('../analyse_solution/solutions/dedicated3.pickle', 'wb') as handle:
        pickle.dump(instance_sols, handle, protocol=pickle.HIGHEST_PROTOCOL)

