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

layoutFile = r'data/layout/1-1-1-2-1.xlayo'
podInfoFile = 'data/sku{}/pods_infos.txt'.format(SKU)

instances = {}
instances[24, 2] = r'data/sku{}/layout_sku_{}_2.xml'.format(SKU, SKU)

storagePolicies = {}
# storagePolicies['dedicated'] = 'data/sku{}/pods_items_dedicated_1.txt'.format(SKU)
storagePolicies['mixed'] = 'data/sku{}/pods_items_mixed_shevels_1-5.txt'.format(SKU)

orders = {}
# orders['10_5'] = r'data/sku{}/orders_10_mean_5_sku_{}.xml'.format(SKU, SKU)
orders['20_5'] = r'data/sku{}/orders_20_mean_5_sku_{}.xml'.format(SKU, SKU)

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


    def assign_orders_to_batches_greedy(self, pack_station):
        orders_of_station = self.assign_orders_to_stations()[pack_station]
        already_assigned = list()
        while len(already_assigned) != len(orders_of_station):
            batch_id = pack_station + "_" + str(self.batch_count)
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
            self.batch_count += 1

    def greedy_cobot_tour(self, batch: Batch):
        try:
            assert len(batch.route) == 0
        except AssertionError:
            batch.route = []
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
        if self.batches:
            self.batches = dict()
        for pack_station in self.warehouseInstance.OutputStations.keys():
            self.assign_orders_to_batches_greedy(pack_station)
        for batch in self.batches.values():
            self.greedy_cobot_tour(batch)
        merge_batches = True
        while merge_batches:
            merge_batches = self.merge_batches_of_stations()
            self.update_solution()

    def merge_batches_of_stations(self):
        """
        if batches (of two different packing stations) can be merged in terms of the weight, this merge
        is performed in this function.
        :return:
        """
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
        self.update_batches_after_merge(merged_batches_df)
        return True

    def update_batches_after_merge(self, merged_batches_df):
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
            # delete previous route
            update_batch.route = []
            self.greedy_cobot_tour(update_batch)
            self.batches.pop(delete_batch.ID)
            del delete_batch

    def update_solution(self):
        """
        function to update the solution after changing the assignment of orders to batches
        --> lösung des tsp muss ausgelagert werden. am ende dann eine funktion die alles aufruft aber selber keine
        heurisitk mehr implementiert
        :return:
        """
        pass


class SimulatedAnnealing(GreedyHeuristic):
    def __init__(self):
        super(SimulatedAnnealing, self).__init__()

    def accept(self, batch: Batch, candidate_route, T, pack_station=None):
        current_route = batch.route
        pack_station = batch.pack_station if not pack_station else pack_station
        currentFitness = self.get_fitness_of_tour(current_route, pack_station)
        candidateFitness = self.get_fitness_of_tour(candidate_route, pack_station)
        # print("currentfit:",currentFitness)
        # print("candidatefit:",candidateFitness)
        if candidateFitness < currentFitness:
            return True
        else:
            if np.random.random() < self.acceptWithProbability(candidateFitness, currentFitness, T):
                return True

    def acceptWithProbability(self, candidateFitness, currentFitness, T):
        # Accept the new tour for all cases where fitness of candidate => fitness current with a probability
        return math.exp(-abs(candidateFitness - currentFitness) / T)

    def two_opt(self, tour):
        length = range(len(tour))
        i, j = random.sample(length, 2)
        tour = tour[:]
        tour[i], tour[j] = tour[j], tour[i]
        reverse_order = [tour[m] for m in reversed(range(i + 1, j))]
        tour[i + 1: j] = reverse_order

        return tour

    # Swaps the position of two points in the solution of each batch
    def swap(self, seq):
        # length is the length of the tour of one batch
        length = range(len(seq))
        i1, i2 = random.sample(length, 2)
        seq[i1], seq[i2] = seq[i2], seq[i1]
        return seq

    def switch_stations(self, batch_id):
        """
        TODO: vary the pack station of batches
        :return:
        """
        curr_ps = self.batches[batch_id].pack_station
        new_ps = np.random.choice(np.setdiff1d(list(self.warehouseInstance.OutputStations.keys()), curr_ps))
        return new_ps

    def simulatedAnnealing(self, batch_id=None):
        # Simulated Annealing Parameters
        if not self.batches:
            # Run the greedy heuristic and get the current solution with the current fitness value
            self.apply_greedy_heuristic()
            self.greedyFitness = self.get_fitness_of_solution()
            print("Fitness of greedy solution: ", self.greedyFitness)
            currentSolution, currentFitness = copy.deepcopy(self.batches), self.greedyFitness
        else:
            currentSolution, currentFitness = copy.deepcopy(self.batches), self.get_fitness_of_solution()
        # Accept the new tour for all cases where fitness of candidate < fitness current
        print("Starting simulated annealing with current fitness: ", currentFitness)
        if batch_id:
            currentSolution =  {batch_id: currentSolution[batch_id]}
        for batch_id, batch in currentSolution.items():
            alpha = 0.975
            T = 4
            iteration = 1
            maxIteration = 1000
            minTemperature = 0.1
            if np.random.random() >= 0.8:
                candidate_ps = self.switch_stations(batch_id)
                print("New pack station for batch {}: ".format(batch_id), candidate_ps)
                batch.pack_station = candidate_ps
                self.greedy_cobot_tour(batch)
            else:
                candidate_ps = batch.pack_station

            while T >= minTemperature and iteration < maxIteration:
                if len(batch.route) > 1:
                    pointCopy = batch.route[:]
                    candidate = self.two_opt(pointCopy)
                    if self.accept(batch, candidate, T, candidate_ps):
                        currentSolution[batch_id].route = candidate
                        currentSolution[batch_id].pack_station = candidate_ps
                T *= alpha
                iteration += 1
            self.batches[batch_id] = currentSolution[batch_id]
        print("Fitness of Simulated Annealing: ", self.get_fitness_of_solution())


class IteratedLocalSearch(SimulatedAnnealing):
    def __init__(self):
        super(IteratedLocalSearch, self).__init__()
        self.simulatedAnnealing()

    def perturbation(self):
        print("Performing perturbation")
        weight_dict = self.get_weight_per_batch()
        probabilities = []
        keys = []
        for key, weight in weight_dict.items():
            probabilities.append(1-weight/self.batch_weight)
            keys.append(key)
        probabilities = [float(i) / sum(probabilities) for i in probabilities]
        destroy_batch = np.random.choice(keys, p=probabilities)
        # destroy_batch = np.random.choice(keys)
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
                batch_id = pack_station + "_" + str(self.batch_count)  # todo make batch count accessible
                self.batches[batch_id] = Batch(batch_id, pack_station, self.station_id_bot_id_dict[pack_station])
                self.update_batches_from_new_order(order, batch_id)
                self.batch_count += 1
            else:
                # choose the batch with maximum workload after assignment
                new_batch_of_order = max(candidate_batches.keys(), key=(lambda k: candidate_batches[k]))
                self.update_batches_from_new_order(new_order=order, batch_id=new_batch_of_order)
            # update weight_dict
            weight_dict = self.get_weight_per_batch()
        print("fitness after perturbation: ", self.get_fitness_of_solution())

    def update_batches_from_new_order(self, new_order, batch_id):
        update_batch = self.batches[batch_id]
        update_batch.orders.append(new_order)
        update_batch.items.extend(self.get_items_by_order(new_order))
        update_batch.weight += self.get_total_weight_by_order(new_order)
        update_batch.route = []
        self.greedy_cobot_tour(update_batch)
        #print("solution before sim ann: ", self.get_fitness_of_solution())
        self.simulatedAnnealing(batch_id)

    def perform_ils(self, num_iters, threshold):
        iter = 0
        iter_last = 0
        curr_fit = self.get_fitness_of_solution()
        best_fit = curr_fit
        curr_sol = copy.deepcopy(self.batches)
        best_sol = copy.deepcopy(curr_sol)
        while iter < num_iters:
            iter += 1
            self.perturbation()
            neighbor_fit = self.get_fitness_of_solution()
            if neighbor_fit < curr_fit:
                curr_fit = neighbor_fit  # accept new solution
                curr_sol = copy.deepcopy(self.batches)
                iter_last = iter
                if curr_fit < best_fit:
                    best_sol = copy.deepcopy(self.batches)
                    best_fit = self.get_fitness_of_solution()
            elif neighbor_fit >= curr_fit and (iter - iter_last) > threshold:
                self.apply_greedy_heuristic()
                curr_sol = self.get_fitness_of_solution()  # reset
            else:
                self.batches = copy.deepcopy(curr_sol)  # don´t accept new solution and stick with the current one
        return best_sol, best_fit


# ----------------------------------------------------------------------------------------------------------------------


class GreedyMixedShelves(Demo):
    def __init__(self):
        super(GreedyMixedShelves, self).__init__()
        self.item_id_pod_id_dict = defaultdict(dict)
        self.solution = None
        self.orders_of_batches = dict()
        self.batches = dict()
        self.batch_count = 0
        self.fill_item_id_pod_id_dict()
        self.fill_station_id_bot_id_dict()
        self.warehouseInstance.Orders = {str(key): value for key, value in enumerate(self.warehouseInstance.Orders)}
        self.could_not_assign = defaultdict(list)

    def fill_item_id_pod_id_dict(self):
        """
        For the mixed shelve policy, the amount of items stored in a shelf is a viable information and thus needs
        to be retrieved here. In a dedicated policy where we assume that the number of items stored in at least equal
        to the number of ordered items, we dont need the supply quantity.
        :return:
        """
        trie = self.build_trie_for_items()
        for key, pod in self.warehouseInstance.Pods.items():
            for item in pod.Items:
                ids = item.ID
                color, product = ids.split("/")
                item_id = trie[product][color]
                self.item_id_pod_id_dict[item_id][key] = int(item.Count)


    def get_items_by_order(self, order_idx):
        """
        In contrary to the base get_items_by_order function this function returns a list containing an item as
        often as it has been ordered instead of only containing the unique items of an order. By doing so, each item
        (even items of the same SKU) are considered as unique and for each item positoin we only need to pick 1 item
        (quantity) of a shelf (and thus only need to check whether a shelf has more than zero items of that SKU.
        :param order_idx:
        :return:
        """
        item_ids = []
        for item_id, item in self.warehouseInstance.Orders[order_idx].Positions.items():
            item_ids.extend([item_id] * int(item.Count))
        return item_ids

    def get_items_plus_quant_by_order(self, order_idx) -> dict:
        """
        different approach to get_items_by_order but not implemented / tested yet. Returns a dictionary with unique
        items plus the order quantity as value.
        :param order_idx:
        :return:
        """
        item_ids = dict()
        for item_id, item in self.warehouseInstance.Orders[order_idx].Positions.items():
            item_ids[item_id] = int(item.Count)
        return item_ids

    def get_total_weight_by_order(self, order_idx):
        """
        In correspondence with the new get_items_by_order function, the items dont have to be multiplied with
        the order quantity
        :param order_idx:
        :return:
        """
        weights = []
        items_of_order = self.get_items_by_order(order_idx)
        for item in items_of_order:
            weights.append(self.get_weight_by_item(item))
        return np.sum(weights)

    def get_station_with_min_total_distance(self, order_idx, retrieve_station_only=False):
        """
        function to retrieve the station which minimizes the total distance of the items of the
        specified order
        :param order_idx: index or the order
        :param retrieve_station_only: indicates whether to return the station with min distance only or
        a together with the distance which can be used for further decision making.
        :return:
        """
        items = self.get_items_plus_quant_by_order(order_idx)  # todo refactor with get_items_by_order function
        item_id_pod_id_dict = copy.deepcopy(self.item_id_pod_id_dict)
        order_pack_dists = dict()
        for pack_station in list(self.warehouseInstance.OutputStations.keys()):
            item_dists = []
            for item_id, item_quant in items.items():
                shelves_with_item = list(item_id_pod_id_dict[item_id].keys())
                shelf_distances = {}
                for shelf in shelves_with_item:
                    if item_id_pod_id_dict[item_id][shelf] >= item_quant:
                        shelf_distances[shelf] = self.distance_ij[pack_station, shelf]
                shelf_with_min_dist = min(shelf_distances.keys(), key=(lambda k: shelf_distances[k]))
                item_id_pod_id_dict[item_id][shelf_with_min_dist] -= item_quant
                item_dists.append(shelf_distances[shelf_with_min_dist])
            order_pack_dists[pack_station] = np.sum(item_dists)
        return min(order_pack_dists.keys(), key=(lambda k: order_pack_dists[k]))

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

    def greedy_next_order_to_batch(self, batch: BatchNew, already_assigned, orders_of_station, forbidden=None):
        if forbidden is None:
            forbidden = []
        other_orders_of_station = np.setdiff1d(orders_of_station, already_assigned+forbidden)
        print("other unassigned orders in the same station ({}) as this batch: ".format(batch.pack_station), other_orders_of_station)
        sum_min_dist_to_item = dict()
        for order in other_orders_of_station:
            min_distances = []
            for item_in_batch_id, item_in_batch in batch.items.items():
                dist_per_item_to_curr_item = []
                for item in self.get_items_by_order(order):
                    min_dist_shelf = min([self.distance_ij[item_in_batch.shelf, shelf] for shelf in list(self.item_id_pod_id_dict[item].keys())])
                    dist_per_item_to_curr_item.append(min_dist_shelf)
                min_distances.append(min(dist_per_item_to_curr_item))

            sum_min_dist_to_item[order] = np.sum(min_distances)  # average instead? otherwise we always pick small orders

        return min(sum_min_dist_to_item.keys(), key=(lambda k: sum_min_dist_to_item[k]))

    def assign_shelves_to_items(self, curr_node, items_to_pick):
        item_penalties = {}
        item_shelves = {}
        for item in items_to_pick:
            penalty_vals = {}
            max_kappa = max(int(self.item_id_pod_id_dict[item.orig_ID][shelf]) for shelf in item.shelves)
            for shelf in item.shelves:
                dist = self.distance_ij[curr_node, shelf]
                cappa = self.item_id_pod_id_dict[item.orig_ID][shelf]
                penalty_vals[shelf] = dist / (cappa / max_kappa) if cappa > 0 else float('inf')
            min_shelf_for_item = min(penalty_vals.keys(), key=(lambda k: penalty_vals[k]))
            penalty_for_item = penalty_vals[min_shelf_for_item]
            item_penalties[item.ID] = penalty_for_item
            item_shelves[item.ID] = min_shelf_for_item
        min_item = min(item_penalties.keys(), key=(lambda k: item_penalties[k]))
        min_shelf = item_shelves[min_item]

        return min_item, min_shelf

    def replenish_shelves(self, batch: BatchNew):
        """
        if a batch is destroyed, the items that were taken from the shelves during the tour need to be added to the
        shelves again.
        :param batch:
        :return:
        """
        for item_id, item in batch.items.items():
            if item.shelf:
                self.item_id_pod_id_dict[item.orig_ID][item.shelf] += 1
            else:
                pass  # SKU not scheduled yet, thus no item has been taken from the shelves

    def calc_increase_dist(self, route, shelf, i, pack_station):
        if len(route) == 0:
            return self.distance_ij[pack_station, shelf] + self.distance_ij[shelf, pack_station]
        # we favor solutions where the same shelve is visited all at once and not at different parts of the route,
        # thus give it a negative penalty here
        try:
            if route[i] == shelf:
                return -.1
        except IndexError:
            if route[i-1] == shelf:
                return -.1
        # if the shelf at the current position of the tour is not equal to the candidate shelf, the distance added to
        # the tour by candidate shelf j at position i is calculated as follows: d_{i-1, j} + d{j, i} - d{i-1, i}
        # (pack stations are added manually as they are not "part" of route object)
        if i == 0:
            add_dist = self.distance_ij[pack_station, shelf] + self.distance_ij[shelf, route[i]]
            subtr_dist = self.distance_ij[pack_station, route[i]]
        elif i == len(route):
            add_dist = self.distance_ij[route[-1], shelf] + self.distance_ij[shelf, pack_station]
            subtr_dist = self.distance_ij[route[-1], pack_station]
        else:
            add_dist = self.distance_ij[route[i-1], shelf] + self.distance_ij[shelf, route[i]]
            subtr_dist = self.distance_ij[route[i-1], route[i]]
        return add_dist - subtr_dist

    def greedy_cobot_tour(self, batch: BatchNew, items=None):
        """
        function to determine a (initial) route in a greedy manner
        :param batch:
        :param items:
        :return:
        """
        if not items:
            items = copy.deepcopy(batch.items)
            batch.route = []  # if all items of a batch shell be scheduled, the current tour needs to be discarded
            self.replenish_shelves(batch)  # replenish shelves
        else:
            items = copy.deepcopy(items)

        for item in items.values():
            batch_copy = copy.deepcopy(batch)
            shelf_add_distances = {}
            blacklisted = []
            for shelf in item.shelves:
                if self.item_id_pod_id_dict[item.orig_ID][shelf] == 0:
                    blacklisted.append(shelf)
                added_dists = {i: self.calc_increase_dist(batch_copy.route, shelf, i, batch_copy.pack_station)
                               for i in range(len(batch.route) + 1)}
                min_pos, min_pos_dist = min(added_dists.items(), key=itemgetter(1))
                shelf_add_distances[shelf, min_pos] = min_pos_dist
               # else:
                   # print("Shelf {} has not more units of item {}.".format(shelf, item.orig_ID))
            # min_shelf, min_pos = min(shelf_add_distances, key=shelf_add_distances.get)
            while True:
                min_shelf, min_pos = min(shelf_add_distances, key=shelf_add_distances.get)
                if min_shelf in blacklisted:
                    shelf_add_distances.pop((min_shelf,min_pos))
                    self.could_not_assign[item.ID].append(min_shelf)
                else:
                    break
            batch.route_insert(min_pos, min_shelf, item)
            batch.items[item.ID].shelf = min_shelf
            self.item_id_pod_id_dict[item.orig_ID][min_shelf] -= 1

    def greedy_cobot_tour(self, batch: BatchNew, items=None):
        """
        function to determine a (initial) route in a greedy manner
        :param batch:
        :param items:
        :return:
        """
        if not items:
            items = copy.deepcopy(batch.items)
            batch.route = []  # if all items of a batch shell be scheduled, the current tour needs to be discarded
            self.replenish_shelves(batch)  # replenish shelves
        else:
            items = copy.deepcopy(items)

        for item in items.values():
            batch_copy = copy.deepcopy(batch)
            shelf_add_distances = {}
            blacklisted = []
            for shelf in item.shelves:
                if self.item_id_pod_id_dict[item.orig_ID][shelf] == 0:
                    blacklisted.append(shelf)
                added_dists = {i: self.calc_increase_dist(batch_copy.route, shelf, i, batch_copy.pack_station)
                               for i in range(len(batch.route) + 1)}
                min_pos, min_pos_dist = min(added_dists.items(), key=itemgetter(1))
                shelf_add_distances[shelf, min_pos] = min_pos_dist
               # else:
                   # print("Shelf {} has not more units of item {}.".format(shelf, item.orig_ID))
            # min_shelf, min_pos = min(shelf_add_distances, key=shelf_add_distances.get)
            while True:
                min_shelf, min_pos = min(shelf_add_distances, key=shelf_add_distances.get)
                if min_shelf in blacklisted:
                    shelf_add_distances.pop((min_shelf,min_pos))
                    self.could_not_assign[item.ID].append(min_shelf)
                else:
                    break
            batch.route_insert(min_pos, min_shelf, item)
            batch.items[item.ID].shelf = min_shelf
            self.item_id_pod_id_dict[item.orig_ID][min_shelf] -= 1

    def get_new_batch_id(self):
        """after destroying a batch, new IDs have to be given to new batches. This function looks, if IDs are
        available within a sequence of numbers, or otherwise adds one to the highest ID number"""
        if not self.batches:
            return 0
        curr_batch_ids = [int(batch.ID.split("_")[1]) for batch in self.batches.values()]
        sequence = np.arange(min(curr_batch_ids), max(curr_batch_ids)+1, 1)
        gaps = np.setdiff1d(sequence, curr_batch_ids)
        if len(gaps)==0:
            return max(curr_batch_ids) + 1
        else:
            return min(gaps)

    def assign_orders_to_batches_greedy(self, pack_station):
        orders_of_station = self.assign_orders_to_stations()[pack_station]
        already_assigned = list()
        while len(already_assigned) != len(orders_of_station):
            batch_id = pack_station + "_" + str(self.get_new_batch_id())
            batch = BatchNew(batch_id, pack_station, self.station_id_bot_id_dict[pack_station])
            print("initialized new batch with ID: ", batch.ID)
            # initialize the batch with a random order
            init_order = np.random.choice(np.setdiff1d(orders_of_station, already_assigned))
            items_of_order = self.get_items_by_order(init_order)
            weight_of_order = self.get_total_weight_by_order(init_order)
            init_order = OrderOfBatch(init_order, items_of_order, weight_of_order, self.item_id_pod_id_dict)
            batch.add_order(init_order)
            self.greedy_cobot_tour(batch)

            already_assigned.append(init_order.ID)

            forbidden_for_batch = []
            while batch.weight < self.batch_weight and not len(
                    np.union1d(already_assigned, forbidden_for_batch)) == len(orders_of_station):

                new_order = self.greedy_next_order_to_batch(batch, already_assigned, orders_of_station, forbidden_for_batch)
                weight_of_order = self.get_total_weight_by_order(new_order)
                print("Chosen order: ", new_order)
                if (batch.weight + weight_of_order) <= self.batch_weight:
                    print("and it also fits in the current batch ({})".format(batch_id))
                    already_assigned.append(new_order)
                    items_of_order = self.get_items_by_order(new_order)
                    weight_of_order = self.get_total_weight_by_order(new_order)
                    new_order = OrderOfBatch(new_order, items_of_order, weight_of_order, self.item_id_pod_id_dict)
                    batch.add_order(new_order)
                    self.greedy_cobot_tour(batch, items=new_order.items)
                else:
                    print("but it would add too much weight to the batch, go on to next...")
                    forbidden_for_batch.append(new_order)
                print("the current batch ({}) looks as follows: {}".format(batch.ID, batch.orders))
            self.batches[batch_id] = batch
            self.batch_count += 1

    def apply_greedy_heuristic(self):
        self.item_id_pod_id_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
        if self.batches:
            self.batches = dict()
        for pack_station in self.warehouseInstance.OutputStations.keys():
            self.assign_orders_to_batches_greedy(pack_station)

    def get_fitness_of_batch(self, batch: BatchNew):
        distances = []
        distances.append(self.distance_ij[batch.pack_station, batch.route[0]])
        for first, second in zip(batch.route, batch.route[1:]):
            distances.append(self.distance_ij[first,second])
        distances.append(self.distance_ij[batch.pack_station, batch.route[-1]])
        return np.sum(distances)

    def get_fitness_of_tour(self, batch_route, pack_station):
        distances = []
        distances.append(self.distance_ij[pack_station, batch_route[0]])
        for first, second in zip(batch_route, batch_route[1:]):
            distances.append(self.distance_ij[first, second])
        distances.append(self.distance_ij[pack_station, batch_route[-1]])
        return np.sum(distances)

    def get_fitness_of_solution(self):
        distances = []
        for batch in self.batches.values():
            distances.append(self.get_fitness_of_batch(batch))
        return np.sum(distances)


class SimulatedAnnealingMixed(GreedyMixedShelves):
    def __init__(self):
        super(SimulatedAnnealingMixed, self).__init__()

    def accept(self, current_route, candidate_route, T, pack_station=None):
        # current_route = batch.route
        # pack_station = batch.pack_station if not pack_station else pack_station
        currentFitness = self.get_fitness_of_tour(current_route, pack_station)
        candidateFitness = self.get_fitness_of_tour(candidate_route, pack_station)
        # print("currentfit:",currentFitness)
        # print("candidatefit:",candidateFitness)
        if candidateFitness < currentFitness:
            return True
        else:
            if np.random.random() < self.acceptWithProbability(candidateFitness, currentFitness, T):
                return True

    def acceptWithProbability(self, candidateFitness, currentFitness, T):
        # Accept the new tour for all cases where fitness of candidate => fitness current with a probability
        return math.exp(-abs(candidateFitness - currentFitness) / T)

    def two_opt(self, tour):
        length = range(len(tour))
        i, j = np.random.choice(length, 2)
        tour = tour[:]
        tour[i], tour[j] = tour[j], tour[i]
        reverse_order = [tour[m] for m in reversed(range(i + 1, j))]
        tour[i + 1: j] = reverse_order

        return tour

    def switch_stations(self, batch_id):
        """
        TODO: vary the pack station of batches
        :return:
        """
        curr_ps = self.batches[batch_id].pack_station
        new_ps = np.random.choice(np.setdiff1d(list(self.warehouseInstance.OutputStations.keys()), curr_ps))
        return new_ps

    def change_shelf_assignment(self, batch: BatchNew, route=None):
        curr_fitness = self.get_fitness_of_batch(batch)
        batch_copy = copy.deepcopy(batch)
        if route:
            batch_copy.route = route
        shelf_w_one_item_to_pick = [k for k, v in list(batch.items_of_shelves.items()) if len(v) == 1]
        if len(shelf_w_one_item_to_pick) > 0:
            shelf_to_destroy = np.random.choice(shelf_w_one_item_to_pick)
            item_of_shelf = batch.items_of_shelves[shelf_to_destroy][0]
            batch_copy.route.remove(shelf_to_destroy)
            batch_copy.items_of_shelves.pop(shelf_to_destroy)
            shelf_add_distances = {}
            blacklisted = []
            for shelf in np.setdiff1d(item_of_shelf.shelves, shelf_to_destroy):
                if self.item_id_pod_id_dict[item_of_shelf.orig_ID][shelf] == 0:
                    blacklisted.append(shelf)
                added_dists = {i: self.calc_increase_dist(batch_copy.route, shelf, i, batch_copy.pack_station)
                               for i in range(len(batch_copy.route) + 1)}
                min_pos, min_pos_dist = min(added_dists.items(), key=itemgetter(1))
                shelf_add_distances[shelf, min_pos] = min_pos_dist

            # min_shelf, min_pos = min(shelf_add_distances, key=shelf_add_distances.get)
            while True:
                min_shelf, min_pos = min(shelf_add_distances, key=shelf_add_distances.get)
                if min_shelf in blacklisted:
                    shelf_add_distances.pop(min_shelf)
                    self.could_not_assign[item_of_shelf].append(min_shelf)
                else:
                    break
            # item.shelf = min_shelf
            batch_copy.route_insert(min_pos, min_shelf, item_of_shelf)
            batch_copy.items[item_of_shelf.ID].shelf = min_shelf
            cand_fit = self.get_fitness_of_batch(batch_copy)
            #if cand_fit < curr_fitness:
            #    self.batches[batch.ID] = batch_copy
            #    self.item_id_pod_id_dict[item_of_shelf.orig_ID][shelf_to_destroy] += 1
            #    self.item_id_pod_id_dict[item_of_shelf.orig_ID][min_shelf] -= 1
            return batch_copy, item_of_shelf, shelf_to_destroy, min_shelf
        else:
            return batch_copy, None, None, None




    def simulatedAnnealing(self, alpha=0.975, T=4, maxIteration=1000, minTemperature=0.1, k=1, batch_id=None):
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
        for batch_id, batch in currentSolution.items():
            iteration = 1
            if np.random.random() >= 0.8:
                candidate_ps = self.switch_stations(batch_id)
                print("New pack station for batch {}: ".format(batch_id), candidate_ps)
                batch.pack_station = candidate_ps
                self.greedy_cobot_tour(batch)  # recalculate the greedy tour with new pack station
            else:
                candidate_ps = batch.pack_station

            while T >= minTemperature and iteration < maxIteration:
                if len(batch.route) > 1:
                    curr_route = batch.route[:]
                    #candidate = self.two_opt(pointCopy)
                    batch.route = self.two_opt(batch.route)
                    if k == 2:
                        batch, item, old_shelf, new_shelf = self.change_shelf_assignment(batch)
                    else:
                        item = None
                    if self.accept(curr_route, batch.route, T, candidate_ps):
                        currentSolution[batch_id] = batch
                        if item:
                            self.item_id_pod_id_dict[item.orig_ID][old_shelf] += 1
                            self.item_id_pod_id_dict[item.orig_ID][old_shelf] -= 1
                T *= alpha
                iteration += 1
            self.batches[batch_id] = currentSolution[batch_id]
            self.batches = {value.ID: value for value in self.batches.values()}  # update keys in accordence to possible new batch ids
        print("Fitness of Simulated Annealing: ", self.get_fitness_of_solution())



class IteratedLocalSearchMixed(SimulatedAnnealingMixed):
    def __init__(self):
        super(IteratedLocalSearchMixed, self).__init__()
        self.simulatedAnnealing()  # get initial solution

    def perturbation(self):
        print("Performing perturbation")
        weight_dict = self.get_weight_per_batch()
        probabilities = []
        keys = []
        for key, weight in weight_dict.items():
            probabilities.append(1-weight/self.batch_weight)
            keys.append(key)
        probabilities = [float(i) / sum(probabilities) for i in probabilities]
        destroy_batch = np.random.choice(keys, p=probabilities)
        # destroy_batch = np.random.choice(keys)
        weight_dict.pop(destroy_batch)
        orders = self.batches[destroy_batch].orders
        self.replenish_shelves(self.batches[destroy_batch])
        self.batches.pop(destroy_batch)
        # sort orders with respect to their weight in descending order (try to assign the biggest orders first)
        weight_of_orders = {k: v.weight for k, v in sorted(orders.items(), key=lambda item: item[1].weight, reverse=True)}
        for order, weight_of_order in weight_of_orders.items():
            candidate_batches = {}
            for key, weight in weight_dict.items():
                if weight_of_order+weight <= self.batch_weight:
                    candidate_batches[key] = weight_of_order+weight
            if not candidate_batches:
                pack_station = self.get_station_with_min_total_distance(order, True)
                batch_id = pack_station + "_" + str(self.get_new_batch_id())
                self.batches[batch_id] = BatchNew(batch_id, pack_station, self.station_id_bot_id_dict[pack_station])
                self.update_batches_from_new_order(order, batch_id)
                self.batch_count += 1
            else:
                # choose the batch with maximum workload after assignment
                new_batch_of_order = max(candidate_batches.keys(), key=(lambda k: candidate_batches[k]))
                self.update_batches_from_new_order(new_order=order, batch_id=new_batch_of_order)
            # update weight_dict
            weight_dict = self.get_weight_per_batch()
        print("fitness after perturbation: ", self.get_fitness_of_solution())

    def update_batches_from_new_order(self, new_order, batch_id):
        update_batch = self.batches[batch_id]
        items_of_order = self.get_items_by_order(new_order)
        weight_of_order = self.get_total_weight_by_order(new_order)
        new_order = OrderOfBatch(new_order, items_of_order, weight_of_order, self.item_id_pod_id_dict)
        update_batch.add_order(new_order)
        self.greedy_cobot_tour(update_batch, new_order.items)
        print("partial solution before sim ann (update step): ", self.get_fitness_of_solution())
        self.simulatedAnnealing(batch_id=batch_id)

    def perform_ils(self, num_iters, threshold):
        iter = 0
        iter_last = 0
        curr_fit = self.get_fitness_of_solution()
        best_fit = curr_fit
        curr_sol = copy.deepcopy(self.batches)
        best_sol = copy.deepcopy(curr_sol)
        while iter < num_iters:
            iter += 1
            self.perturbation()
            neighbor_fit = self.get_fitness_of_solution()
            if neighbor_fit < curr_fit:
                curr_fit = neighbor_fit  # accept new solution
                curr_sol = copy.deepcopy(self.batches)
                iter_last = iter
                if curr_fit < best_fit:
                    best_sol = copy.deepcopy(self.batches)
                    best_fit = self.get_fitness_of_solution()
            elif neighbor_fit >= curr_fit and (iter - iter_last) > threshold:
                self.item_id_pod_id_dict = copy.deepcopy(self.item_id_pod_id_dict_copy)
                self.apply_greedy_heuristic()
                curr_sol = self.get_fitness_of_solution()  # reset
            else:
                self.batches = copy.deepcopy(curr_sol)  # don´t accept new solution and stick with the current one
        return best_sol, best_fit


class VariableNeighborhoodSearch(IteratedLocalSearchMixed):
    """
    we need different neighborhoods (x \in N_k, where k = {1,...,k_max} and k_max is typically 2 or 3)
    Idea: Shake1 is perturbation as it is used in ILS. Shake2 is then to destroy 2 batches at once.
    Shake3 could include the assignment of items to shelves (ok das passiert schon bei greedy cobot tour).
    """
    def __init__(self):
        super(VariableNeighborhoodSearch, self).__init__()

    def shake(self, k):
        print("Performing perturbation")
        weight_dict = self.get_weight_per_batch()
        probabilities = []
        keys = []
        for key, weight in weight_dict.items():
            probabilities.append(1 - weight / self.batch_weight)
            keys.append(key)
        probabilities = [float(i) / sum(probabilities) for i in probabilities]
        destroy_batches = np.random.choice(keys, k, replace=False, p=probabilities)  # destroy k batches

        if k==4:
            self.simulatedAnnealing(batch_id=np.random.choice(np.setdiff1d(list(weight_dict.keys()), destroy_batches)), k=2)

        [weight_dict.pop(destroy_batch) for destroy_batch in destroy_batches]
        # merge dictionaries of orders of the batches that should be destroyed
        orders = eval("{"+",".join(["**self.batches[destroy_batches[{}]].orders".format(i) for i in range(len(destroy_batches))])+"}")
        [self.replenish_shelves(self.batches[destroy_batch]) for destroy_batch in destroy_batches]
        [self.batches.pop(destroy_batch) for destroy_batch in destroy_batches]
        # sort orders with respect to their weight in descending order (try to assign the biggest orders first)
        weight_of_orders = {k: v.weight for k, v in
                            sorted(orders.items(), key=lambda item: item[1].weight, reverse=True)}
        for order, weight_of_order in weight_of_orders.items():
            candidate_batches = {}
            for key, weight in weight_dict.items():
                if weight_of_order + weight <= self.batch_weight:
                    candidate_batches[key] = weight_of_order + weight
            if not candidate_batches:
                pack_station = self.get_station_with_min_total_distance(order, True)
                batch_id = pack_station + "_" + str(self.get_new_batch_id())
                self.batches[batch_id] = BatchNew(batch_id, pack_station, self.station_id_bot_id_dict[pack_station])
                self.update_batches_from_new_order(order, batch_id)
                self.batch_count += 1
            else:
                # choose the batch with maximum workload after assignment
                new_batch_of_order = max(candidate_batches.keys(), key=(lambda k: candidate_batches[k]))
                self.update_batches_from_new_order(new_order=order, batch_id=new_batch_of_order)
            # update weight_dict
            weight_dict = self.get_weight_per_batch()
        print("fitness after perturbation: ", self.get_fitness_of_solution())

    def reduced_vns(self, max_iters, t_max, k_max):
        iters = 0
        starttime = time.time()
        t = time.time() - starttime
        curr_fit = self.get_fitness_of_solution()
        best_fit = curr_fit
        curr_sol = copy.deepcopy(self.batches)
        best_sol = copy.deepcopy(curr_sol)
        while iters < max_iters and t < t_max:
            k = 1
            while k < k_max:
                self.shake(k)
                neighbor_fit = self.get_fitness_of_solution()
                if neighbor_fit < curr_fit:
                    curr_fit = neighbor_fit  # accept new solution
                    curr_sol = copy.deepcopy(self.batches)
                    k = 1
                    if curr_fit < best_fit:
                        best_sol = copy.deepcopy(self.batches)
                        best_fit = self.get_fitness_of_solution()
                else:
                    self.batches = copy.deepcopy(curr_sol)  # don´t accept new solution and stick with the current one
                    k += 1
            iters += 1
            t = time.time() - starttime
        return best_sol



if __name__ == "__main__":
    np.random.seed(52331681)
    # _demo = Demo()
    # if _demo.is_storage_dedicated:
    #     greedy = GreedyHeuristic()
    #     greedy.apply_greedy_heuristic()
    #     print("old heuristic: ", greedy.get_fitness_of_solution())
    #ils = IteratedLocalSearchMixed()
    #ils.perform_ils(1, 30)
    ils = VariableNeighborhoodSearch()
    # for i in range(1000):
    #     rd_btch = np.random.choice(list(ils.batches.keys()))
    #     ils.change_shelf_assignment(ils.batches[rd_btch])
    ils.reduced_vns(15, 50, 3)

    print("new heuristic: ", ils.get_fitness_of_solution())
