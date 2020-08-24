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
from instance_demo import Demo

# ------------------------------------------- CONFIG -------------------------------------------------------------------
SKU = "24"  # options: 24 and 360

layoutFile = r'data/layout/1-1-1-2-1.xlayo'
podInfoFile = 'data/sku{}/pods_infos.txt'.format(SKU)

instances = {}
instances[24, 2] = r'data/sku{}/layout_sku_{}_2.xml'.format(SKU, SKU)

storagePolicies = {}
#storagePolicies['dedicated'] = 'data/sku{}/pods_items_dedicated_1.txt'.format(SKU)
storagePolicies['mixed'] = 'data/sku{}/pods_items_mixed_shevels_1-5.txt'.format(SKU)

orders = {}
# orders['10_5'] = r'data/sku{}/orders_10_mean_5_sku_{}.xml'.format(SKU, SKU)
orders['20_5'] = r'data/sku{}/orders_20_mean_5_sku_{}.xml'.format(SKU, SKU)

# ----------------------------------------------------------------------------------------------------------------------

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
        vary the pack station of batches
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
            currentSolution = {batch_id: currentSolution[batch_id]}
        for batch_id, batch in currentSolution.items():
            alpha = 0.975
            T = 4
            iteration = 1
            maxIteration = 1000
            minTemperature = 0.1

            while T >= minTemperature and iteration < maxIteration:
                batch_copy = copy.deepcopy(batch)
                if np.random.random() >= 0.8:
                    candidate_ps = self.switch_stations(batch_id)
                    print("New pack station for batch {}: ".format(batch_id), candidate_ps)
                    batch.pack_station = candidate_ps
                    # self.greedy_cobot_tour(batch)
                else:
                    candidate_ps = batch.pack_station
                if len(batch_copy.route) > 1:
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