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

    def get_fitness_of_tour(self, tour, pack_station):
        complete_tour = [pack_station] + tour + [pack_station]
        distances = []
        for first, second in zip(complete_tour, complete_tour[1:]):
            distances.append(self.distance_ij[first, second])
        return np.sum(distances)

    def get_fitness_of_solution(self, solution):
        distances = []
        for pack_station, value in solution.items():
            for batch, tour in value.items():
                distances.append(self.get_fitness_of_tour(tour, pack_station))
        return np.sum(distances)


class GreedyHeuristic(Demo):
    def __init__(self):
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
                item_dists.append(self.distance_ij[pack_station, item])
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

    def count_min_distances_for_order(self, order_idx, forbidden=None):
        """
        greedy: for any item of the current batch (specified in order_idx) the order which has an item that
        minimizes the distance to that item over all items of all unassigned order of the same station is
        determined (alternatively we can measure the total minimum distance of the items of an order to
        all the items in the current batch --> might be more informative)
        :param order_idx:
        :param forbidden:
        :return:
        """
        if forbidden is None:
            forbidden = []  # to exclude orders that don´t fit in the batch anymore due to limited capacity
        order_idx = [order_idx] if not isinstance(order_idx, list) else order_idx
        station_assignments = self.assign_orders_to_stations()
        station_of_order = [k for k, v in station_assignments.items() if any([idx in v for idx in order_idx])][0]
        other_orders_of_station = list(set(station_assignments[station_of_order]).difference(set(order_idx+forbidden)))
        print("other unassigned orders in the same station ({}) as this order: ".format(station_of_order),
              other_orders_of_station)
        count_min_dist_items_to_order = defaultdict(int)
        min_dist_to_item = dict()
        for idx in order_idx:
            for item in self.get_items_by_order(idx):
                min_dist_to_item_for_order = dict()
                for order in other_orders_of_station:
                    dists_to_item = []
                    for item2 in self.get_items_by_order(order):
                        dists_to_item.append(self.distance_ij[item, item2])
                    min_dist_to_item_for_order[order] = min(dists_to_item)
                min_dist_to_item[item] = min(min_dist_to_item_for_order.keys(),
                                             key=(lambda k: min_dist_to_item_for_order[k]))
                count_min_dist_items_to_order[
                    min(min_dist_to_item_for_order.keys(), key=(lambda k: min_dist_to_item_for_order[k]))] += 1
        return max(count_min_dist_items_to_order.keys(), key=(lambda k: count_min_dist_items_to_order[k]))

    def get_all_items_of_batch(self, batch):
        items = []
        for order in batch:
            items.extend(self.get_items_by_order(order))
        return items

    def greedy_next_order_to_batch(self, batch, forbidden=None):
        if forbidden is None:
            forbidden = []  # to exclude orders that don´t fit in the batch anymore due to limited capacity
        batch = [batch] if not isinstance(batch, list) else batch
        station_assignments = self.assign_orders_to_stations()
        station_of_order = [k for k, v in station_assignments.items() if any([idx in v for idx in batch])][0]
        other_orders_of_station = list(
            set(station_assignments[station_of_order]).difference(set(batch + forbidden)))
        print("other unassigned orders in the same station ({}) as this order: ".format(station_of_order),
              other_orders_of_station)
        sum_min_dist_to_item = dict()
        for order in other_orders_of_station:
            min_distances = []
            for item_in_batch in self.get_all_items_of_batch(batch):
                min_distances.append(min([self.distance_ij[item_in_batch, item]
                                          for item in self.get_items_by_order(order)]))
            sum_min_dist_to_item[order] = np.mean(min_distances)  # average instead? otherwise we always pick small orders
        return min(sum_min_dist_to_item.keys(), key=(lambda k: sum_min_dist_to_item[k]))

    def assign_orders_to_batches(self, pack_station_idx):
        orders_of_station = self.assign_orders_to_stations()[pack_station_idx]
        already_assigned = list()
        orders_of_batch = defaultdict(list)
        items_of_batch = defaultdict(list)
        batch_count = 0
        while len(already_assigned) != len(orders_of_station):
            np.random.seed(14213)
            init_order = np.random.choice(list(set(orders_of_station).difference(already_assigned)))
            orders_of_batch[batch_count].append(init_order)
            already_assigned.append(init_order)
            curr_weight = self.get_total_weight_by_order(init_order)
            forbidden_for_batch = []
            while curr_weight < self.batch_weight and not len(
                    set(already_assigned).union(forbidden_for_batch)) == len(orders_of_station):
                new_order = self.greedy_next_order_to_batch(already_assigned, forbidden_for_batch)
                # new_order = self.count_min_distances_for_order(already_assigned, forbidden_for_batch)
                print("Chosen order: ", new_order)
                if (curr_weight + self.get_total_weight_by_order(new_order)) <= self.batch_weight:
                    print("and it also fits in the current batch ({})".format(batch_count))
                    already_assigned.append(new_order)
                    orders_of_batch[batch_count].append(new_order)
                    curr_weight += self.get_total_weight_by_order(new_order)
                else:
                    print("but it would add too much weight to the batch, go on to next...")
                    forbidden_for_batch.append(new_order)
                    continue
            for order in orders_of_batch[batch_count]:
                items_of_batch[batch_count].extend(self.get_items_by_order(order))
            batch_count += 1
        print("The assignment of orders to batches looks as follows: ", orders_of_batch)
        return items_of_batch

    def apply_greedy_heuristic(self):
        tours_of_pack_station = dict()
        for pack_station in list(self.warehouseInstance.OutputStations.keys()):
            tour_of_batch = dict()
            for batch, items in self.assign_orders_to_batches(pack_station).items():
                items = list(set(items[:]))
                tour = []
                distances_to_pack = {item: self.distance_ij[pack_station, self.item_id_pod_id_dict[item]] for item in items}
                min_item = min(distances_to_pack.keys(), key=(lambda k: distances_to_pack[k]))
                tour.append(min_item)  # TODO in items tauchen doppelte einträge auf. Wie geht man damit um?
                while len(tour) != len(items):
                    distances_to_current_item = {
                        item: self.distance_ij[tour[-1], self.item_id_pod_id_dict[item]]
                        for item in list(set(items).difference(tour))
                    }
                    min_item = min(distances_to_current_item.keys(), key=(lambda k: distances_to_current_item[k]))
                    tour.append(min_item)
                tour_of_batch[batch] = tour
            tours_of_pack_station[pack_station] = tour_of_batch
        return tours_of_pack_station


if __name__ == "__main__":
    greedy = GreedyHeuristic()
    starttime = time.time()
    solution = greedy.apply_greedy_heuristic()
    print("runtime: ", time.time() - starttime)
    print("fitness: ", greedy.get_fitness_of_solution(solution))
    print(solution)
