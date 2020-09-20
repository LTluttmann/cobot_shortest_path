"""
This script implements heuristics for the AGV assisted warehouse routing problem under the assumption of a MIXED
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
import heapq
try:
    from Queue import LifoQueue
except ImportError:
    from queue import LifoQueue
import sys
from collections import Counter
# ------------------------------------------- CONFIG -------------------------------------------------------------------
SKU = "24"  # options: 24 and 360
SUBSCRIPT = "_a"
NUM_ORDERS = 10

layoutFile = r'data/layout/1-1-1-2-1.xlayo'
podInfoFile = 'data/sku{}/pods_infos.txt'.format(SKU)

instances = {}
instances[24, 2] = r'data/sku{}/layout_sku_{}_2.xml'.format(SKU, SKU)

storagePolicies = {}
#storagePolicies['dedicated'] = 'data/sku{}/pods_items_dedicated_1.txt'.format(SKU)
storagePolicies['mixed'] = 'data/sku{}/pods_items_mixed_shevels_1-5.txt'.format(SKU)

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
        self.batches = None
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
        """
        trie structure to retrieve item IDs based on their color and letter
        :return: trie: (letter) --> (color) --> (ID)
        """
        trie = defaultdict(dict)
        for key, item in self.warehouseInstance.ItemDescriptions.items():
            trie[item.Letter][item.Color] = item.ID
        return trie

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

    def fill_station_id_bot_id_dict(self):
        """
        This function build a dictionary with pack stations as keys and their respective cobot as value
        """
        for key, bot in self.warehouseInstance.Bots.items():
            self.station_id_bot_id_dict[bot.OutputStation] = key

    # define get functions for better readability and accessibility of data
    def get_items_by_order(self, order_idx):
        """
        In contrary to the dedicated get_items_by_order function this function returns a list containing an item as
        often as it has been ordered instead of only containing the unique items of an order. By doing so, each item
        (even items of the same SKU) are considered as unique and for each item positoin we only need to pick 1 item
        (quantity) of a shelf (and thus only need to check whether a shelf has more than zero items of that SKU.
        :param order_idx: ID of order
        :return: item of specified order
        """
        item_ids = []
        for item_id, item in self.warehouseInstance.Orders[order_idx].Positions.items():
            item_ids.extend([item_id] * int(item.Count))
        return item_ids

    def get_weight_by_item(self, item_idx):
        """
        retireve weight of an item specified by its ID
        :param item_idx: item ID
        :return: item weight
        """
        return self.warehouseInstance.ItemDescriptions[item_idx].Weight

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

    def get_weight_per_batch(self):
        """
        calculates the total weight of all the items already assigned to the batches
        :return: dictionary with batch ID as key and its respective weight as value
        """
        weight_dict = dict()
        for batch_id, batch in self.batches.items():
            weight_dict[batch_id] = batch.weight
        return weight_dict

    def get_assigned_items_of_order(self, order_id):
        return [item for batch in self.batches.keys() for item in self.batches[batch].items.values() if item.order == order_id]

    def get_orders_assigned_to_station(self, station):
        return np.unique([self.get_assigned_items_of_order(order)[0].order for order in self.warehouseInstance.Orders.keys() if self.get_assigned_items_of_order(order)[0].ps == station])

    def get_distance_between_items(self, item1, item2):
        """
        getter for the distance between items. Must first retrieve the shelf the items are stored in
        """
        shelf1 = self.item_id_pod_id_dict[item1]
        shelf2 = self.item_id_pod_id_dict[item2]
        return self.distance_ij[shelf1, shelf2]

    def get_fitness_of_batch(self, batch: BatchNew):
        """
        getter for the distance between items. Must first retrieve the shelf the items are stored in
        """
        distances = []
        distances.append(self.distance_ij[batch.pack_station, batch.route[0]])
        for first, second in zip(batch.route, batch.route[1:]):
            distances.append(self.distance_ij[first, second])
        distances.append(self.distance_ij[batch.pack_station, batch.route[-1]])
        return np.round(np.sum(distances), 1)

    def get_fitness_of_tour(self, batch_route, pack_station):
        """
        just like get_fitness_of_batch, but a route can be passed directly
        """
        distances = []
        distances.append(self.distance_ij[pack_station, batch_route[0]])
        for first, second in zip(batch_route, batch_route[1:]):
            distances.append(self.distance_ij[first, second])
        distances.append(self.distance_ij[pack_station, batch_route[-1]])
        return np.round(np.sum(distances), 1)

    def get_fitness_of_solution(self):
        """
        calculates the fitness of a solution by summing up all the distances of the batches
        """
        distances = []
        for batch in self.batches.values():
            distances.append(self.get_fitness_of_batch(batch))
        return np.round(np.sum(distances), 1)

    def get_batches_for_station(self, pack_station):
        return [batch for batch in self.batches.values() if batch.pack_station == pack_station]

    def write_solution_to_xml(self, path):
        """
        function to write the solution in .xml format
        :param path: path + filename of the xml to be written
        """
        # create the file structure
        root = ET.Element('root')
        collecting = ET.SubElement(root, "Collecting")
        split = ET.SubElement(collecting, "Split")
        pack_stations = ET.SubElement(collecting, "PackStations")
        for pack_station in self.warehouseInstance.OutputStations.keys():
            ps_sum = ET.SubElement(split, 'PackStation')
            ps_sum.set("ID", pack_station)
            ps = ET.SubElement(pack_stations, 'PackStation')
            ps.set("ID", pack_station)
            for batch in self.batches.values():
                if batch.pack_station == pack_station:
                    # set up the batches for summary
                    bt_sum = ET.SubElement(ps_sum, 'Batch')
                    bt_sum.set('ID', batch.ID)
                    # set up batches for detailed solution
                    bt = ET.SubElement(ps, 'Batch')
                    bt.set('BatchNumber', batch.ID)
                    bt.set('Distance', str(self.get_fitness_of_batch(batch)))
                    items_data = ET.SubElement(bt, "ItemsData")
                    Ord = ET.SubElement(items_data, "Orders")
                    for order_id, order in self.warehouseInstance.Orders.items():
                        order_element = ET.SubElement(Ord, "Order")
                        order_element.set("ID", order_id)
                        if order_id in batch.orders:
                            # write orders to batches in summary of solution
                            order_sum = ET.SubElement(bt_sum, "Order")
                            order_sum.text = order_id
                            for item in batch.items.values():
                                if not item.orig_ID in order.Positions:
                                    continue
                                # write items their pod and type to the orders
                                item_element = ET.SubElement(order_element, "Item")
                                item_element.set("ID", item.orig_ID)
                                item_element.set("Pod", item.shelf)
                                item_desc = self.warehouseInstance.ItemDescriptions[item.orig_ID]
                                item_element.set("Type", item_desc.Color + "/" + item_desc.Letter)
                    edges = ET.SubElement(bt, "Edges")
                    for edge in batch.edges:
                        edge_element = ET.SubElement(edges, "Edge")
                        edge_element.set("StartNode", edge.StartNode)
                        edge_element.set("EndNode", edge.EndNode)
        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
        with open(path, "w") as f:
            f.write(xmlstr)


class GreedyMixedShelves(Demo):
    """
    This class implements a greedy heuristic for the AGV assisted warehouse routing problem under the assumption of
    a MIXED storage policy. This class inherits from Demo class to access the functions and attributes defined
    there
    """
    def __init__(self):
        super(GreedyMixedShelves, self).__init__()
        self.item_id_pod_id_dict = defaultdict(dict)
        self.solution = None
        self.orders_of_batches = dict()
        self.batches = dict()
        self.fill_item_id_pod_id_dict()
        self.fill_station_id_bot_id_dict()
        self.warehouseInstance.Orders = {str(key): value for key, value in enumerate(self.warehouseInstance.Orders)}
        self.warehouseInstance.Orders = {key: value for key, value in self.warehouseInstance.Orders.items() if
                                         self.get_total_weight_by_order(key) <= 18}  # exclude  too big orders
        self.item_id_pod_id_dict_orig = copy.deepcopy(self.item_id_pod_id_dict)  # save copies of shelves in order to rollback and test the solution

    def get_station_with_min_total_distance(self, order_idx):
        """
        function to retrieve the station which minimizes the total distance of the items of the
        specified order. The distance of an item to a packing station is measured by the minimum distance shelf storing
        that item
        :param order_idx: index or the order
        """
        items = self.get_items_plus_quant_by_order(order_idx)  # todo refactor with get_items_by_order function
        order_pack_dists = dict()
        for pack_station in list(self.warehouseInstance.OutputStations.keys()):
            item_dists = []
            for item_id, item_quant in items.items():
                shelves_with_item = list(self.item_id_pod_id_dict[item_id].keys())
                shelf_distances = {}
                for shelf in shelves_with_item:
                    shelf_distances[shelf] = self.distance_ij[pack_station, shelf]
                shelf_with_min_dist = min(shelf_distances.keys(), key=(lambda k: shelf_distances[k]))
                self.item_id_pod_id_dict[item_id][shelf_with_min_dist] -= 1
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
            station_of_order = self.get_station_with_min_total_distance(order_id)
            orders_of_stations[station_of_order].append(order_id)
        self.item_id_pod_id_dict = copy.deepcopy(self.item_id_pod_id_dict_orig)
        return orders_of_stations

    def greedy_next_order_to_batch(self, batch: BatchSplit, already_assigned, items_of_station, forbidden=None) -> ItemOfBatch:
        """
        determines which order out of the not assigned orders will be assigned to a given batch. The order which
        minimizes the minimum distance of all items of the order to any item in the batch is chosen.
        Again, we define as distance the distance of the shelf minimizing the distance to an item. The reasoning here
        is, that a order which is stored close to some item that is already in the batch will not add much distance to
        the tour itself.
        :param batch:
        :param already_assigned: list of orders which are already assigned to batches
        :param orders_of_station: list of orders that are assigned to the packing station of the batch
        :param forbidden: orders which have investigated already but would add to much weight to the batch
        :return:
        """
        if forbidden is None:
            forbidden = []
        other_items_of_station = {item_id: item for item_id, item in items_of_station.items() if item_id not in already_assigned+forbidden} # np.setdiff1d(list(items_of_station.keys()), already_assigned+forbidden)
        print("other unassigned orders in the same station ({}) as this batch: ".format(batch.pack_station), other_items_of_station)
        min_dist_to_item = dict()
        for item in other_items_of_station.values():
            min_distances = []
            for item_in_batch_id, item_in_batch in batch.items.items():
                min_dist_shelf = min([self.distance_ij[item_in_batch.shelf, shelf] for shelf in list(self.item_id_pod_id_dict[item.orig_ID].keys())])
                min_distances.append(min_dist_shelf)

            min_dist_to_item[item] = np.min(min_distances)  # average instead? otherwise we always pick small orders

        return min(min_dist_to_item.keys(), key=(lambda k: min_dist_to_item[k]))

    def replenish_shelves(self, batch: BatchSplit):
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
        """
        given a route, this function calculates for a given shelf and a given position at which the shelf is inserted
        in the route how much distance this shelf would add to the tour.
        :param route: current rout
        :param shelf: new node / shelf
        :param i: position to insert the new node at
        :param pack_station: pack station of the batch
        :return:
        """
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

    def greedy_cobot_tour(self, batch: BatchSplit, item=None):
        """
        function to determine a (initial) route in a greedy manner. This is done by looking for the shelf / position
        combination of each item to be inserted in the tour that minimizes the distance and add those shelfes at
        the corresponding positions until all items are included in the tour
        :param batch:
        :param items:
        :return:
        """
        if not item:
            items = copy.deepcopy(batch.items).values()
            batch.route = []  # if all items of a batch shall be scheduled, the current tour needs to be discarded
            self.replenish_shelves(batch)  # replenish shelves
        else:
            items = [item]
        for item in items:
            batch_copy = copy.deepcopy(batch)  # make a copy to not change the current batch if not intended
            shelf_add_distances = {}
            for shelf in item.shelves:
                if self.item_id_pod_id_dict[item.orig_ID][shelf] > 0:  # can only take item if available at that shelf
                    # look for each position of the tour how much distance the new shelf would add
                    added_dists = {i: self.calc_increase_dist(batch_copy.route, shelf, i, batch_copy.pack_station)
                                   for i in range(len(batch.route) + 1)}
                    # determine the position at which the shelf adds the lowest distance
                    min_pos, min_pos_dist = min(added_dists.items(), key=itemgetter(1))
                    shelf_add_distances[shelf, min_pos] = min_pos_dist
                else:
                    pass
                    # print("Shelf {} has not more units of item {}.".format(shelf, item.orig_ID))

            # retrieve the shelf for which the item adds the lowest possible distance to the current tour
            min_shelf, min_pos = min(shelf_add_distances, key=shelf_add_distances.get)
            # insert the shelf at the corresponding position
            batch.route_insert(min_pos, min_shelf, item)
            batch.items[item.ID].shelf = min_shelf
            # remove one unit of the item from the shelf
            self.item_id_pod_id_dict[item.orig_ID][min_shelf] -= 1

    def get_new_batch_id(self):
        """after destroying a batch, new IDs have to be given to new batches. This function looks, if IDs are
        available within a sequence of numbers, or otherwise adds one to the highest ID number"""
        if not self.batches:
            return 0
        curr_batch_ids = [int(batch.ID) for batch in self.batches.values()]
        sequence = np.arange(0, max(curr_batch_ids)+1, 1)
        gaps = np.setdiff1d(sequence, curr_batch_ids)
        if len(gaps) == 0:
            return max(curr_batch_ids) + 1
        else:
            return min(gaps)

    def init_items_of_orders(self, orders, pack_station):
        items = {}
        item_count = 0
        for order in orders:
            for item in self.get_items_by_order(order):
                id = "C{}".format(str(item_count))
                items[id] = ItemOfBatch(id, orig_ID=item, shelves=list(self.item_id_pod_id_dict[item].keys()),
                                         weight=self.get_weight_by_item(item), pack_station=pack_station, order=order)
                item_count += 1
        return items

    def assign_items_to_batches_greedy(self, pack_station):
        """
        function to assign all orders of a given packing station. Goes on until all orders are assigned to a batch.
        For a given batch, orders are assigned to it until no order can be assigned to it without violating the
        capacity restriction. To assign an order to a given batch the function defined above is used
        :param pack_station: ID of the pack station
        :return:
        """
        orders_of_station = self.assign_orders_to_stations()[pack_station]
        already_assigned = list()
        items = self.init_items_of_orders(orders_of_station, pack_station)
        while len(already_assigned) != len(items):
            batch_id = str(self.get_new_batch_id())
            batch = BatchSplit(batch_id, pack_station, self.station_id_bot_id_dict[pack_station])
            print("initialized new batch with ID: ", batch.ID)
            # initialize the batch with a random order
            init_item = items[np.random.choice(np.setdiff1d(list(items.keys()), already_assigned))]
            batch.add_item(init_item)
            self.greedy_cobot_tour(batch, init_item)
            already_assigned.append(init_item.ID)
            forbidden_for_batch = []
            while batch.weight < self.batch_weight and not len(
                    np.union1d(already_assigned, forbidden_for_batch)) == len(items):
                new_item = self.greedy_next_order_to_batch(batch, already_assigned, items, forbidden_for_batch)
                if (batch.weight + new_item.weight) <= self.batch_weight:
                    print("and it also fits in the current batch ({})".format(batch_id))
                    already_assigned.append(new_item.ID)
                    batch.add_item(new_item)
                    self.greedy_cobot_tour(batch, item=new_item)
                else:
                    print("but it would add too much weight to the batch, go on to next...")
                    forbidden_for_batch.append(new_item.ID)
                print("the current batch ({}) looks as follows: {}".format(batch.ID, batch.items))
            self.batches[batch_id] = batch

    def apply_greedy_heuristic(self):
        """
        applies the functions defined above to determine a greedy solution to the problem
        """
        if self.batches:
            self.batches = dict()
        for pack_station in self.warehouseInstance.OutputStations.keys():
            self.assign_items_to_batches_greedy(pack_station)


class SimulatedAnnealingMixed(GreedyMixedShelves):
    def __init__(self):
        super(SimulatedAnnealingMixed, self).__init__()

    def accept(self, curr_batch: BatchNew, candidate_batch: BatchNew, T):
        """
        determines whether a candidate solution is to be accepted or not
        :param curr_batch: batch from current solution which is to be improved
        :param candidate_batch: batch from candidate solution
        :param T: current temperature
        """
        currentFitness = self.get_fitness_of_tour(curr_batch.route, curr_batch.pack_station)
        candidateFitness = self.get_fitness_of_tour(candidate_batch.route, candidate_batch.pack_station)
        if candidateFitness < currentFitness:
            return True
        else:
            if np.random.random() <= self.acceptWithProbability(candidateFitness, currentFitness, T):
                return True

    def acceptWithProbability(self, candidateFitness, currentFitness, T):
        # Accept the new tour for all cases where fitness of candidate => fitness current with a probability
        return math.exp(-abs(candidateFitness - currentFitness) / T)

    def two_opt(self, batch: BatchNew, currentSolution, batch_id, T):
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

    def two_opt_randomized(self, batch: BatchNew, currentSolution, batch_id, T):
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
        tour[i + 1: j] = reverse_order
        if self.accept(curr_batch=currentSolution[batch_id], candidate_batch=batch, T=T):
            currentSolution[batch_id] = batch
            return True
        else:
            batch.route = curr_tour
            return False

    def switch_stations(self, batch: BatchNew):
        """
        vary the pack station of batches
        :return:
        """
        curr_ps = batch.pack_station
        new_ps = np.random.choice(np.setdiff1d(list(self.warehouseInstance.OutputStations.keys()), curr_ps))
        return new_ps

    def simulatedAnnealing(self, alpha=0.975, maxIteration=1000, minTemperature=0.1, mutation_prob=0.2,
                           max_it_without_change=5, batch_id=None):
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
            iteration = 1
            it_without_change = 0
            T = max(self.distance_ij.values()) - min(list(self.distance_ij.values()))
            while T >= minTemperature and iteration < maxIteration and it_without_change < max_it_without_change:
                batch = copy.deepcopy(currentSolution[batch_id])
                if len(batch.route) > 1:
                    made_change = self.two_opt_randomized(batch, currentSolution, batch_id, T)
                    it_without_change += 1 if not made_change else 0
                    T *= alpha
                    iteration += 1
                else:
                    T = 0
            self.batches[batch_id] = currentSolution[batch_id]


class IteratedLocalSearchMixed(SimulatedAnnealingMixed):
    def __init__(self):
        super(IteratedLocalSearchMixed, self).__init__()
        self.simulatedAnnealing()  # get initial solution

    def update_batches_from_new_item(self, new_item, batch_id):
        """
        does all the necessary updating when a order is added to a batch. This includes optimizing the tour through
        greedy and simulated annealing heuristics
        :param new_order:
        :param batch_id:
        :return:
        """
        self.batches[batch_id].add_item(new_item)
        self.greedy_cobot_tour(self.batches[batch_id], new_item)
        self.simulatedAnnealing(batch_id=batch_id, mutation_prob=0.2)

    def perturbation(self, history=None):
        """
        performs the perturbation. The perturbation operation implements a destroy and repair strategy. A random
        batch is destroyed and its orders are distributed among the other batches. If an order does not fit in any
        of the other batches, a new batch is created.
        :param history:
        :return:
        """
        print("Performing perturbation")
        weight_dict = self.get_weight_per_batch()
        destroy_batch = np.random.choice(list(weight_dict.keys()))
        weight_dict.pop(destroy_batch)
        items = self.batches[destroy_batch].items
        self.replenish_shelves(self.batches[destroy_batch])
        self.batches.pop(destroy_batch)
        # sort orders with respect to their weight in descending order (try to assign the biggest orders first)
        weight_of_items = {k: v for k, v in sorted(items.items(), key=lambda item: item[1].weight, reverse=True)}
        for item in weight_of_items.values():
            candidate_batches = {}
            for key, batch in self.batches.items():
                if item.weight + batch.weight <= self.batch_weight and batch.pack_station == item.ps:
                    candidate_batches[key] = item.weight + batch.weight
            if not candidate_batches:
                pack_station = item.ps
                batch_id = str(self.get_new_batch_id())
                self.batches[batch_id] = BatchSplit(batch_id, pack_station, self.station_id_bot_id_dict[pack_station])
                self.update_batches_from_new_item(item, batch_id)
                history = batch_id
            else:
                # choose the batch with maximum workload after assignment
                new_batch_of_item = max(candidate_batches.keys(), key=(lambda k: candidate_batches[k]))
                new_batch_of_item = np.random.choice(list(candidate_batches.keys()))
                new_batch_of_item = min(candidate_batches.keys(), key=(lambda k: candidate_batches[k]))
                self.update_batches_from_new_item(new_item=item, batch_id=new_batch_of_item)
        print("fitness after perturbation: ", self.get_fitness_of_solution())
        return history

    def change_ps_of_order(self):
        print("changes orders")
        # if not ps_to_add_ordr_to:
        order_i = np.random.choice(list(self.warehouseInstance.Orders.keys()))
        first_order_ps = self.get_assigned_items_of_order(order_i)[0].ps

        cand_orders = self.get_orders_assigned_to_station(np.random.choice(np.setdiff1d(list(self.warehouseInstance.OutputStations.keys()), first_order_ps)))
        if not cand_orders:
            cand_orders = np.setdiff1d(list(self.warehouseInstance.Orders.keys()), order_i)
        order_j = np.random.choice(cand_orders)

        items_of_orders = [item for order in [order_i, order_j] for item in self.get_assigned_items_of_order(order)]

        # delete items from current batches
        for item in items_of_orders:
            self.batches[item.batch].del_item(item)
            if len(self.batches[item.batch].items) == 0:
                self.batches.pop(item.batch)
            else:
                self.greedy_cobot_tour(self.batches[item.batch])
                self.simulatedAnnealing(batch_id=item.batch)
                #shelves_of_batch = [item.shelf for item in self.batches[item.batch].items.values()]
                #if item.shelf not in shelves_of_batch:
                #    self.batches[item.batch].route.remove(item.shelf)
            self.item_id_pod_id_dict[item.orig_ID][item.shelf] += 1
        # add items
        just_changed=False
        for item in items_of_orders:
            new_ps = np.random.choice(np.setdiff1d(list(self.warehouseInstance.OutputStations.keys()), item.ps))
            batches = self.get_batches_for_station(new_ps)
            candidate_batches = {}
            for batch in batches:
                if item.weight + batch.weight <= self.batch_weight: ## and batch.pack_station == item.ps:
                    candidate_batches[batch.ID] = item.weight + batch.weight
            if not candidate_batches:
                print("create new batch")
                batch_id = str(self.get_new_batch_id())
                self.batches[batch_id] = BatchSplit(batch_id, new_ps, self.station_id_bot_id_dict[new_ps])
                self.update_batches_from_new_item(item, batch_id)
                just_changed=True
            else:
                # choose the batch with maximum workload after assignment
                new_batch_of_item = max(candidate_batches.keys(), key=(lambda k: candidate_batches[k]))
                self.update_batches_from_new_item(new_item=item, batch_id=new_batch_of_item)

    def replace_item(self, batch: BatchSplit, add_item: ItemOfBatch, remove_item: ItemOfBatch):
        """
        does all the necessary updating when a order is replaced with another. This includes optimizing the tour through
        greedy and simulated annealing heuristics
        """
        batch.del_item(remove_item)
        batch.add_item(add_item)
        self.greedy_cobot_tour(batch)  # optimize tour to accurately validate the change in fitness
        self.simulatedAnnealing(batch_id=batch.ID, mutation_prob=0.2)

    def determine_switchable_items_randomized(self, batch_i, batch_j):
        """
        given two batches (batch ids), this function determines one order of each batch which can be exchanged between
        the batches (wrt to capacity restrictions).
        :param batch_i:
        :param batch_j:
        :return:
        """
        batch_i_items = list(batch_i.items.values())
        batch_j_items = list(batch_j.items.values())
        np.random.shuffle(batch_i_items)
        np.random.shuffle(batch_j_items)
        for item_i in batch_i_items:
            for item_j in batch_j_items:
                weight_diff = item_i.weight - item_j.weight
                if weight_diff < 0:  # if the weight diff is negative item_j is heavier than item_i
                    # in this case we need to check if order_j still fits in batch_i
                    if abs(weight_diff) <= self.batch_weight - batch_i.weight:
                        return item_i, item_j  # if yes, return the orders --> switch will be performed
                elif weight_diff > 0:  # if the weight diff is positive order_i is heavier than order_j
                    # in this case we need to check if order_i still fits in batch_j
                    if weight_diff <= self.batch_weight - batch_j.weight:
                        return item_i, item_j
        # if no items can be exchanged return None
        return None, None

    def randomized_local_search(self, max_iters=15, history=None):
        """
        perform simple randomized swap of orders to batches. This is done for a given number of iterations on
        randomly drawn orders.
        Neighborhood solution is accepted if it results in a better fitness value
        :return:
        """
        iters = 0
        curr_fit = self.get_fitness_of_solution()
        curr_sol = copy.deepcopy(self.batches)
        while iters < max_iters:
            batch_i = np.random.choice([batch for key, batch in self.batches.items()])  #if key != history
            cand_batch_j = [batch for key, batch in self.batches.items() if batch.ID != batch_i.ID and batch.pack_station == batch_i.pack_station]
            if len(cand_batch_j) == 0:
                # self.change_ps_of_order()  # TODO irgenwas schlaues überlegen
                iters += 1
                continue
            else:
                batch_j = np.random.choice(cand_batch_j)
            item_i, item_j = self.determine_switchable_items_randomized(batch_i, batch_j)
            if item_i:
                self.replace_item(batch_i, add_item=item_j, remove_item=item_i)
                self.replace_item(batch_j, add_item=item_i, remove_item=item_j)
                if self.get_fitness_of_solution() < curr_fit or sum([len(b.route) for b in self.batches.values()]) < sum([len(b.route) for b in curr_sol.values()]):
                    curr_sol = copy.deepcopy(self.batches)
                    curr_fit = self.get_fitness_of_solution()
                else:
                    self.batches = copy.deepcopy(curr_sol)
            iters += 1

    def kk(self, number_list):  # Karmarkar-Karp heuristic, adopted and adapted from partition package
        pairs = LifoQueue()
        group1, group2 = [], []
        heap = [(-1 * i.weight, i.ID) for i in number_list]
        heapq.heapify(heap)
        while len(heap) > 1:
            i, labeli = heapq.heappop(heap)
            j, labelj = heapq.heappop(heap)
            pairs.put((labeli, labelj))
            heapq.heappush(heap, (i - j, labeli))
        group1.append(heapq.heappop(heap)[1])

        while not pairs.empty():
            pair = pairs.get()
            if pair[0] in group1:
                group2.append(pair[1])
            elif pair[0] in group2:
                group1.append(pair[1])
        return group1, group2

    def optimized_perturbation(self):
        improvement = True
        curr_sol = copy.deepcopy(self.batches)
        change = False
        while np.ceil(sum([self.get_total_weight_by_order(order) for order in self.warehouseInstance.Orders.keys()]) / 18) < len(self.batches) and improvement:
            change = True
            weight_dict = self.get_weight_per_batch()
            probabilities = []
            keys = []
            for key, weight in weight_dict.items():
                probabilities.append(1 - weight / self.batch_weight)
                keys.append(key)
            probabilities = [float(i) / sum(probabilities) for i in probabilities]
            destroy_batch = np.random.choice(keys, p=probabilities)
            weight_dict.pop(destroy_batch)
            items = self.batches[destroy_batch].items
            # self.replenish_shelves(self.batches[destroy_batch])
            self.batches.pop(destroy_batch)
            # sort orders with respect to their weight in descending order (try to assign the biggest orders first)
            weight_of_items = {k: v for k, v in sorted(items.items(), key=lambda item: item[1].weight, reverse=True)}
            for item in weight_of_items.values():
                candidate_batches = {}
                for key, batch in self.batches.items():
                    if batch.pack_station == item.ps:
                        candidate_batches[key] = item.weight + batch.weight
                if not candidate_batches:
                    pass
                else:
                    new_batch_of_item = min(candidate_batches.keys(), key=(lambda k: candidate_batches[k]))
                    self.update_batches_from_new_item(new_item=item, batch_id=new_batch_of_item)
            if all([batch.weight <= self.batch_weight for batch in self.batches.values()]):
                improvement = True
            else:
                improvement = self.repair()
            if improvement:
                curr_sol = copy.deepcopy(self.batches)
                for batch in self.batches.values():
                    if len(batch.route) == 0:
                        self.greedy_cobot_tour(batch)
                        self.simulatedAnnealing(batch_id=batch.ID)
            else:
                self.batches = copy.deepcopy(curr_sol)
        return bool(improvement * change)

    def repair(self):
        improvement = True

        while not all([batch.weight <= self.batch_weight for batch in self.batches.values()]) and improvement:

            improvement = False
            curr_sol = copy.deepcopy(self.batches)
            inf_batch = next(batch for batch in self.batches.values() if batch.weight > self.batch_weight)
            self.batches.pop(inf_batch.ID)
            for other_batch in [batch for batch in copy.deepcopy(self.batches).values() if batch.pack_station == inf_batch.pack_station]:
                slack = np.abs(inf_batch.weight - other_batch.weight)
                self.batches.pop(other_batch.ID)  # print(set(inf_batch.items.keys()).intersection(other_batch.items.keys()))
                items_of_batches = dict(**copy.deepcopy(inf_batch.items), **copy.deepcopy(other_batch.items))
                batch1_items, batch2_items = self.kk([i for i in items_of_batches.values()])
                batch1_items = [i for i in items_of_batches.values() if i.ID in batch1_items]
                batch2_items = [i for i in items_of_batches.values() if i.ID in batch2_items]

                b1_id = str(self.get_new_batch_id())
                b1 = BatchSplit(b1_id, inf_batch.pack_station)
                [b1.add_item(item) for item in batch1_items]
                self.batches[b1.ID] = b1
                b2_id = str(self.get_new_batch_id())
                b2 = BatchSplit(b2_id, inf_batch.pack_station)
                [b2.add_item(item) for item in batch2_items]
                self.batches[b2.ID] = b2
                if all([batch.weight <= self.batch_weight for batch in self.batches.values()]):
                    return True
                elif np.abs(b1.weight - b2.weight) < slack:
                    improvement = self.repair()
                    return improvement
                else:
                    self.batches = copy.deepcopy(curr_sol)
                    self.batches.pop(inf_batch.ID)
        return improvement

    def local_search_shelves(self):
        for batch in self.batches.values():
            if len(batch.items) / len(batch.route) < 2:
                batch.route = []
                self.replenish_shelves(batch)
                assigned = []
                while len(assigned) != len(batch.items):
                    to_be_assigned = [item for item in batch.items.values() if item.ID not in [i.ID for i in assigned]]
                    shelf_counts = Counter([shelf for item in to_be_assigned for shelf in item.shelves])
                    top_shelf = max(shelf_counts.keys(), key=(lambda k: shelf_counts[k]))
                    items_in_top_shelf = [item for item in to_be_assigned if top_shelf in item.shelves]
                    for item in items_in_top_shelf:
                        if self.item_id_pod_id_dict[item.orig_ID][top_shelf] > 0:
                            item.shelf = top_shelf
                            self.item_id_pod_id_dict[item.orig_ID][top_shelf] -= 1
                            assigned.append(item)
                    batch.route.append(top_shelf)
                self.simulatedAnnealing(batch_id=batch.ID)
            else:
                continue

    def perform_ils(self, num_iters, t_max):
        """
        implements the Iterated Local Search
        """

        iter = 0
        starttime = time.time()
        t = time.time() - starttime
        curr_fit = self.get_fitness_of_solution()
        best_fit = curr_fit
        curr_sol = copy.deepcopy(self.batches)
        best_sol = copy.deepcopy(curr_sol)
        item_id_pod_id_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
        T = self.get_fitness_of_solution() * 0.02
        history = None
        while iter < num_iters and t < t_max:
            #if np.random.random() < 0.2:
            #    self.change_ps_of_order()
            improvement = self.optimized_perturbation()
            if not improvement:
                self.perturbation()
            self.local_search_shelves()
            self.randomized_local_search(15)
            neighbor_fit = self.get_fitness_of_solution()
            print("perturbation: curr fit: {}; cand fit {}".format(curr_fit, neighbor_fit))
            if neighbor_fit < curr_fit or improvement:  # or self.acceptWithProbability(neighbor_fit, curr_fit, T):
                curr_fit = neighbor_fit  # accept new solution
                curr_sol = copy.deepcopy(self.batches)
                item_id_pod_id_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
                if curr_fit < best_fit:
                    best_sol = copy.deepcopy(self.batches)
                    best_fit = self.get_fitness_of_solution()
            else:
                self.batches = copy.deepcopy(curr_sol)  # don´t accept new solution and stick with the current one
                self.item_id_pod_id_dict = copy.deepcopy(item_id_pod_id_dict_copy)
            iter += 1
            T *= 0.8
            t = time.time() - starttime
        self.batches = best_sol


class VariableNeighborhoodSearch(IteratedLocalSearchMixed):
    """
    we need different neighborhoods (x \in N_k, where k = {1,...,k_max} and k_max is typically 2 or 3)
    Idea: Shake1 is perturbation as it is used in ILS. Shake2 is then to destroy 2 batches at once and so on
    """
    def __init__(self):
        super(VariableNeighborhoodSearch, self).__init__()

    def shake(self, k):
        """
        performs the perturbation. The perturbation operation implements a destroy and repair strategy. A random
        batch is destroyed and its orders are distributed among the other batches. If an order does not fit in any
        of the other batches, a new batch is created.
        :param history:
        :return:
        """
        print("Performing perturbation")
        weight_dict = self.get_weight_per_batch()
        destroy_batches = np.random.choice(list(weight_dict.keys()), k, replace=False)
        [weight_dict.pop(destroy_batch) for destroy_batch in destroy_batches]
        # merge dictionaries of orders of the batches that should be destroyed
        items = eval("{" + ",".join(
            ["**self.batches[destroy_batches[{}]].items".format(i) for i in range(len(destroy_batches))]) + "}")
        [self.replenish_shelves(self.batches[destroy_batch]) for destroy_batch in destroy_batches]
        [self.batches.pop(destroy_batch) for destroy_batch in destroy_batches]
        # sort orders with respect to their weight in descending order (try to assign the biggest orders first)
        weight_of_items = {k: v for k, v in sorted(items.items(), key=lambda item: item[1].weight, reverse=True)}

        for item in weight_of_items.values():
            candidate_batches = {}
            for key, batch in self.batches.items():
                if item.weight + batch.weight <= self.batch_weight and batch.pack_station == item.ps:
                    candidate_batches[key] = item.weight + batch.weight
            if not candidate_batches:
                pack_station = item.ps
                batch_id = str(self.get_new_batch_id())
                self.batches[batch_id] = BatchSplit(batch_id, pack_station, self.station_id_bot_id_dict[pack_station])
                self.update_batches_from_new_item(item, batch_id)
                history = batch_id
            else:
                # choose the batch with maximum workload after assignment
                new_batch_of_item = max(candidate_batches.keys(), key=(lambda k: candidate_batches[k]))
                new_batch_of_order = np.random.choice(list(candidate_batches.keys()))
                new_batch_of_item = min(candidate_batches.keys(), key=(lambda k: candidate_batches[k]))
                self.update_batches_from_new_item(new_item=item, batch_id=new_batch_of_item)
        print("fitness after perturbation: ", self.get_fitness_of_solution())

    def reduced_vns(self, max_iters, t_max, k_max):
        """
        :param max_iters:  maximum number of iterations
        :param t_max: maximum cpu time
        :param k_max: maximum number of different neighborhoods / shake operations to be performed
        """
        iters = 0
        starttime = time.time()
        t = time.time() - starttime
        curr_fit = self.get_fitness_of_solution()
        best_fit = curr_fit
        curr_sol = copy.deepcopy(self.batches)
        best_sol = copy.deepcopy(curr_sol)
        T = self.get_fitness_of_solution() * 0.04
        while iters < max_iters and t < t_max:
            print("Start iteration {} of VNS".format(str(iters)))
            k = 1
            if np.random.random() < 0.2:
                self.change_ps_of_order()
            while k < k_max:
                self.shake(k)
                self.randomized_local_search(9)
                neighbor_fit = self.get_fitness_of_solution()
                if neighbor_fit < curr_fit:  # or self.acceptWithProbability(neighbor_fit, curr_fit, T):
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
            T *= 0.9
            t = time.time() - starttime
        self.batches = best_sol



if __name__ == "__main__":
    SKUS = ["24"]  # options: 24 and 360
    SUBSCRIPTS = ["_b"]
    NUM_ORDERSS = [10]  # [10,
    MEANS = ["5"]
    instance_sols = {}
    model_sols = {}
    for SKU in SKUS:
        for SUBSCRIPT in SUBSCRIPTS:
            for NUM_ORDERS in NUM_ORDERSS:
                for MEAN in MEANS:
                    # CAUTION: SCRIPT WONT RUN IF ALL SOLUTIONS ARE WRITTEN AND THIS IS NOT PUT IN COMMENTS
                    #if os.path.isfile('solutions/final/orders_{}_mean_5_sku_{}{}_{}.xml'.format(str(NUM_ORDERS), SKU, SUBSCRIPT, "mixed")):
                    #    continue  # skip iteration if the instance has been solved already
                    #try:
                    layoutFile = r'data/layout/1-1-1-2-1.xlayo'
                    podInfoFile = 'data/sku{}/pods_infos.txt'.format(SKU)
                    instances = {}
                    instances[24, 2] = r'data/sku{}/layout_sku_{}_2.xml'.format(SKU, SKU)

                    storagePolicies = {}
                    #storagePolicies['dedicated'] = 'data/sku{}/pods_items_dedicated_1.txt'.format(SKU)
                    storagePolicies['mixed'] = 'data/sku{}/pods_items_mixed_shevels_1-5.txt'.format(SKU)

                    orders = {}
                    orders['{}_5'.format(str(NUM_ORDERS))] = r'data/sku{}/orders_{}_mean_{}_sku_{}{}.xml'.format(SKU, str(NUM_ORDERS), MEAN, SKU, SUBSCRIPT)
                    sols_and_runtimes = {}
                    runtimes = [0, 4, 8, 13, 20, 30, 40, 50, 60, 80, 100, 120]
                    runtimes = [45]
                    for runtime in runtimes:
                        np.random.seed(523381)
                        if runtime == 0:
                            ils = GreedyMixedShelves()
                            ils.apply_greedy_heuristic()
                        else:
                            vns = VariableNeighborhoodSearch()
                            vns.perform_ils(num_iters=1500, t_max=runtime)
                            #vns.reduced_vns(1500, runtime, 2)
                    print(sols_and_runtimes)
                    instance_sols[(SKU, SUBSCRIPT, NUM_ORDERS)] = sols_and_runtimes
                    model_sols[(SKU, SUBSCRIPT, NUM_ORDERS, MEAN, "ILS")] = vns.get_fitness_of_solution()
                    #model_sols[(SKU, SUBSCRIPT, NUM_ORDERS, "VNS")] = vns.get_fitness_of_solution()


    #with open('../analyse_solution/solutions/mixed360_random_ls_not_random_twoopt.pickle', 'wb') as handle:
    #    pickle.dump(instance_sols, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('../analyse_solution/solutions/mixed_fitness_ils_vns.pickle', 'wb') as handle:
        pickle.dump(model_sols, handle, protocol=pickle.HIGHEST_PROTOCOL)