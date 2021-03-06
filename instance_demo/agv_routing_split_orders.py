"""
This script implements heuristics for the AGV assisted warehouse routing problem under the assumption of a MIXED
storage policy
"""

# example to use
import numpy as np
import networkx as nx
import itertools
import pickle
import os
import os.path
from os import path
import math
import json
import time
from operator import itemgetter
from xml.dom import minidom
import rafs_instance as instance
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
import pandas as pd
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
orders[f'{str(NUM_ORDERS)}_5'] = f'data/sku{SKU}/orders_{str(NUM_ORDERS)}_mean_5_sku_{SKU}{SUBSCRIPT}.xml'
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

        self.batch_weight = 18.
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

    def get_weight_per_batch(self, pack_station=None):
        """
        calculates the total weight of all the items already assigned to the batches
        :return: dictionary with batch ID as key and its respective weight as value
        """
        weight_dict = dict()
        for batch_id, batch in self.batches.items():
            if pack_station and batch.pack_station == pack_station:
                weight_dict[batch_id] = batch.weight
            elif not pack_station:
                weight_dict[batch_id] = batch.weight
        return weight_dict

    def get_assigned_items_of_order(self, order_id):
        return [item for batch in self.batches.keys() for item in self.batches[batch].items.values() if item.order == order_id]

    def get_orders_assigned_to_station(self, station):
        orders = []
        for order in self.warehouseInstance.Orders.keys():
            if not self.get_assigned_items_of_order(order):
                continue
            elif self.get_assigned_items_of_order(order)[0].ps == station:
                orders.append(order)
        return np.unique(orders)

    def get_distance_between_items(self, item1, item2):
        """
        getter for the distance between items. Must first retrieve the shelf the items are stored in
        """
        shelf1 = self.item_id_pod_id_dict[item1]
        shelf2 = self.item_id_pod_id_dict[item2]
        return self.distance_ij[shelf1, shelf2]

    def get_fitness_of_batch(self, batch: BatchSplit):
        """
        getter for the distance between items. Must first retrieve the shelf the items are stored in
        """
        if len(batch.route) == 0:
            return 0
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
                    bt.set('Weight', str(batch.weight))
                    items_data = ET.SubElement(bt, "ItemsData")
                    for item in [i for batch in self.batches.values() for i in batch.items.values()]:
                        # item_element = ET.SubElement(Ord, "Item")
                        # item_element.set("ID", item.ID)
                        if item.ID in batch.items.keys():
                            # write orders to batches in summary of solution
                            item_sum = ET.SubElement(bt_sum, "Item")
                            item_sum.text = item.ID
                            # write items their pod and type to the orders
                            item_element = ET.SubElement(items_data, "Item")
                            item_element.set("ID", item.ID)
                            item_element.set("Pods_available", item.shelves)
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
        self.item_count = 0
        self.fill_item_id_pod_id_dict()
        self.fill_station_id_bot_id_dict()
        self.warehouseInstance.Orders = {str(key): value for key, value in enumerate(self.warehouseInstance.Orders)}
        #self.warehouseInstance.Orders = {key: value for key, value in self.warehouseInstance.Orders.items() if
        #                                 self.get_total_weight_by_order(key) <= 18}  # exclude  too big orders
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
                item_dists.append(shelf_distances[shelf_with_min_dist])
            order_pack_dists[pack_station] = np.sum(item_dists)
        min_ps, dist = min(order_pack_dists.items(), key=itemgetter(1))
        order_pack_dists.pop(min_ps)
        savings = min(order_pack_dists.values()) - dist
        return min_ps, savings

    def assign_orders_to_stations(self):
        """
        assigns an order to a given station according to the minimum total distance. May lead to unbalanced
        assignments, e.g. that all orders are assigned to only one station.
        :return:
        """
        orders_of_stations = defaultdict(list)
        weight_of_station = defaultdict(int)
        num_ps = len(self.warehouseInstance.OutputStations)

        order_dists = {
            order: self.get_station_with_min_total_distance(order) for order in self.warehouseInstance.Orders.keys()
        }

        order_dists = {k: v[0] for k, v in sorted(order_dists.items(), key=lambda item: item[1], reverse=True)}

        total_weight = sum(
            [self.get_total_weight_by_order(order) for order in self.warehouseInstance.Orders.keys()]
        )
        for order_id, station in order_dists.items():
            if not len(orders_of_stations.get(station, [])) >= .5 * len(self.warehouseInstance.Orders):  #  weight_of_station[station] + self.get_total_weight_by_order(order_id) > total_weight / num_ps:
                orders_of_stations[station].append(order_id)
                weight_of_station[station] += self.get_total_weight_by_order(order_id)
            else:
                alt_station = np.random.choice(np.setdiff1d(list(self.warehouseInstance.OutputStations.keys()), station))
                orders_of_stations[alt_station].append(order_id)
                weight_of_station[alt_station] += self.get_total_weight_by_order(order_id)

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
        :param items_of_station: list of items that are assigned to the packing station of the batch
        :param forbidden: orders which have investigated already but would add to much weight to the batch
        :return:
        """
        if forbidden is None:
            forbidden = []
        other_items_of_station = {
            item_id: item for item_id, item in items_of_station.items() if item_id not in already_assigned+forbidden
        }  # np.setdiff1d(list(items_of_station.keys()), already_assigned+forbidden)
        print("other unassigned orders in the same station ({}) as this batch: ".format(batch.pack_station),
              other_items_of_station)
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
        :param item:
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
        for order in orders:
            for item in self.get_items_by_order(order):
                id = "C{}".format(str(self.item_count))
                items[id] = ItemOfBatch(id, orig_ID=item, shelves=list(self.item_id_pod_id_dict[item].keys()),
                                         weight=self.get_weight_by_item(item), pack_station=pack_station, order=order)
                self.item_count += 1
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
        self.apply_greedy_heuristic()

    def accept(self, curr_batch: BatchSplit, candidate_batch: BatchSplit, T):
        """
        determines whether a candidate solution is to be accepted or not
        :param curr_batch: batch from current solution which is to be improved
        :param candidate_batch: batch from candidate solution
        :param T: current temperature
        :return: Accept->bool; improvement->bool
        """
        currentFitness = self.get_fitness_of_tour(curr_batch.route, curr_batch.pack_station)
        candidateFitness = self.get_fitness_of_tour(candidate_batch.route, candidate_batch.pack_station)
        if candidateFitness < currentFitness:
            return True, True
        else:
            if np.random.random() <= self.acceptWithProbability(candidateFitness, currentFitness, T):
                return True, False
            else:
                return False, False

    def acceptWithProbability(self, candidateFitness, currentFitness, T):
        # Accept the new tour for all cases where fitness of candidate => fitness current with a probability
        return math.exp(-abs(candidateFitness - currentFitness) / T)

    def two_opt(self, batch: BatchSplit, currentSolution, T):
        """
        implements the two opt heuristic. The algorithm iterates over every pair of edges of the tour and interchanges
        them.
        :param batch: the batch of the neighborhood solution
        :param currentSolution: the current solution to be improved
        :param T: current temperature
        """
        curr_tour = batch.route[:]
        curr_tour.insert(0, batch.pack_station)
        for i in range(len(curr_tour) - 2):
            for j in range(i + 2, len(curr_tour)):
                tour = batch.route
                tour.insert(0, batch.pack_station)
                tour[i + 1] = tour[j]
                reverse_order = [curr_tour[m] for m in reversed(range(i + 1, j))]
                tour[i + 2: j + 1] = reverse_order
                tour.remove(batch.pack_station)
                accept = self.accept(curr_batch=currentSolution[batch.ID], candidate_batch=batch, T=T)
                if accept[0]:
                    return accept[0] * accept[1]
                else:
                    old_tour = curr_tour[:]
                    old_tour.remove(batch.pack_station)
                    batch.route = old_tour
                    continue
        return False

    def two_opt_randomized(self, batch: BatchSplit, currentSolution, T):
        """
        randomized version of two-opt in order to speed the local search up. Randomly selects to edges
        """
        curr_tour = batch.route[:]
        tour = batch.route
        length = range(len(tour))
        i, j = random.sample(length, 2)
        tour[i], tour[j] = tour[j], tour[i]
        reverse_order = [tour[m] for m in reversed(range(i + 1, j))]
        tour[i + 1: j] = reverse_order
        accept = self.accept(curr_batch=currentSolution[batch.ID], candidate_batch=batch, T=T)
        if accept[0]:
            return accept[0] * accept[1]
        else:
            batch.route = curr_tour
            return False

    def simulatedAnnealing(self, alpha=0.975, maxIteration=1000, minTemperature=0.1,
                           max_it_without_improvement=7, batch: BatchSplit=None):
        """
        implements a simulated annealing to optimize the tour of a given batch or all batches at once. Neighborhood
        solutions are generated using two opt, either randomized or full.
        :param alpha: cooling parameter
        :param maxIteration: maximum iterations
        :param minTemperature: temperature at which to terminate the algorithm if reached
        :param max_it_without_improvement: maximum number of iterations allowed without changes in the solution
        :param batch: if only a single batch shall be optimized, specify it here
        """
        curr_sol = copy.deepcopy(self.batches)
        batches = self.batches.values() if not batch else [batch]
        for batch in batches:
            iteration = 0
            it_without_improvement = 0
            T = max(self.distance_ij.values()) - min(list(self.distance_ij.values()))
            while T >= minTemperature and iteration < maxIteration and it_without_improvement < max_it_without_improvement:
                if len(batch.route) > 2:
                    improved = self.two_opt_randomized(batch, curr_sol, T)
                    it_without_improvement += 1 if not improved else 0
                    T *= alpha
                    iteration += 1
                else:
                    T = 0  # route cannot be optimized as it consist of only one point


class IteratedLocalSearchMixed(SimulatedAnnealingMixed):

    @staticmethod
    def kk(number_list):  # Karmarkar-Karp heuristic, adopted and adapted from partition package
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

    @staticmethod
    def gini(x):
        mad = np.abs(np.subtract.outer(x, x)).mean()
        rmad = mad/np.mean(x)
        g = 0.5 * rmad
        return g

    def __init__(self):
        super(IteratedLocalSearchMixed, self).__init__()
        self.simulatedAnnealing()  # get initial solution

    def update_batches_from_new_item(self, new_item, batch_id):
        """
        does all the necessary updating when a order is added to a batch. This includes optimizing the tour through
        greedy and simulated annealing heuristics
        :param new_item:
        :param batch_id:
        :return:
        """
        self.batches[batch_id].add_item(new_item)
        self.greedy_cobot_tour(self.batches[batch_id], new_item)
        self.simulatedAnnealing(batch=self.batches[batch_id])

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
                # new_batch_of_item = min(candidate_batches.keys(), key=(lambda k: candidate_batches[k]))
                self.update_batches_from_new_item(new_item=item, batch_id=new_batch_of_item)
        print("fitness after perturbation: ", self.get_fitness_of_solution())
        return history

    def change_ps_of_order(self):
        print("change station of random orders")
        order_i = np.random.choice(list(self.warehouseInstance.Orders.keys()))
        first_order_ps = self.get_assigned_items_of_order(order_i)[0].ps
        cand_orders = self.get_orders_assigned_to_station(
            np.random.choice(np.setdiff1d(list(self.warehouseInstance.OutputStations.keys()), first_order_ps))
        )
        if len(cand_orders) == 0:
            cand_orders = np.setdiff1d(list(self.warehouseInstance.Orders.keys()), order_i)
        order_j = np.random.choice(cand_orders)
        items_of_orders = [
            item for order in [order_i, order_j] for item in self.get_assigned_items_of_order(order)
        ]
        # delete items from current batches
        for item in items_of_orders:
            self.batches[item.batch].del_item(item)
            if len(self.batches[item.batch].items) == 0:
                self.batches.pop(item.batch)
            self.item_id_pod_id_dict[item.orig_ID][item.shelf] += 1
        # add items
        for item in items_of_orders:
            new_ps = np.random.choice(np.setdiff1d(list(self.warehouseInstance.OutputStations.keys()), item.ps))
            item.ps = new_ps
            batches = self.get_batches_for_station(new_ps)
            candidate_batches = {}
            for batch in batches:
                if item.weight + batch.weight <= self.batch_weight and batch.pack_station == item.ps:
                    candidate_batches[batch.ID] = item.weight + batch.weight
            if not candidate_batches:
                print("create new batch")
                batch_id = str(self.get_new_batch_id())
                self.batches[batch_id] = BatchSplit(batch_id, new_ps, self.station_id_bot_id_dict[new_ps])
                self.update_batches_from_new_item(item, batch_id)
            else:
                # choose the batch with minimum workload after assignment
                new_batch_of_item = min(candidate_batches.keys(), key=(lambda k: candidate_batches[k]))
                # new_batch_of_item = np.random.choice(list(candidate_batches.keys()))
                self.update_batches_from_new_item(new_item=item, batch_id=new_batch_of_item)

    def replace_item(self, batch_i: BatchSplit, batch_j: BatchSplit, item_i: ItemOfBatch, item_j: ItemOfBatch):
        """
        does all the necessary updating when a order is replaced with another. This includes optimizing the tour through
        greedy and simulated annealing heuristics
        """
        batch_i.del_item(item_i)
        self.item_id_pod_id_dict[item_i.orig_ID][item_i.shelf] += 1
        batch_j.del_item(item_j)
        self.item_id_pod_id_dict[item_j.orig_ID][item_j.shelf] += 1
        batch_i.add_item(item_j)
        self.greedy_cobot_tour(batch_i, item=item_j)  # optimize tour to accurately validate the change in fitness
        self.simulatedAnnealing(batch=batch_i)

        batch_j.add_item(item_i)
        self.greedy_cobot_tour(batch_j, item=item_i)  # optimize tour to accurately validate the change in fitness
        self.simulatedAnnealing(batch=batch_j)

    def exchange_item(self, batch_add: BatchSplit, batch_del: BatchSplit, item: ItemOfBatch):
        """
        does all the necessary updating when a order is replaced with another. This includes optimizing the tour through
        greedy and simulated annealing heuristics
        """
        batch_del.del_item(item)
        self.item_id_pod_id_dict[item.orig_ID][item.shelf] += 1
        if len(batch_del.items) == 0:
            self.batches.pop(batch_del.ID)
        batch_add.add_item(item)
        self.greedy_cobot_tour(batch_add, item=item)  # optimize tour to accurately validate the change in fitness
        self.simulatedAnnealing(batch=batch_add)
        self.simulatedAnnealing(batch=batch_del)

    def determine_switchable_items_randomized(self, batch_i, batch_j, processed):
        """
        given two batches (batch ids), this function determines one order of each batch which can be exchanged between
        the batches (wrt to capacity restrictions).
        :param batch_i:
        :param batch_j:
        :return:
        """
        #assert len(set([i for batch in self.batches.values() for i in batch.items.keys()])) == 50
        batch_i_items = [i for i in batch_i.items.values() if i.ID not in processed]
        batch_j_items = [j for j in batch_j.items.values() if j.ID not in processed]
        np.random.shuffle(batch_i_items)
        np.random.shuffle(batch_j_items)
        for item_i in batch_i_items:
            for item_j in batch_j_items:
                weight_diff = item_i.weight - item_j.weight
                if (weight_diff <= 0 and abs(weight_diff) <= self.batch_weight - batch_i.weight) or (
                        weight_diff > 0 and weight_diff <= self.batch_weight - batch_j.weight
                ):
                    return item_i, item_j
        # if no items can be exchanged return None
        return None, None

    def randomized_local_search(self, max_iters=25):
        """
        perform simple randomized swap of items to batches. This is done for a given number of iterations on
        randomly drawn orders.
        Neighborhood solution is accepted if it results in a better fitness value
        :return:
        """
        iters = 0
        curr_fit = self.get_fitness_of_solution()
        curr_sol = copy.deepcopy(self.batches)
        item_id_pod_id_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
        processed = []
        while iters < max_iters:
            batch_i = np.random.choice([batch for key, batch in self.batches.items()])
            cand_batch_j = [
                batch for key, batch in self.batches.items() if
                batch.ID != batch_i.ID and batch.pack_station == batch_i.pack_station
            ]
            if len(cand_batch_j) == 0:
                continue
            batch_j = np.random.choice(cand_batch_j)
            item_i, item_j = self.determine_switchable_items_randomized(batch_i, batch_j, processed)
            if item_i:
                processed.extend([item_i.ID, item_j.ID])
                self.replace_item(batch_i, batch_j, item_i, item_j)

                self.local_search_shelves(batch_i)
                self.replace_items_adding_node(batch_i)

                self.local_search_shelves(batch_j)
                self.replace_items_adding_node(batch_j)
                # calc gini
                new_gini = self.gini([len(i) for i in batch_i.items_of_shelves.values()]) + self.gini(
                    [len(j) for j in batch_j.items_of_shelves.values()]
                )
                old_gini = self.gini([len(i) for i in curr_sol[batch_i.ID].items_of_shelves.values()]) + self.gini(
                    [len(j) for j in curr_sol[batch_j.ID].items_of_shelves.values()]
                )

                if self.get_fitness_of_solution() < curr_fit or (
                        self.get_fitness_of_solution() == curr_fit and new_gini > old_gini
                ):
                    curr_sol = copy.deepcopy(self.batches)
                    curr_fit = self.get_fitness_of_solution()
                else:
                    self.batches = copy.deepcopy(curr_sol)
                    self.item_id_pod_id_dict = copy.deepcopy(item_id_pod_id_dict_copy)
            iters += 1

    def determine_switchable_items(
            self, batch_i, batch_j, processed_items, curr_fit, curr_sol, item_id_pod_id_dict_copy
    ):
        """
        given two batches (batch ids), this function determines one order of each batch which can be exchanged between
        the batches (wrt to capacity restrictions).
        :param batch_i:
        :param batch_j:
        :return:
        """
        batch_i_items = [i for i in batch_i.items.values() if i.ID not in processed_items]
        batch_j_items = [j for j in batch_j.items.values() if j.ID not in processed_items]
        for item_i in batch_i_items:
            for item_j in batch_j_items:
                if {item_i.ID, item_j.ID} in processed_items:
                    continue
                processed_items.append({item_i.ID, item_j.ID})
                weight_diff = item_i.weight - item_j.weight
                if (weight_diff <= 0 and abs(weight_diff) <= self.batch_weight - batch_i.weight) or (
                        weight_diff > 0 and weight_diff <= self.batch_weight - batch_j.weight
                ):
                    self.replace_item(batch_i, batch_j, item_i, item_j)

                    del_batch_i = self.replace_items_adding_node(batch_i)
                    self.local_search_shelves(batch_i)

                    del_batch_j = self.replace_items_adding_node(batch_j)
                    self.local_search_shelves(batch_j)
                    # calc gini
                    new_gini = self.gini([len(i) for i in batch_i.items_of_shelves.values()]) + self.gini([len(j) for j in batch_j.items_of_shelves.values()])
                    old_gini = self.gini([len(i) for i in curr_sol[batch_i.ID].items_of_shelves.values()]) + self.gini(
                        [len(j) for j in curr_sol[batch_j.ID].items_of_shelves.values()])

                    if self.get_fitness_of_solution() < curr_fit or (self.get_fitness_of_solution() == curr_fit and new_gini>old_gini):
                        curr_sol = copy.deepcopy(self.batches)
                        curr_fit = self.get_fitness_of_solution()
                        item_id_pod_id_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
                        if del_batch_i or del_batch_j:
                            return
                    else:
                        self.batches = copy.deepcopy(curr_sol)
                        self.item_id_pod_id_dict = copy.deepcopy(item_id_pod_id_dict_copy)

                return self.determine_switchable_items(
                    self.batches[batch_i.ID], self.batches[batch_j.ID], processed_items, curr_fit, curr_sol,
                    item_id_pod_id_dict_copy
                )

    def local_search(self, processed=None):
        """
        perform simple randomized swap of items to batches. This is done for a given number of iterations on
        randomly drawn orders.
        Neighborhood solution is accepted if it results in a better fitness value
        :return:
        """
        processed = [] if not processed else processed
        curr_fit = self.get_fitness_of_solution()
        curr_sol = copy.deepcopy(self.batches)
        item_id_pod_id_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
        memory = []
        for batch_i in self.batches.values():
            for batch_j in [
                batch for batch in self.batches.values() if batch.ID != batch_i.ID and
                                                            batch.pack_station == batch_i.pack_station
            ]:
                if batch_i.ID == batch_j.ID or {batch_i.ID, batch_j.ID} in processed:
                    continue

                self.determine_switchable_items(batch_i, batch_j, memory, curr_fit, curr_sol,item_id_pod_id_dict_copy)
                processed.append({batch_i.ID, batch_j.ID})
                return self.local_search(processed)

    def optimized_perturbation(self):
        print("try to minimize the number of batches")
        assert all([len(batch.route) > 0 for batch in self.batches.values()])
        improvement = True
        change = False
        curr_sol = copy.deepcopy(self.batches)
        item_id_pod_id_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
        while improvement and np.ceil(
                sum([self.get_total_weight_by_order(order) for order in self.warehouseInstance.Orders.keys()]) / 18
        ) < len(self.batches):
            change = True
            num_batches_per_station = {
                pack_station: len(self.get_batches_for_station(pack_station)) for pack_station in self.warehouseInstance.OutputStations
            }
            if any([np.abs(v_1-v_2) > 0 for k_1, v_1 in num_batches_per_station.items() for k_2, v_2 in num_batches_per_station.items() if not k_1==k_2]):
                ps_to_destroy_from = max(num_batches_per_station.keys(), key=( lambda k: num_batches_per_station[k]))
                weight_dict = self.get_weight_per_batch(ps_to_destroy_from)
            else:
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
            self.replenish_shelves(self.batches[destroy_batch])
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
                print(f"found solution with one batch less. Number of batches now is {len(self.batches)}")
                for batch in self.batches.values():
                    if len(batch.route) == 0:
                        self.greedy_cobot_tour(batch)
                        self.simulatedAnnealing(batch=batch)
                curr_sol = copy.deepcopy(self.batches)
                item_id_pod_id_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
                assert all([len(batch.route) > 0 for batch in self.batches.values()])
            else:
                self.batches = copy.deepcopy(curr_sol)
                self.item_id_pod_id_dict = copy.deepcopy(item_id_pod_id_dict_copy)
        return bool(improvement * change)

    def repair(self):
        improvement = True
        while not all([batch.weight <= self.batch_weight for batch in self.batches.values()]) and improvement:
            improvement = False
            curr_sol = copy.deepcopy(self.batches)
            item_id_pod_id_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
            inf_batch = next(batch for batch in self.batches.values() if batch.weight > self.batch_weight)
            for other_batch in [batch for batch in copy.deepcopy(self.batches).values() if batch.pack_station == inf_batch.pack_station and batch.ID != inf_batch.ID]:
                self.batches.pop(inf_batch.ID)
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
                    self.item_id_pod_id_dict = copy.deepcopy(item_id_pod_id_dict_copy)
                    # self.batches.pop(inf_batch.ID)
        return improvement

    def local_search_shelves(self, batch: BatchSplit):
        if len(batch.route) == 0:
            return
        elif len(batch.items) / len(batch.route) < 2 or len(batch.route) >= 2:
            curr_batch = copy.deepcopy(batch)
            batch.route = []
            item_id_pod_it_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
            self.replenish_shelves(batch)
            assigned = []
            while len(assigned) != len(batch.items):
                to_be_assigned = [item for item in batch.items.values() if item.ID not in [i.ID for i in assigned]]
                shelves = [shelf for item in to_be_assigned for shelf in item.shelves if shelf not in batch.route]
                shelf_counts = Counter(shelves)
                rank = pd.DataFrame.from_dict(
                    shelf_counts, orient='index', columns=["num_shelves"]
                ).rank(method="min", ascending=False).merge(
                    pd.DataFrame.from_dict(
                        {
                            shelf: min([self.distance_ij[node, shelf] for node in batch.route+[batch.pack_station]]) for shelf in shelves
                        }, orient="index", columns=["distance"]
                    ).rank(method="min", ascending=True), left_index=True, right_index=True
                )
                top_shelf = rank.assign(mean_rank=rank.mean(axis=1)).mean_rank.idxmin()
                items_in_top_shelf = [item for item in to_be_assigned if top_shelf in item.shelves]
                for item in items_in_top_shelf:
                    if self.item_id_pod_id_dict[item.orig_ID][top_shelf] > 0:
                        item.shelf = top_shelf
                        self.item_id_pod_id_dict[item.orig_ID][top_shelf] -= 1
                        assigned.append(item)
                batch.route.append(top_shelf)
            self.simulatedAnnealing(batch=batch)
            if not self.get_fitness_of_batch(batch) < self.get_fitness_of_batch(curr_batch):
                setattr(batch, "items", curr_batch.items)
                setattr(batch, "route", curr_batch.route)
                assert len(batch.route) > 0
                assert batch is self.batches[batch.ID]
                self.item_id_pod_id_dict = item_id_pod_it_dict_copy

    def check_for_exchangable_item(self, batch, item_to_exchange, candidate_batches: list):
        for batch2 in candidate_batches:
            candidate_items = [len(np.intersect1d(item.shelves, np.setdiff1d(batch.route, item_to_exchange.shelf))) > 0 for item in batch2.items.values()]
            if sum(candidate_items) == 0:
                continue
            candidate_items = list(itertools.compress(batch2.items.values(), candidate_items))
            for item in candidate_items:
                weight_diff = item_to_exchange.weight - item.weight
                if (weight_diff <= 0 and abs(weight_diff) <= self.batch_weight - batch.weight) or (
                        weight_diff > 0 and weight_diff <= self.batch_weight - batch2.weight
                ):
                    self.replace_item(batch_i=batch, batch_j=batch2, item_i=item_to_exchange, item_j=item)
                    return
            exchange_items = []
            rest_kappa = self.batch_weight - batch.weight + item_to_exchange.weight
            miss_kappa = np.abs(self.batch_weight - batch2.weight - item_to_exchange.weight)
            for item in sorted(candidate_items, key=lambda x: x.weight, reverse=False):
                if item.weight <= rest_kappa:
                    exchange_items.append(item)
                    rest_kappa -= item.weight
                    if sum([item_rem.weight for item_rem in exchange_items]) >= miss_kappa:
                        for item2 in exchange_items:
                            self.exchange_item(batch_add=batch, batch_del=batch2, item=item2)
                        self.exchange_item(batch_add=batch2, batch_del=batch, item=item_to_exchange)
                        return
                else:
                    return

    def replace_items_adding_node(self, batch: BatchSplit):
        for items_of_shelf in list(batch.items_of_shelves.values()):
            if len(items_of_shelf) < 4:
                for item_to_replace in items_of_shelf:
                    # candidate batches are all other batches of the same pack station which also include the shelf of
                    # the considered item in their route and at this shelf there are also more then 1 items picked up
                    cand_batches = [
                        cand_batch for cand_batch in self.batches.values() if
                        cand_batch.pack_station == batch.pack_station and not cand_batch.ID == batch.ID
                        and any([x in cand_batch.route for x in item_to_replace.shelves])
                    ]
                    if len(cand_batches) == 0:
                        continue
                    if any([item_to_replace.weight + batch2.weight <= self.batch_weight for batch2 in cand_batches]):
                        batch2 = cand_batches[
                            [item_to_replace.weight + batch2.weight <= self.batch_weight for batch2 in cand_batches].index(True)
                        ]
                        self.exchange_item(batch_add=batch2, batch_del=batch, item=item_to_replace)
                        if not self.batches.get(batch.ID, None):
                            return True
                    else:
                        assert batch is self.batches[batch.ID]
                        self.check_for_exchangable_item(batch, item_to_replace, cand_batches)

    def reduce_number_of_batches(self, max_tries=100):
        print("try to reduce the number of batches")
        tries = 0
        tries2 = 0
        imp = False
        while np.ceil(
                sum([self.get_total_weight_by_order(order) for order in self.warehouseInstance.Orders.keys()]) / 18
        ) < len(self.batches) and tries < max_tries:
            total_weight_per_station = {
                station: sum(
                    [self.get_total_weight_by_order(order) for order in self.get_orders_assigned_to_station(station)]
                ) for station in self.warehouseInstance.OutputStations.keys()
            }
            weight_not_at_optimum = [
                np.ceil(weight / 18) < len(
                    self.get_batches_for_station(station)
                ) for station, weight in total_weight_per_station.items()
            ]
            if any(weight_not_at_optimum) and tries2 < 5:
                self.optimized_perturbation()
                tries2 += 1
            else:
                self.change_ps_of_order()
            tries += 1
        return imp

    def perform_ils(self, t_max):
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
        while t < t_max:
            self.reduce_number_of_batches()
            self.local_search()
            neighbor_fit = self.get_fitness_of_solution()
            print("perturbation: curr fit: {}; cand fit {}".format(curr_fit, neighbor_fit))
            if neighbor_fit < curr_fit:  # or improvement:  # or self.acceptWithProbability(neighbor_fit, curr_fit, T):
                curr_fit = neighbor_fit  # accept new solution
                curr_sol = copy.deepcopy(self.batches)
                item_id_pod_id_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
                if curr_fit < best_fit:
                    best_sol = copy.deepcopy(self.batches)
                    best_fit = self.get_fitness_of_solution()
            else:
                self.batches = copy.deepcopy(curr_sol)  # don´t accept new solution and stick with the current one
                self.item_id_pod_id_dict = copy.deepcopy(item_id_pod_id_dict_copy)
                self.perturbation()
                self.change_ps_of_order()
            iter += 1
            t = time.time() - starttime
        self.batches = best_sol


class VariableNeighborhoodSearch(IteratedLocalSearchMixed):
    """
    we need different neighborhoods (x \in N_k, where k = {1,...,k_max} and k_max is typically 2 or 3)
    Idea: Shake1 is perturbation as it is used in ILS. Shake2 is then to destroy 2 batches at once and so on
    """
    def __init__(self):
        super(VariableNeighborhoodSearch, self).__init__()

    def is_balanced(self):
        ps1 = len(self.get_batches_for_station("OutD0"))
        ps2 = len(self.get_batches_for_station("OutD1"))
        if np.abs(ps1-ps2) > 1:
            return False
        else:
            return True

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
            ["**self.batches[destroy_batches[{}]].items".format(i) for i in range(len(destroy_batches))]
        ) + "}")
        [self.replenish_shelves(self.batches[destroy_batch]) for destroy_batch in destroy_batches]
        [self.batches.pop(destroy_batch) for destroy_batch in destroy_batches]
        # sort orders with respect to their weight in descending order (try to assign the biggest orders first)
        weight_of_items = {
            k: v for k, v in sorted(items.items(), key=lambda item: item[1].weight, reverse=True)
        }
        for item in weight_of_items.values():
            candidate_batches = dict()
            for key, batch in self.batches.items():
                if item.weight + batch.weight <= self.batch_weight and batch.pack_station == item.ps:
                    candidate_batches[key] = item.weight + batch.weight
            if not candidate_batches:
                pack_station = item.ps
                batch_id = str(self.get_new_batch_id())
                self.batches[batch_id] = BatchSplit(batch_id, pack_station, self.station_id_bot_id_dict[pack_station])
                self.update_batches_from_new_item(item, batch_id)
            else:
                # choose the batch with maximum workload after assignment
                new_batch_of_item = np.random.choice(list(candidate_batches.keys()))
                self.update_batches_from_new_item(new_item=item, batch_id=new_batch_of_item)
        print("fitness after perturbation: ", self.get_fitness_of_solution())

    def reduced_vns(self, t_max, k_max):
        """
        :param t_max: maximum cpu time
        :param k_max: maximum number of different neighborhoods / shake operations to be performed
        """
        starttime = time.time()
        t = time.time() - starttime
        curr_fit = self.get_fitness_of_solution()
        best_fit = curr_fit
        curr_sol = copy.deepcopy(self.batches)
        best_sol = copy.deepcopy(curr_sol)
        item_id_pod_id_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
        while t < t_max:
            k = 1
            while k <= k_max:
                self.shake(k)
                if np.random.random() < .99:
                    # todo implement check whether difference
                    self.change_ps_of_order()
                self.reduce_number_of_batches(30)
                print("perform local search")
                for batch in list(self.batches.values()):
                    self.replace_items_adding_node(batch)
                    self.local_search_shelves(batch)
                #self.local_search()
                self.randomized_local_search(30)
                neighbor_fit = self.get_fitness_of_solution()
                print("curr fit: {}; cand fit {}".format(curr_fit, neighbor_fit))
                if neighbor_fit < curr_fit:
                    print("better solution found")
                    curr_fit = neighbor_fit  # accept new solution
                    curr_sol = copy.deepcopy(self.batches)
                    k = 1
                    if curr_fit < best_fit:
                        best_sol = copy.deepcopy(self.batches)
                        best_fit = self.get_fitness_of_solution()
                else:
                    print(f"solution is worse. Performing perturbation with k={k}")
                    self.batches = copy.deepcopy(curr_sol)  # don´t accept new solution and stick with the current one
                    self.item_id_pod_id_dict = copy.deepcopy(item_id_pod_id_dict_copy)
                    k += 1
            t = time.time() - starttime
        print(f"best solution found has a fitness of {best_fit}")
        self.batches = best_sol

if __name__ == "__main__":
    # todo für 1.6 orderline ausführen
    SKUS = ["24"]  # options: 24 and 360
    SUBSCRIPTS = [""]
    NUM_ORDERSS = [10]  # [10,
    MEANS = ["5"]  # possible: 1x6, 5, 10
    instance_sols = {}
    model_sols = {}
    for SKU in SKUS:
        for SUBSCRIPT in SUBSCRIPTS:
            for NUM_ORDERS in NUM_ORDERSS:
                for MEAN in MEANS:
                    print("Now optimizing: SKU={}; Order={}; Subscript={}".format(SKU, NUM_ORDERS, SUBSCRIPT))
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
                    runtimes = [120]
                    for runtime in runtimes:
                        np.random.seed(123523381)
                        if runtime == 0:
                            vns = GreedyMixedShelves()
                            vns.apply_greedy_heuristic()
                        else:
                            vns = VariableNeighborhoodSearch()
                            #vns.perform_ils(t_max=runtime)
                            vns.reduced_vns(t_max=runtime, k_max=2)
                    STORAGE_STRATEGY = "dedicated" if vns.is_storage_dedicated else "mixed"
                    vns.write_solution_to_xml(
                       'solutions/split_orders_{}_mean_{}_sku_{}{}_{}.xml'.format(str(NUM_ORDERS), MEAN, SKU,
                                                                            SUBSCRIPT, STORAGE_STRATEGY)
                    )
                    print(sols_and_runtimes)
                    instance_sols[(SKU, SUBSCRIPT, NUM_ORDERS)] = sols_and_runtimes
                    model_sols[(SKU, SUBSCRIPT, NUM_ORDERS, MEAN, "ILS")] = vns.get_fitness_of_solution()
                    #model_sols[(SKU, SUBSCRIPT, NUM_ORDERS, "VNS")] = vns.get_fitness_of_solution()

    #with open('../analyse_solution/solutions/mixed360_random_ls_not_random_twoopt.pickle', 'wb') as handle:
    #    pickle.dump(instance_sols, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('../analyse_solution/solutions/mixed_fitness_ils_vns.pickle', 'wb') as handle:
        pickle.dump(model_sols, handle, protocol=pickle.HIGHEST_PROTOCOL)
