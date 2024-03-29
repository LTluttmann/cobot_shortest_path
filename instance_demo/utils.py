"""
This script contains container classes for the objects of the solution
"""

import xml.etree.cElementTree as ET
from collections import defaultdict
import re


class Edge:
    def __init__(self, i, j):
        self.StartNode = i
        self.EndNode = j


# container class for a batch
class Batch:
    def __init__(self, ID, pack_station, cobot, item_id_pod_id_dict):
        self.ID = ID
        self.orders = []
        self.items = []
        self.route = []
        self.weight = 0
        self.cobot = cobot
        self.pack_station = pack_station
        self.item_id_pod_id_dict = item_id_pod_id_dict

    def determine_edges(self, item_id_pod_id_dict):
        edges = [self.pack_station]
        for item in self.route:
            edges.append(item_id_pod_id_dict[item])
        edges.append(self.pack_station)
        return edges

    @property
    def edges(self):
        route = self.determine_edges(self.item_id_pod_id_dict)
        edges = []
        for first, second in zip(route, route[1:]):
            if first == second:
                continue
            else:
                edges.append(Edge(first, second))
        return edges


class ItemOfOrder:
    def __init__(self, ID, orig_ID, shelves):
        self.ID = ID
        self.orig_ID = orig_ID
        self.shelves = shelves
        self.shelf = None  # shelf where this item is going to be picked up


class OrderOfBatch:
    def __init__(self, ID, items, weight, item_shelf_mapping, batch_id):
        self.ID = ID
        self.weight = weight
        self.item_counts = defaultdict(int)
        self.items = self.init_items(items, item_shelf_mapping)  # dictionary for all items of an order
        self.batch_id = batch_id
        self.shelves = self.init_shelves()

    def init_items(self, items, item_shelf_mapping):
        item_dict = {}
        for item in items:
            shelves = list(item_shelf_mapping[item].keys())
            item_of_order = ItemOfOrder(self.ID + "_" + item + "_" + str(self.item_counts[item]), item, shelves)
            self.item_counts[item] += 1
            item_dict[item_of_order.ID] = item_of_order
        return item_dict

    def init_shelves(self):
        assert self.items
        shelves = defaultdict(int)
        for item in self.items.values():
            for shelf in item.shelves:
                shelves[shelf] += 1
        return shelves

    @property
    def chosen_shelves(self):
        shelves = defaultdict(int)
        for item in self.items.values():
            if item.shelf:
                shelves[item.shelf] += 1
        return shelves


class BatchNew:
    def __init__(self, ID, pack_station, cobot=None):
        self.ID = ID
        self.cobot = cobot
        self.pack_station = pack_station
        self.orders = {}  # dictionary for all orders of a batch
        self.items = {}
        self.route = []
        self.weight = 0

    @property
    def items_of_shelves(self):
        items_of_shelves = defaultdict(list)
        for item in self.items.values():
            items_of_shelves[item.shelf].append(item)
        return items_of_shelves

    @property
    def shelves(self):
        counts = defaultdict(int)
        for i in [shelf for item in self.items.values() for shelf in item.shelves]:
            counts[i] += 1
        return counts

    def add_order(self, order: OrderOfBatch):
        self.orders[order.ID] = order
        for item in order.items.values():
            self.items[item.ID] = item
            assert self.items[item.ID] is item
        self.weight += order.weight
        order.batch_id = self.ID

    def del_order(self, order: OrderOfBatch, item_id_pod_id_dict):
        self.orders.pop(order.ID)
        for item in order.items.values():

            item_id_pod_id_dict[item.orig_ID][item.shelf] += 1
            self.items.pop(item.ID)

        self.weight -= order.weight
        self.route = [
            x for x in self.route if not x in set(self.route).difference(list(self.items_of_shelves.keys()))
        ]

    @property
    def pack_station(self):
        return self._pack_station

    @pack_station.setter
    def pack_station(self, val):
        # counter = self.ID.split("_")[-1]
        # self.ID = val + "_" + counter
        self._pack_station = val

    @property
    def route(self):
        return self._route

    @route.setter
    def route(self, val: list):
        try:
            edges = [val[0]]
            for first, second in zip(val, val[1:]):
                if first == second:
                    continue
                else:
                    edges.append(second)
            self._route = edges
        except IndexError:
            self._route = val

    def route_insert(self, position, val, item):
        self.route = self.route[:position] + [val] + self.route[position:]
        # item.shelf = val
        assert item is self.items[item.ID]

    @property
    def edges(self):
        edges = []
        route = self.route[:]
        route.insert(0, self.pack_station)
        route.append(self.pack_station)
        for first, second in zip(route, route[1:]):
            if first == second:
                continue
            else:
                edges.append(Edge(first, second))
        return edges


class ItemOfBatch:
    def __init__(self, ID, orig_ID, shelves, weight, pack_station, order):
        self.ID = ID
        self.orig_ID = orig_ID
        self.shelves = shelves
        self.shelf = None  # shelf where this item is going to be picked up
        self.weight = weight
        self.ps = pack_station
        self.order = order
        self.batch = None


class BatchSplit:
    def __init__(self, ID, pack_station, cobot=None):
        self.ID = ID
        self.cobot = cobot
        self.pack_station = pack_station
        self.items = {}
        self.route = []
        self.weight = 0

    @property
    def items_of_shelves(self):
        items_of_shelves = defaultdict(list)
        for item in self.items.values():
            items_of_shelves[item.shelf].append(item)
        return items_of_shelves

    def add_item(self, item: ItemOfBatch):
        if item.ID in self.items.keys():
            raise AssertionError
        self.items[item.ID] = item
        self.weight += item.weight
        item.batch = self.ID

    def del_item(self, item: ItemOfBatch):
        self.items.pop(item.ID)
        if item.shelf not in self.items_of_shelves.keys():
            self.route.remove(item.shelf)
        self.weight -= item.weight

    @property
    def pack_station(self):
        return self._pack_station

    @pack_station.setter
    def pack_station(self, val):
        # counter = self.ID.split("_")[-1]
        # self.ID = val + "_" + counter
        self._pack_station = val

    @property
    def route(self):
        return self._route

    @route.setter
    def route(self, val: list):
        try:
            edges = [val[0]]
            for first, second in zip(val, val[1:]):
                if first == second:
                    continue
                else:
                    edges.append(second)
            self._route = edges
        except IndexError:
            self._route = val

    def route_insert(self, position, val, item):
        self.route = self.route[:position] + [val] + self.route[position:]
        self.items[item.ID].shelf = val

    @property
    def edges(self):
        edges = []
        route = self.route[:]
        route.insert(0, self.pack_station)
        route.append(self.pack_station)
        for first, second in zip(route, route[1:]):
            if first == second:
                continue
            else:
                edges.append(Edge(first, second))
        return edges
