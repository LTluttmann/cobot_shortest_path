"""
This script contains container classes for the objects of the solution
"""

import xml.etree.cElementTree as ET
from collections import defaultdict
import re

# container class for a batch
class Batch:
    def __init__(self, ID, pack_station, cobot=None):
        self.ID = ID
        self.orders = []
        self.items = []
        self.route = []
        self.weight = 0
        self.cobot = cobot
        self.pack_station = pack_station


class Edge:
    def __init__(self, i, j):
        self.StartNode = i
        self.EndNode = j


class ItemOfOrder:
    def __init__(self, ID, orig_ID, shelves):
        self.ID = ID
        self.orig_ID = orig_ID
        self.shelves = shelves
        self.shelf = None  # shelf where this item is going to be picked up


class OrderOfBatch:
    def __init__(self, ID, items, weight, item_shelf_mapping):
        self.ID = ID
        self.weight = weight
        self.item_counts = defaultdict(int)
        self.items = self.init_items(items, item_shelf_mapping)  # dictionary for all items of an order

    def init_items(self, items, item_shelf_mapping):
        item_dict = {}
        for item in items:
            shelves = list(item_shelf_mapping[item].keys())
            item_of_order = ItemOfOrder(self.ID + "_" + item + "_" + str(self.item_counts[item]), item, shelves)
            self.item_counts[item] += 1
            item_dict[item_of_order.ID] = item_of_order
        return item_dict


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

    def add_order(self, order: OrderOfBatch):
        self.orders[order.ID] = order
        for item in order.items.values():
            self.items[item.ID] = item
        self.weight += order.weight

    def del_order(self, order: OrderOfBatch):
        self.orders.pop(order.ID)
        for item in order.items.values():
            self.items.pop(item.ID)
        self.weight -= order.weight

    @property
    def pack_station(self):
        return self._pack_station

    @pack_station.setter
    def pack_station(self, val):
        counter = self.ID.split("_")[-1]
        self.ID = val + "_" + counter
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


class Solution:
    pass  # todo


if __name__ == "__main__":
    tree = ET.parse("log_example.xml")
    root = tree.getroot()
    for bot in root.iter("Bot"):
        for batch in bot.iter("Batch"):
            if batch.get("ID"):
                for order in bot.iter("Order"):
                    print(bot.get("ID"), batch.get("ID"), order.text)
            elif batch.get("BatchNumber"):
                for edge in batch.iter("Edge"):
                    print(edge.get("EndNode"), edge.get("StartNode"))