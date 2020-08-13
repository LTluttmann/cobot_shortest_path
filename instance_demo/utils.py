import xml.etree.cElementTree as ET
from collections import defaultdict

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
    def __init__(self, ID):
        self.ID = ID
        self.shelf = None  # shelf where this item is going to be picked up


class OrderOfBatch:
    def __init__(self, ID, items):
        self.ID = ID
        self.items = self.init_items(items)  # dictionary for all items of an order

    def init_items(self, items):
        item_dict = {}
        for item in items:
            item = ItemOfOrder(item)
            item_dict[item.ID] = item
        return item_dict


class BatchNew:
    def __init__(self, ID, pack_station, cobot=None):
        self.ID = ID
        self.cobot = cobot
        self.pack_station = pack_station
        self.orders = {}  # dictionary for all orders of a batch
        self.items = []
        self.edges = []
        self.weight = 0

    def add_order(self, order: OrderOfBatch):
        self.orders[order.ID] = order
        self.items.extend(list(order.items.keys()))

# Using @property decorator
class Celsius:
    def __init__(self, temperature=0):
        self.temperature = temperature

    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32

    @property
    def temperature(self):
        print("Getting value...")
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        print("Setting value...")
        if value < -273.15:
            raise ValueError("Temperature below -273 is not possible")
        self._temperature = value



class Solution:
    pass


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