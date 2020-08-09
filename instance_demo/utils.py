import xml.etree.cElementTree as ET


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