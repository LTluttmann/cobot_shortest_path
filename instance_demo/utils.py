

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