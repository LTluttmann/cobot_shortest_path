"""
This script implements some tests of the validity of a solution, ie whether all constraints are met
"""

from agv_routing_mixed import GreedyMixedShelves, SimulatedAnnealingMixed, VariableNeighborhoodSearch, IteratedLocalSearchMixed
import numpy as np

class SolutionTester(VariableNeighborhoodSearch):

    def __init__(self):
        super(SolutionTester, self).__init__()

    def count_num_items_taken(self):
        shelves_taken = self.item_id_pod_id_dict
        shelves_orig = self.item_id_pod_id_dict_orig
        total_taken = []
        for key1, shelf in shelves_orig.items():
            for key2, item_count in shelf.items():
                new = shelves_taken[key1][key2]
                orig = item_count
                taken = orig - new
                total_taken.append(taken)
        return sum(total_taken)

    def count_num_requested_items(self):
        """
        test whether the number of items taken from the shelves corresponds with the number of items that were ordered
        :return: Assertion, is either true or throws an error
        """
        req = []
        for order_key, order in self.warehouseInstance.Orders.items():
            items = self.get_items_by_order(order_key)
            req.append(len(items))
        req = sum(req)
        taken = self.count_num_items_taken()
        print(taken, req)
        assert taken == req

    def weight_test(self):
        total_weight_of_orders = np.sum([self.get_total_weight_by_order('{}'.format(i))
                                         for i in range(len(self.warehouseInstance.Orders.keys()))])
        total_weight_of_batches = np.sum([batch.weight for batch in self.batches.values()])
        assert int(total_weight_of_orders) == int(total_weight_of_batches)  # transform to ints to avoid rounding errors

    def capacity_test(self):
        assert all([batch.weight <= self.batch_weight for batch in self.batches.values()])


if __name__ == "__main__":
    st = SolutionTester()
    st.reduced_vns(10, 30, 3)
    # st.simulatedAnnealing()
    #st.perform_ils(30, 50)
    st.count_num_requested_items()
    st.weight_test()
    st.capacity_test()