from instance_demo import GreedyMixedShelves, SimulatedAnnealingMixed, VariableNeighborhoodSearch
import numpy as np

class SolutionTester(SimulatedAnnealingMixed):

    def __init__(self):
        super(SolutionTester, self).__init__()

    def count_num_items_taken(self):
        shelves_taken = self.item_id_pod_id_dict
        shelves_orig = self.item_id_pod_id_dict_copy
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
        assert taken == req

    def weight_test(self):
        total_weight_of_orders = np.sum([self.get_total_weight_by_order('{}'.format(i))
                                         for i in range(len(self.warehouseInstance.Orders.keys()))])
        total_weight_of_batches = np.sum([batch.weight for batch in self.batches.values()])
        assert total_weight_of_orders == total_weight_of_batches

    def capacity_test(self):
        assert all([batch.weight <= self.batch_weight for batch in self.batches.values()])


if __name__ == "__main__":
    st = SolutionTester()
    # st.reduced_vns(130, 50, 3)
    st.simulatedAnnealing()
    st.count_num_requested_items()
    st.weight_test()
    st.capacity_test()