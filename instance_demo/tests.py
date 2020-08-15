from instance_demo import GreedyMixedShelves


class SolutionTester(GreedyMixedShelves):

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


if __name__ == "__main__":
    st = SolutionTester()
    st.apply_greedy_heuristic()
    st.count_num_requested_items()