"""
This script implements heuristics for the AGV assisted warehouse routing problem under the assumption of a MIXED
storage policy
"""

# example to use
import numpy as np
import pickle
import math
import time
from operator import itemgetter
from utils import *
import copy
import random
import heapq
from operator import attrgetter
try:
    from Queue import LifoQueue
except ImportError:
    from queue import LifoQueue
from collections import Counter
import pandas as pd
import itertools

from prepare_data import Demo


class GreedyMixedShelves(Demo):
    """
    This class implements a greedy heuristic for the AGV assisted warehouse routing problem under the assumption of
    a MIXED storage policy. This class inherits from Demo class to access the functions and attributes defined
    there
    """

    def __init__(self, input_files):
        super(GreedyMixedShelves, self).__init__(*input_files)
        self.item_id_pod_id_dict = defaultdict(dict)
        self.solution = None
        self.orders_of_batches = dict()
        self.batches = dict()
        self.fill_item_id_pod_id_dict()
        self.fill_station_id_bot_id_dict()
        self.warehouseInstance.Orders = {
            str(key): value for key, value in enumerate(self.warehouseInstance.Orders)
        }
        # exclude too big orders
        self.warehouseInstance.Orders = {
            key: value
            for key, value in self.warehouseInstance.Orders.items()
            if self.get_total_weight_by_order(key) <= 18
        }
        self.item_id_pod_id_dict_orig = copy.deepcopy(
            self.item_id_pod_id_dict
        )  # save copies of shelves in order to rollback and test the solution
        self.total_weight = sum(
            [
                self.get_total_weight_by_order(order)
                for order in self.warehouseInstance.Orders.keys()
            ]
        )

    def tests(self):
        assert all(
            [
                batch_i.items[item] is batch_i.orders[item.split("_")[0]].items[item]
                for batch_i in self.batches.values()
                for item in batch_i.items.keys()
            ]
        )
        assert all(
            [
                all([x in batch.route for x in batch.items_of_shelves.keys()])
                for batch in self.batches.values()
            ]
        )
        assert all([len(batch.route) > 0 for batch in self.batches.values()])
        assert all([
            abs(len(self.get_batches_for_station(ps1)) - len(self.get_batches_for_station(ps2))) < 2
            for ps1, ps2 in list(itertools.combinations(self.warehouseInstance.OutputStations.keys(), 2))
        ])
        assert set(self.warehouseInstance.Orders.keys()).symmetric_difference(set([x for batch in self.batches.values() for x in batch.orders])) == set()
        # test whether the stock is correct
        item_id_pod_id_dict_orig_copy = copy.deepcopy(self.item_id_pod_id_dict_orig)
        for batch in self.batches.values():
            for item in batch.items.values():
                item_id_pod_id_dict_orig_copy[item.orig_ID][item.shelf] -= 1
        try:
            assert item_id_pod_id_dict_orig_copy == self.item_id_pod_id_dict
        except AssertionError as e:
            for ik, item in self.item_id_pod_id_dict.items():
                for sk, shelf in item.items():
                    if shelf != item_id_pod_id_dict_orig_copy[ik][sk]:
                        print(ik, sk)
            raise e

        assert all([stock >= 0 for item in self.item_id_pod_id_dict.values() for stock in item.values()])

    def get_station_with_min_total_distance(self, order_idx):
        """
        function to retrieve the station which minimizes the total distance of the items of the
        specified order. The distance of an item to a packing station is measured by the minimum distance shelf storing
        that item
        :param order_idx: index or the order
        """
        items = self.get_items_plus_quant_by_order(
            order_idx
        )  # todo refactor with get_items_by_order function
        order_pack_dists = dict()
        for pack_station in list(self.warehouseInstance.OutputStations.keys()):
            item_dists = []
            for item_id, item_quant in items.items():
                shelves_with_item = list(self.item_id_pod_id_dict[item_id].keys())
                shelf_distances = {}
                for shelf in shelves_with_item:
                    shelf_distances[shelf] = self.distance_ij[pack_station, shelf]
                item_dists.extend([min(shelf_distances.values())] * item_quant)
            order_pack_dists[pack_station] = np.sum(item_dists)
        ordered_list = sorted(
            order_pack_dists.items(), key=lambda item: item[1], reverse=False
        )
        savings = [
            ordered_list[i + 1][1] - ordered_list[i][1]
            for i in range(len(ordered_list) - 1)
        ] + [-float("inf")]
        return {k[0]: savings[i] for i, k in enumerate(ordered_list)}

    def do_station_assignment(self, order):
        station_dict = self.get_station_with_min_total_distance(order)
        num_ps = len(self.warehouseInstance.OutputStations)
        station_dict_copy = station_dict.copy()
        while True:
            try:
                best_station = next(iter(station_dict))
            except StopIteration:
                best_station = next(iter(station_dict_copy))
                return best_station
            weight_of_station = sum(
                [
                    self.get_total_weight_by_order(order)
                    for order in self.get_orders_assigned_to_station(best_station)
                ]
            )
            if not np.ceil(
                (weight_of_station + self.get_total_weight_by_order(order))
                / self.batch_weight
            ) > np.ceil(np.ceil(self.total_weight / self.batch_weight) / num_ps):
                return best_station
            else:
                station_dict.pop(best_station)

    def assign_orders_to_stations(self):
        """
        assigns an order to a given station according to the minimum total distance. May lead to unbalanced
        assignments, e.g. that all orders are assigned to only one station.
        :return:
        """

        def do_assignment(order_dists, orders_of_stations=None, weight_of_station=None):

            orders_of_stations = (
                defaultdict(list) if not orders_of_stations else orders_of_stations
            )
            weight_of_station = (
                defaultdict(int) if not weight_of_station else weight_of_station
            )
            num_ps = len(self.warehouseInstance.OutputStations)

            order_dists = {
                k: v
                for k, v in sorted(
                    order_dists.items(),
                    key=lambda item: list(item[1].values())[0],
                    reverse=True,
                )
            }

            dict_copy = order_dists.copy()
            for order, distances in order_dists.items():
                for station in distances.keys():
                    if not np.ceil(
                        (
                            weight_of_station[station]
                            + self.get_total_weight_by_order(order)
                        )
                        / self.batch_weight
                    ) > np.ceil(
                        np.ceil(self.total_weight / self.batch_weight) / num_ps
                    ):
                        orders_of_stations[station].append(order)
                        weight_of_station[station] += self.get_total_weight_by_order(
                            order
                        )
                        dict_copy.pop(order)
                        break
                    elif (
                        len(dict_copy[order]) == 1
                    ):  # fits in no ps under optimal assumptions
                        lighter_station = min(
                            weight_of_station.keys(),
                            key=(lambda k: weight_of_station[k]),
                        )
                        orders_of_stations[lighter_station].append(order)
                        weight_of_station[
                            lighter_station
                        ] += self.get_total_weight_by_order(order)
                        dict_copy.pop(order)
                        break
                    else:
                        dict_copy[order].pop(station)
                        orders_of_stations = do_assignment(
                            dict_copy, orders_of_stations, weight_of_station
                        )
                        return orders_of_stations
            return orders_of_stations

        order_dists = {
            order: self.get_station_with_min_total_distance(order)
            for order in self.warehouseInstance.Orders.keys()
        }

        orders_of_stations = do_assignment(order_dists)

        return orders_of_stations

    def greedy_next_order_to_batch(
        self, batch: BatchNew, already_assigned, orders_of_station, forbidden=None
    ):
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
        other_orders_of_station = np.setdiff1d(
            orders_of_station, already_assigned + forbidden
        )
        print(
            "other unassigned orders in the same station ({}) as this batch: ".format(
                batch.pack_station
            ),
            other_orders_of_station,
        )
        sum_min_dist_to_item = dict()
        for order in other_orders_of_station:
            min_distances = []
            for item_in_batch_id, item_in_batch in batch.items.items():
                dist_per_item_to_curr_item = []
                for item in self.get_items_by_order(order):
                    min_dist_shelf = min(
                        [
                            self.distance_ij[item_in_batch.shelf, shelf]
                            for shelf in list(self.item_id_pod_id_dict[item].keys())
                        ]
                    )
                    dist_per_item_to_curr_item.append(min_dist_shelf)
                min_distances.append(min(dist_per_item_to_curr_item))

            sum_min_dist_to_item[order] = np.sum(
                min_distances
            )  # average instead? otherwise we always pick small orders

        return min(sum_min_dist_to_item.keys(), key=(lambda k: sum_min_dist_to_item[k]))

    def replenish_shelves(self, batch: BatchNew):
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
            return (
                self.distance_ij[pack_station, shelf]
                + self.distance_ij[shelf, pack_station]
            )
        # we favor solutions where the same shelve is visited all at once and not at different parts of the route,
        # thus give it a negative penalty here
        try:
            if route[i] == shelf:
                return -0.1
        except IndexError:
            if route[i - 1] == shelf:
                return -0.1
        # if the shelf at the current position of the tour is not equal to the candidate shelf, the distance added to
        # the tour by candidate shelf j at position i is calculated as follows: d_{i-1, j} + d{j, i} - d{i-1, i}
        # (pack stations are added manually as they are not "part" of route object)
        if i == 0:
            add_dist = (
                self.distance_ij[pack_station, shelf]
                + self.distance_ij[shelf, route[i]]
            )
            subtr_dist = self.distance_ij[pack_station, route[i]]
        elif i == len(route):
            add_dist = (
                self.distance_ij[route[-1], shelf]
                + self.distance_ij[shelf, pack_station]
            )
            subtr_dist = self.distance_ij[route[-1], pack_station]
        else:
            add_dist = (
                self.distance_ij[route[i - 1], shelf]
                + self.distance_ij[shelf, route[i]]
            )
            subtr_dist = self.distance_ij[route[i - 1], route[i]]
        return add_dist - subtr_dist

    def greedy_cobot_tour(self, batch: BatchNew, items=None):
        """
        function to determine a (initial) route in a greedy manner. This is done by looking for the shelf / position
        combination of each item to be inserted in the tour that minimizes the distance and add those shelfes at
        the corresponding positions until all items are included in the tour
        :param batch:
        :param items:
        :return:
        """
        if not items:
            items = batch.items  # copy.deepcopy(batch.items)
            batch.route = (
                []
            )  # if all items of a batch shall be scheduled, the current tour needs to be discarded
            self.replenish_shelves(batch)  # replenish shelves
        else:
            items = items  # copy.deepcopy(items)

        for item in items.values():  # iterate over all items in the batch
            batch_copy = copy.deepcopy(
                batch
            )  # make a copy to not change the current batch if not intended
            shelf_add_distances = {}
            for shelf in item.shelves:
                if (
                    self.item_id_pod_id_dict[item.orig_ID][shelf] > -1000
                ):  # can only take item if available at that shelf
                    # look for each position of the tour how much distance the new shelf would add
                    added_dists = {
                        i: self.calc_increase_dist(
                            batch_copy.route, shelf, i, batch_copy.pack_station
                        )
                        for i in range(len(batch.route) + 1)
                    }
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
            item.shelf = min_shelf
            assert item is batch.items[item.ID]
            # remove one unit of the item from the shelf
            self.item_id_pod_id_dict[item.orig_ID][min_shelf] -= 1

    def get_new_batch_id(self):
        """after destroying a batch, new IDs have to be given to new batches. This function looks, if IDs are
        available within a sequence of numbers, or otherwise adds one to the highest ID number"""
        if not self.batches:
            return 0
        curr_batch_ids = [int(batch.ID) for batch in self.batches.values()]
        sequence = np.arange(0, max(curr_batch_ids) + 1, 1)
        gaps = np.setdiff1d(sequence, curr_batch_ids)
        if len(gaps) == 0:
            return max(curr_batch_ids) + 1
        else:
            return min(gaps)

    def assign_orders_to_batches_greedy(self, pack_station):
        """
        function to assign all orders of a given packing station. Goes on until all orders are assigned to a batch.
        For a given batch, orders are assigned to it until no order can be assigned to it without violating the
        capacity restriction. To assign an order to a given batch the function defined above is used
        :param pack_station: ID of the pack station
        :return:
        """
        orders_of_station = self.assign_orders_to_stations()[pack_station]
        already_assigned = list()
        while len(already_assigned) != len(orders_of_station):
            batch_id = str(self.get_new_batch_id())
            batch = BatchNew(
                batch_id, pack_station, self.station_id_bot_id_dict[pack_station]
            )
            print("initialized new batch with ID: ", batch.ID)
            # initialize the batch with a random order
            init_order = np.random.choice(
                np.setdiff1d(orders_of_station, already_assigned)
            )
            items_of_order = self.get_items_by_order(init_order)
            weight_of_order = self.get_total_weight_by_order(init_order)
            init_order = OrderOfBatch(
                init_order,
                items_of_order,
                weight_of_order,
                self.item_id_pod_id_dict,
                batch.ID,
            )
            batch.add_order(init_order)
            self.greedy_cobot_tour(batch)
            already_assigned.append(init_order.ID)
            forbidden_for_batch = []
            while batch.weight < self.batch_weight and not len(
                np.union1d(already_assigned, forbidden_for_batch)
            ) == len(orders_of_station):
                new_order = self.greedy_next_order_to_batch(
                    batch, already_assigned, orders_of_station, forbidden_for_batch
                )
                weight_of_order = self.get_total_weight_by_order(new_order)
                print("Chosen order: ", new_order)
                if (batch.weight + weight_of_order) <= self.batch_weight:
                    print("and it also fits in the current batch ({})".format(batch_id))
                    already_assigned.append(new_order)
                    items_of_order = self.get_items_by_order(new_order)
                    weight_of_order = self.get_total_weight_by_order(new_order)
                    new_order = OrderOfBatch(
                        new_order,
                        items_of_order,
                        weight_of_order,
                        self.item_id_pod_id_dict,
                        batch.ID,
                    )
                    batch.add_order(new_order)
                    self.greedy_cobot_tour(batch, items=new_order.items)
                else:
                    print(
                        "but it would add too much weight to the batch, go on to next..."
                    )
                    forbidden_for_batch.append(new_order)
                print(
                    "the current batch ({}) looks as follows: {}".format(
                        batch.ID, batch.orders
                    )
                )
            self.batches[batch_id] = batch

    def apply_greedy_heuristic(self):
        """
        applies the functions defined above to determine a greedy solution to the problem
        """
        if self.batches:
            self.batches = dict()
        for pack_station in self.warehouseInstance.OutputStations.keys():
            self.assign_orders_to_batches_greedy(pack_station)


class SimulatedAnnealingMixed(GreedyMixedShelves):
    def __init__(self, input_files):
        super(SimulatedAnnealingMixed, self).__init__(input_files)
        self.apply_greedy_heuristic()

    def accept(self, curr_fitness: BatchNew, candidate_batch: BatchNew, T):
        """
        determines whether a candidate solution is to be accepted or not
        :param curr_batch: batch from current solution which is to be improved
        :param candidate_batch: batch from candidate solution
        :param T: current temperature
        :return: Accept->bool; improvement->bool
        """
        candidate_fitness = self.get_fitness_of_tour(
            candidate_batch.route, candidate_batch.pack_station
        )
        if candidate_fitness < curr_fitness:
            return True, True
        else:
            if self.acceptWithProbability(
                candidate_fitness, curr_fitness, T
            ):
                return True, False
            else:
                return False, False

    def acceptWithProbability(self, candidateFitness, currentFitness, T):
        # Accept the new tour for all cases where fitness of candidate => fitness current with a probability
        return np.random.random() <= np.exp(-abs(candidateFitness - currentFitness) / T)

    def two_opt(self, batch: BatchNew, curr_fitness, T):
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
                tour[i + 2 : j + 1] = reverse_order
                tour.remove(batch.pack_station)
                accept, improve = self.accept(
                    curr_fitness=curr_fitness, candidate_batch=batch, T=T
                )
                if accept:
                    return accept, improve
                else:
                    old_tour = curr_tour[:]
                    old_tour.remove(batch.pack_station)
                    batch.route = old_tour
                    continue
        return False, False

    def two_opt_randomized(self, batch: BatchNew, curr_fitness, T):
        """
        randomized version of two-opt in order to speed the local search up. Randomly selects to edges
        """
        curr_tour = batch.route[:]
        tour = batch.route
        length = range(len(tour))
        i, j = random.sample(length, 2)
        tour[i], tour[j] = tour[j], tour[i]
        reverse_order = [tour[m] for m in reversed(range(i + 1, j))]
        tour[i + 1 : j] = reverse_order
        accept, improve = self.accept(
            curr_fitness=curr_fitness, candidate_batch=batch, T=T
        )
        if accept:
            return accept, improve
        else:
            batch.route = curr_tour
            return False, False

    def swap(self, batch: BatchNew, curr_fitness, T):
        curr_tour = batch.route[:]
        pairs = list(itertools.combinations(range(len(curr_tour)), 2))
        pairs = np.random.permutation(pairs)
        for idx1, idx2 in pairs:
            # idx1, idx2 = np.random.choice(range(len(curr_tour)), 2, replace=False)
            batch.route[idx1], batch.route[idx2] = batch.route[idx2], batch.route[idx1]
            accept, improve = self.accept(
                curr_fitness=curr_fitness, candidate_batch=batch, T=T
            )
            if accept:
                return accept, improve
            else:
                batch.route = curr_tour[:]
                continue
        return False, False

    def relocate(self, batch: BatchNew, curr_fitness, T):
        curr_tour = batch.route[:]
        for idx1 in range(len(curr_tour)):
            for idx2 in set(range(len(curr_tour))).difference([1]):
                batch.route.insert(idx2, batch.route.pop(idx1))
                accept, improve = self.accept(
                    curr_fitness=curr_fitness, candidate_batch=batch, T=T
                )
                if accept:
                    return accept, improve
                else:
                    batch.route = curr_tour[:]
                    continue
        return False, False

    def switch_stations(self, batch: BatchNew):
        """
        vary the pack station of batches
        :return:
        """
        curr_ps = batch.pack_station
        new_ps = np.random.choice(
            np.setdiff1d(list(self.warehouseInstance.OutputStations.keys()), curr_ps)
        )
        return new_ps

    def get_weight_update(self, accepted, improved, best_fitness):
        omega1, omega2, omega3, omega4 = 5, 3, 0, 0
        if self.get_fitness_of_solution() < best_fitness:
            return omega1
        elif improved:
            return omega2
        elif accepted:
            return omega3
        else:
            return omega4

    def simulatedAnnealing(
        self,
        alpha=0.95,
        maxIteration=1000,
        minTemperature=0.1,
        max_it_without_improvement=4,
        batch: BatchNew = None,
        lambda_param=0.75,
        init_weights=1,
    ):
        """
        implements a simulated annealing to optimize the tour of a given batch or all batches at once. Neighborhood
        solutions are generated using two opt, either randomized or full.
        :param alpha: cooling parameter
        :param maxIteration: maximum iterations
        :param minTemperature: temperature at which to terminate the algorithm if reached
        :param max_it_without_improvement: maximum number of iterations allowed without changes in the solution
        :param batch: if only a single batch shall be optimized, specify it here
        :param lambda_param: weight for previous operator weight in update step
        :param init_weights: initial weights of all operators
        """

        batches = self.batches.values() if not batch else [batch]
        operations = [self.two_opt, self.swap, self.relocate]

        weights = [init_weights] * len(operations)

        for batch in batches:

            best_batch = copy.deepcopy(batch)
            best_fit = self.get_fitness_of_batch(best_batch)
            curr_fit = self.get_fitness_of_batch(batch)

            iteration = 0
            it_without_improvement = 0

            T = max(self.distance_ij.values()) - min(list(self.distance_ij.values()))
            while (
                T >= minTemperature
                and iteration < maxIteration
                and it_without_improvement < max_it_without_improvement
            ):
                probs = [x / sum(weights) for x in weights]
                operation = np.random.choice(operations, p=probs)
                if len(batch.route) > 2:
                    # improved = self.two_opt_randomized(batch, curr_sol, T)
                    accepted, improved = operation(batch, curr_fit, T)
                    weight_update = self.get_weight_update(accepted, improved, best_fit)
                    weights[operations.index(operation)] = (
                        lambda_param * weights[operations.index(operation)]
                        + (1 - lambda_param) * weight_update
                    )
                    if accepted:
                        curr_fit = self.get_fitness_of_batch(batch)
                        if curr_fit < best_fit:
                            best_fit = curr_fit
                            best_batch = copy.deepcopy(batch)
                    it_without_improvement += 1 if not improved else 0
                    T *= alpha
                    iteration += 1
                else:
                    T = 0  # route cannot be optimized as it consist of only one point
            batch.__dict__ = best_batch.__dict__.copy()


class VariableNeighborhoodSearch(SimulatedAnnealingMixed):
    @staticmethod
    def kk(
        number_list,
    ):  # Karmarkar-Karp heuristic, adopted and adapted from partition package
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

    def __init__(self, input_files):
        super(VariableNeighborhoodSearch, self).__init__(input_files)
        self.simulatedAnnealing()  # get initial solution

    def update_batches_from_new_order(self, new_order, batch_id):
        """
        does all the necessary updating when a order is added to a batch. This includes optimizing the tour through
        greedy and simulated annealing heuristics
        :param new_order:
        :param batch_id:
        :return:
        """
        update_batch = self.batches[batch_id]
        items_of_order = self.get_items_by_order(new_order)
        weight_of_order = self.get_total_weight_by_order(new_order)
        new_order = OrderOfBatch(
            new_order,
            items_of_order,
            weight_of_order,
            self.item_id_pod_id_dict,
            update_batch.ID,
        )
        update_batch.add_order(new_order)
        self.greedy_cobot_tour(update_batch, new_order.items)
        # self.simulatedAnnealing(batch=self.batches[batch_id]) # takes too much time --> perform only at the end

    def shake(self, k):
        batches_to_destroy = k if k <= len(self.batches) else len(self.batches)
        print(f"Performing perturbation with k={k}")
        orders = {}
        for i in range(batches_to_destroy):
            destroy_batch = self.choose_batch_for_destruction()
            orders = {**orders, **self.batches[destroy_batch].orders}
            self.replenish_shelves(self.batches[destroy_batch])
            self.batches.pop(destroy_batch)

        weight_dict = self.get_weight_per_batch()
        # sort orders with respect to their weight in descending order (try to assign the biggest orders first)
        weight_of_orders = {
            key: v.weight
            for key, v in sorted(
                orders.items(), key=lambda item: item[1].weight, reverse=True
            )
        }
        for order, weight_of_order in weight_of_orders.items():
            candidate_batches = {}
            for key, weight in weight_dict.items():
                if weight_of_order + weight <= self.batch_weight:
                    candidate_batches[key] = weight_of_order + weight
            if not candidate_batches:
                pack_station = self.do_station_assignment(order)
                batch_id = str(self.get_new_batch_id())
                self.batches[batch_id] = BatchNew(
                    batch_id, pack_station, self.station_id_bot_id_dict[pack_station]
                )
                self.update_batches_from_new_order(order, batch_id)
            else:
                # choose the batch with maximum workload after assignment
                new_batch_of_order = np.random.choice(list(candidate_batches.keys()))
                #  new_batch_of_order = max(candidate_batches.keys(), key=(lambda k: candidate_batches[k]))
                self.update_batches_from_new_order(
                    new_order=order, batch_id=new_batch_of_order
                )
            # update weight_dict
            weight_dict = self.get_weight_per_batch()
        print("fitness after perturbation: ", self.get_fitness_of_solution())

    def exchange_orders(self, k, processed):
        card = k

        all_orders = {
            order.ID: order for batch in self.batches.values() for order in list(batch.orders.values())
        }

        combs = list(itertools.combinations(
            [order.ID for order in all_orders.values()], 2
        ))
        combs_feas = []
        for comb in combs:
            order_i, order_j = itemgetter(*comb)(all_orders)
            # exclude combinations that come from same batch -> exchanging orders within a batch makes no sense
            if order_i.batch_id == order_j.batch_id:
                continue
            # check weight constraints
            weight_diff = order_i.weight - order_j.weight
            if (
                    weight_diff <= 0 and abs(weight_diff) <= self.batch_weight - self.batches[order_i.batch_id].weight
            ) or (weight_diff > 0 and weight_diff <= self.batch_weight - self.batches[order_j.batch_id].weight):
                combs_feas.append(comb)

        count = 0
        while count < card:
            # refresh so the order objects are bound to the one assigned to the batches (weird behavior in python)
            all_orders = {
                order.ID: order for batch in self.batches.values() for order in list(batch.orders.values())
            }

            combs = list(set(combs).difference(processed))
            if len(combs) == 0:
                return processed

            order_weights = {
                order.ID: sum(
                    [
                        (
                                len(self.batches[order.batch_id].items_of_shelves[key]) -
                                order.chosen_shelves.get(key, 0)
                        ) == 0
                        for key in self.batches[order.batch_id].items_of_shelves.keys()
                    ]
                )
                for order in all_orders.values()
            }

            comb_weights = [sum(itemgetter(*combs[i])(order_weights)) for i in range(len(combs))]

            if sum(comb_weights) == 0:
                probs = [1 / len(combs)] * len(combs)
            else:
                probs = np.array(comb_weights) / sum(comb_weights)

            orders = combs.pop(np.random.choice(range(len(combs)), p=probs))
            order_i, order_j = itemgetter(*orders)(all_orders)

            # somehow have to do that, otherwise objects diverge leading to complex problems
            order_i = self.batches[order_i.batch_id].orders[order_i.ID]
            order_j = self.batches[order_j.batch_id].orders[order_j.ID]

            # through exchange in previous iterations, this case could occur
            if order_i.batch_id == order_j.batch_id:
                continue
            # do not make the same operation twice
            processed.append(orders)
            # get batches of orders
            batch_i = self.batches[order_i.batch_id]
            batch_j = self.batches[order_j.batch_id]
            # has been checked, but through previous iterations could have changed
            weight_diff = order_i.weight - order_j.weight
            if (
                    weight_diff <= 0 and abs(weight_diff) <= self.batch_weight - batch_i.weight
            ) or (weight_diff > 0 and weight_diff <= self.batch_weight - batch_j.weight):
                count += 1

                batch_i.del_order(order_i, self.item_id_pod_id_dict)
                batch_j.add_order(order_i)
                self.greedy_cobot_tour(batch_j, items=order_i.items)

                batch_j.del_order(order_j, self.item_id_pod_id_dict)
                batch_i.add_order(order_j)
                self.greedy_cobot_tour(batch_i, items=order_j.items)

                if len(batch_i.items) == 0:
                    self.batches.pop(batch_i.ID)
                    batch_i = None
                if len(batch_j.items) == 0:
                    self.batches.pop(batch_j.ID)
                    batch_j = None
                while any([
                    abs(len(self.get_batches_for_station(ps1)) - len(self.get_batches_for_station(ps2))) >= 2
                    for ps1, ps2 in list(itertools.combinations(self.warehouseInstance.OutputStations.keys(), 2))
                ]):
                    self.swap_ps_of_batch()
                if batch_i: self.local_search_shelves(batch_i)
                if batch_j: self.local_search_shelves(batch_j)
        return processed

    def determine_switchable_orders_randomized(
        self, k, processed
    ):
        """
        given two batches (batch ids), this function determines one order of each batch which can be exchanged between
        the batches (wrt to capacity restrictions).
        :param batch_i:
        :param batch_j:
        :return:
        """
        batch_i, batch_j = np.random.choice(
            [batch for key, batch in self.batches.items()], 2, replace=False
        )

        batch_i_orders = list(batch_i.orders.values())
        batch_j_order = list(batch_j.orders.values())

        card = min(k, len(batch_i_orders) + len(batch_j_order))

        combs = itertools.combinations(
            [order.ID for order in batch_i_orders + batch_j_order], card
        )
        combs = list(set(combs).difference(processed))
        if len(combs) == 0:
            return processed

        batch_i_orders_weights = {
            order.ID: sum(
                [
                    (len(batch_i.items_of_shelves[key]) - order.chosen_shelves.get(key, 0)) == 0
                    for key in batch_i.items_of_shelves.keys()
                ]
            )
            for order in batch_i_orders
        }

        batch_j_orders_weights = {
            order.ID: sum(
                [
                    (
                        len(batch_j.items_of_shelves[key])
                        - order.chosen_shelves.get(key, 0)
                    )
                    == 0
                    for key in batch_j.items_of_shelves.keys()
                ]
            )
            for order in batch_j_order
        }
        order_weights = {**batch_i_orders_weights, **batch_j_orders_weights}

        comb_weights = [sum([order_weights[order] for order in comb]) for comb in combs]
        if sum(comb_weights) == 0:
            probs = [1 / len(combs)] * len(combs)
        else:
            probs = np.array(comb_weights) / sum(comb_weights)
        orders = combs[np.random.choice(range(len(combs)), p=probs)]

        orders_i = [
            batch_i.orders[order]
            for order in set(batch_i.orders.keys()).intersection(orders)
        ]
        orders_j = [
            batch_j.orders[order]
            for order in set(batch_j.orders.keys()).intersection(orders)
        ]

        processed.append(orders)

        orders_i_weight = sum([order_i.weight for order_i in orders_i])
        orders_j_weight = sum([order_j.weight for order_j in orders_j])

        weight_diff = orders_i_weight - orders_j_weight

        if (
            weight_diff <= 0 and abs(weight_diff) <= self.batch_weight - batch_i.weight
        ) or (weight_diff > 0 and weight_diff <= self.batch_weight - batch_j.weight):
            for order_i in orders_i:
                batch_i.del_order(order_i, self.item_id_pod_id_dict)
                batch_j.add_order(order_i)
                self.greedy_cobot_tour(batch_j, items=order_i.items)
            for order_j in orders_j:
                batch_j.del_order(order_j, self.item_id_pod_id_dict)
                batch_i.add_order(order_j)
                self.greedy_cobot_tour(batch_i, items=order_j.items)
            if len(batch_i.items) == 0:
                self.batches.pop(batch_i.ID)
                batch_i = None
            if len(batch_j.items) == 0:
                self.batches.pop(batch_j.ID)
                batch_j = None
            while any([
                abs(len(self.get_batches_for_station(ps1)) - len(self.get_batches_for_station(ps2))) >= 2
                for ps1, ps2 in list(itertools.combinations(self.warehouseInstance.OutputStations.keys(), 2))
            ]):
                self.swap_ps_of_batch()
            if batch_i: self.local_search_shelves(batch_i)
            if batch_j: self.local_search_shelves(batch_j)
        return processed

    def swap_ps_of_batch(self):
        batches_per_ps = {
            ps: len(self.get_batches_for_station(ps)) for ps in self.warehouseInstance.OutputStations.keys()
        }
        ps = max(batches_per_ps, key=batches_per_ps.get)
        savings_per_batch = {
            key: sum(
                [max(self.get_station_with_min_total_distance(order.ID).values())
                 for order in batch.orders.values()]
            )
            for key, batch in self.batches.items()
            if batch.pack_station == ps
        }
        # batch = min(savings_per_batch, key=savings_per_batch.get)
        batch = np.random.choice(
            list(savings_per_batch.keys()), p=np.array(list(savings_per_batch.values()))/sum(savings_per_batch.values())
        )
        self.batches[batch].pack_station = str(
            np.random.choice(
                np.setdiff1d(
                    list(self.warehouseInstance.OutputStations.keys()),
                    [self.batches[batch].pack_station],
                )
            )
        )
        self.simulatedAnnealing(batch=self.batches[batch])

    def swap_order(self, k, processed):
        all_orders = [
            order for batch in self.batches.values() for order in batch.orders.values()
        ]
        all_order_feas = []
        for order in all_orders:
            if any([(self.batch_weight - batch.weight + order.weight) >= 0 for batch in self.batches.values()]):
                all_order_feas.append(order)

        frac_nodes_to_remove = {
            order: sum([
                (num_items_per_shelf - order.shelves.get(key, 0)) == 0
                for key, num_items_per_shelf in {
                    key: len(x)
                    for key, x in self.batches[order.batch_id].items_of_shelves.items()
                }.items()
            ]) / len(order.items)
            for order in all_order_feas
        }
        count = 0

        if all(np.array(list(frac_nodes_to_remove.values()))==0):
            probs = np.array([1/len(frac_nodes_to_remove)]*len(frac_nodes_to_remove))
        else:
            probs = np.array(list(frac_nodes_to_remove.values()))/sum(frac_nodes_to_remove.values())

        orders = np.random.choice(list(frac_nodes_to_remove.keys()), p=probs, size=min(k, len(probs)), replace=False)
        assert all([order is self.batches[order.batch_id].orders[order.ID] for order in all_orders])
        for order in orders:
            # no clue why, but this has to be done, otherwise order becomes new object which causes all kinds of problems
            order = self.batches[order.batch_id].orders[order.ID]
            batch_to_remove_from = self.batches[order.batch_id]

            interactions = {
                batch: sum([
                    any(set(item.shelves) & set(batch.route))
                    for item in order.items.values()
                ]) for batch in self.batches.values()
                if batch.ID != order.batch_id and (self.batch_weight - (batch.weight + order.weight)) >= 0
            }
            if len(interactions)==0:
                continue
            if all(np.array(list(interactions.values())) == 0):
                probs = [1/len(interactions)] * len(interactions)
            else:
                probs = np.array(list(interactions.values()))/sum(interactions.values())

            batch_to_add_to = np.random.choice(list(interactions.keys()), p=probs)

            if order.weight < self.batch_weight - batch_to_add_to.weight:
                count += 1
                batch_to_remove_from.del_order(order, self.item_id_pod_id_dict)
                batch_to_remove_from.route = [
                    x
                    for x in batch_to_remove_from.route
                    if x in list(batch_to_remove_from.items_of_shelves.keys())
                ]

                batch_to_add_to.add_order(order)
                self.greedy_cobot_tour(batch_to_add_to, items=order.items)
                while any([
                    abs(len(self.get_batches_for_station(ps1)) - len(self.get_batches_for_station(ps2))) >= 2
                    for ps1, ps2 in list(itertools.combinations(self.warehouseInstance.OutputStations.keys(), 2))
                ]):
                    self.swap_ps_of_batch()
                self.local_search_shelves(batch_to_add_to)

        return processed

    def randomized_local_search(self, max_iters=10, k=1, lamb=0.7, alpha=0.9):
        """
        perform simple randomized swap of orders to batches. This is done for a given number of iterations on
        randomly drawn orders.
        Neighborhood solution is accepted if it results in a better fitness value
        :return:
        """
        print("local with k=", k)
        iters = 0
        curr_fit = self.get_fitness_of_solution()
        curr_sol = copy.deepcopy(self.batches)
        best_fit = curr_fit
        best_sol = copy.deepcopy(self.batches)
        item_id_pod_id_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
        processed = []
        operators = [self.swap_order, self.exchange_orders, self.determine_switchable_orders_randomized]
        # operators = [self.swap_order]
        weights = [1] * len(operators)
        T = 0.02 * curr_fit
        while iters < max_iters:
            operator_idx = np.random.choice(np.arange(0, len(operators)), p=np.array(weights)/sum(weights))
            operator = operators[operator_idx]
            processed = operator(k, processed)
            new_fit = self.get_fitness_of_solution()
            if self.get_fitness_of_solution() < curr_fit:
                curr_sol = copy.deepcopy(self.batches)
                curr_fit = self.get_fitness_of_solution()
                item_id_pod_id_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
                processed = []
                if curr_fit < best_fit:
                    best_sol = copy.deepcopy(self.batches)
                    best_fit = curr_fit
                    weights[operator_idx] = lamb * weights[operator_idx] + (1-lamb) * 5
                else:
                    weights[operator_idx] = lamb * weights[operator_idx] + (1-lamb) * 3
            elif self.acceptWithProbability(new_fit, curr_fit, T):
                curr_sol = copy.deepcopy(self.batches)
                curr_fit = self.get_fitness_of_solution()
                item_id_pod_id_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
                processed = []
                weights[operator_idx] = lamb * weights[operator_idx] + (1-lamb) * 1
            else:
                self.batches = copy.deepcopy(curr_sol)
                self.item_id_pod_id_dict = copy.deepcopy(item_id_pod_id_dict_copy)
                weights[operator_idx] = lamb * weights[operator_idx] + (1-lamb) * 1
            iters += 1
            T *= alpha
        # set to best solution

        for batch in self.batches.values():
            batch.__dict__ = best_sol[batch.ID].__dict__.copy()
            assert batch is self.batches[batch.ID]
        self.item_id_pod_id_dict = copy.deepcopy(self.item_id_pod_id_dict_orig)
        for batch in self.batches.values():
            for item in batch.items.values():
                self.item_id_pod_id_dict[item.orig_ID][item.shelf] -= 1

    def repair_negative_stock(self):
        class FindSlack:
            def __init__(self, slack, new_batch, item, shelf, item_id_pod_it_dict):
                self.slack = slack
                self.new_batch = new_batch
                self.item = item
                self.shelf = shelf
                self.item_id_pod_id_dict_c = item_id_pod_it_dict

        blacklist = []
        for ik, item in self.item_id_pod_id_dict.items():
            for sk, shelf in item.items():
                if shelf < 0:
                    blacklist.append((ik, sk))
        for item_key, shelf_key in blacklist:
            slacks = []
            for bid, batch in self.batches.items():
                if item_key in [item.orig_ID for item in batch.items_of_shelves.get(shelf_key, [])]:
                    prev_fitness = self.get_fitness_of_batch(batch)

                    curr_batch = copy.deepcopy(batch)
                    item_id_pod_it_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)

                    self.local_search_shelves(batch, item_key, shelf_key)
                    slacks.append(FindSlack(
                        self.get_fitness_of_batch(batch)-prev_fitness,
                        copy.deepcopy(batch),
                        item_key,
                        shelf_key,
                        copy.deepcopy(self.item_id_pod_id_dict)
                    ))

                    batch.__dict__ = curr_batch.__dict__.copy()
                    self.item_id_pod_id_dict = item_id_pod_it_dict_copy
            min_slack = min(slacks, key=attrgetter("slack"))
            self.batches[min_slack.new_batch.ID].__dict__ = min_slack.new_batch.__dict__.copy()
            self.item_id_pod_id_dict = min_slack.item_id_pod_id_dict_c


    def local_search_shelves(self, batch: BatchNew, blacklist_item=None, blacklist_shelf=None):
        curr_batch = copy.deepcopy(batch)
        item_id_pod_it_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
        if not self.is_storage_dedicated:

            batch.route = []
            self.replenish_shelves(batch)
            assigned = []
            while len(assigned) != len(batch.items):
                to_be_assigned = [
                    item
                    for item in batch.items.values()
                    if item.ID not in [i.ID for i in assigned]
                ]
                shelves = [
                    shelf
                    for item in to_be_assigned
                    for shelf in item.shelves
                    if shelf not in batch.route
                    and not (item.orig_ID == blacklist_item and shelf == blacklist_shelf)
                    # and self.item_id_pod_id_dict[item.orig_ID][shelf] > 0
                ]
                shelf_counts = Counter(shelves)

                for a, b in itertools.combinations(shelf_counts, 2):
                    if self.distance_ij[a, b] <= 0:
                        shelf_counts[a], shelf_counts[b] = (
                            shelf_counts[a] + shelf_counts[b],
                        ) * 2
                # with 50 percent prob use the random draw method
                if np.random.random() < 1:
                    # softmax probability
                    sum_counts = sum([np.exp(count) for count in shelf_counts.values()])
                    probs_shelves = {
                        shelf: np.exp(count) / sum_counts
                        for shelf, count in shelf_counts.items()
                    }

                    dists = {
                        shelf: min(
                            [
                                self.distance_ij[node, shelf]
                                for node in batch.route + [batch.pack_station]
                            ]
                        )
                        for shelf in shelves
                    }
                    max_dist = max(dists.values())
                    diff_dist = {
                        shelf: max_dist - dist for shelf, dist in dists.items()
                    }
                    sum_diff_dist = sum(diff_dist.values())
                    if sum_diff_dist == 0:
                        probs_dist = {
                            shelf: 1 / len(diff_dist) for shelf in diff_dist.keys()
                        }
                    else:
                        probs_dist = {
                            shelf: dist / sum_diff_dist
                            for shelf, dist in diff_dist.items()
                        }
                    probs = {
                        key: (probs_shelves[key] + probs_dist[key]) / 2
                        for key in probs_shelves.keys()
                    }

                    top_shelf = np.random.choice(
                        list(probs.keys()), p=list(probs.values())
                    )
                # with 50 percent prob use the rank method
                else:
                    rank = (
                        pd.DataFrame.from_dict(
                            shelf_counts, orient="index", columns=["num_shelves"]
                        )
                        .rank(method="min", ascending=False)
                        .merge(
                            pd.DataFrame.from_dict(
                                {
                                    shelf: min(
                                        [
                                            self.distance_ij[node, shelf]
                                            for node in batch.route
                                            + [batch.pack_station]
                                        ]
                                    )
                                    for shelf in shelves
                                },
                                orient="index",
                                columns=["distance"],
                            ).rank(method="min", ascending=True),
                            left_index=True,
                            right_index=True,
                        )
                    )
                    # random weights for randomization of shelf picks
                    weight_1 = np.random.random()
                    weight_2 = 1 - weight_1
                    top_shelf = rank.apply(
                        lambda x: np.average(x, weights=[weight_1, weight_2]), axis=1
                    ).idxmin()

                items_in_top_shelf = [
                    item for item in to_be_assigned if top_shelf in item.shelves and not (
                            item.orig_ID == blacklist_item and top_shelf == blacklist_shelf
                    )
                ]
                for item in items_in_top_shelf:
                    # if self.item_id_pod_id_dict[item.orig_ID][top_shelf] > -10000:
                    item.shelf = top_shelf
                    self.item_id_pod_id_dict[item.orig_ID][top_shelf] -= 1
                    assigned.append(item)
                batch.route.append(top_shelf)
        # batch_eh, batch_eh_copy = self.swap_ps(batch)
        batch_eh, batch_eh_copy = None, None
        if not blacklist_item:  # dont call the function within recursion
            self.repair_negative_stock()
        # self.swap_ps(batch)
        self.simulatedAnnealing(batch=batch)
        if not self.get_fitness_of_batch(batch) < self.get_fitness_of_batch(curr_batch) and not blacklist_item: # dont revert solution in case we check blacklist impact
            batch.__dict__ = curr_batch.__dict__.copy()
            if batch_eh: batch_eh.__dict__ = batch_eh_copy.__dict__.copy()
            self.item_id_pod_id_dict = item_id_pod_it_dict_copy

    def swap_ps(self, batch_i):
        curr_fit = self.get_fitness_of_solution()
        for batch_j in [
            batch
            for batch in self.batches.values()
            if batch.pack_station != batch_i.pack_station
        ]:
            batch_i_copy = copy.deepcopy(batch_i)
            batch_j_copy = copy.deepcopy(batch_j)
            batch_i.pack_station = batch_j.pack_station
            batch_j.pack_station = batch_i.pack_station
            self.simulatedAnnealing(batch=batch_i)
            self.simulatedAnnealing(batch=batch_j)
            # self.tests()
            if self.get_fitness_of_solution() < curr_fit:
                return batch_j, batch_j_copy
            else:
                batch_i.__dict__ = batch_i_copy.__dict__.copy()
                batch_j.__dict__ = batch_j_copy.__dict__.copy()
        return None, None

    def reduce_number_of_batches(self, max_tries=3):
        tries = 0
        imp = False
        if not np.ceil(
            sum(
                [
                    self.get_total_weight_by_order(order)
                    for order in self.warehouseInstance.Orders.keys()
                ]
            )
            / 18
        ) < len(self.batches):
            print("Number of Batches at optimum")
        else:
            processed = []
            while (
                np.ceil(
                    sum(
                        [
                            self.get_total_weight_by_order(order)
                            for order in self.warehouseInstance.Orders.keys()
                        ]
                    )
                    / 18
                )
                < len(self.batches)
                and tries < max_tries
            ):

                reduced, destroy_batch = self.optimized_perturbation(already_destroyed=processed)
                processed.append(destroy_batch)
                if not reduced:
                    # self.shake(1)
                    tries += 1
        return imp

    def choose_batch_for_destruction(self, with_prob=False, already_destroyed=[]):
        num_batches_per_station = {
            pack_station: len(self.get_batches_for_station(pack_station))
            for pack_station in self.warehouseInstance.OutputStations
        }
        if any(
                [
                    np.abs(v_1 - v_2) > 0
                    for k_1, v_1 in num_batches_per_station.items()
                    for k_2, v_2 in num_batches_per_station.items()
                    if not k_1 == k_2
                ]
        ):
            ps_to_destroy_from = max(
                num_batches_per_station.keys(),
                key=(lambda k: num_batches_per_station[k]),
            )
            weight_dict = self.get_weight_per_batch(ps_to_destroy_from)
        else:
            weight_dict = self.get_weight_per_batch()
        # [weight_dict.pop(x) for x in already_destroyed]
        transformations = {key: max(weight_dict.values())-weight for key, weight in weight_dict.items()}
        probabilities = {key: trans/sum(transformations.values()) for key, trans in transformations.items()}
        if with_prob:
            destroy_batch = np.random.choice(list(probabilities.keys()), p=list(probabilities.values()))
        else:
            destroy_batch = np.random.choice(list(weight_dict.keys()))
        return destroy_batch

    def optimized_perturbation(self, already_destroyed=[]):
        print("try to minimize the number of batches")
        destroy_batch = None
        improvement = True
        change = False
        curr_sol = copy.deepcopy(self.batches)
        item_id_pod_id_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
        while improvement and np.ceil(
                sum(
                    [
                        self.get_total_weight_by_order(order)
                        for order in self.warehouseInstance.Orders.keys()
                    ]
                )
                / 18
        ) < len(self.batches):
            change = True
            destroy_batch = self.choose_batch_for_destruction(
                with_prob=False, already_destroyed=already_destroyed
            )
            orders_to_reassign = self.batches[destroy_batch].orders
            self.replenish_shelves(self.batches[destroy_batch])
            self.batches.pop(destroy_batch)
            # sort orders with respect to their weight in descending order (try to assign the biggest orders first)
            weight_of_orders = {
                k: v
                for k, v in sorted(
                    orders_to_reassign.items(),
                    key=lambda item: item[1].weight,
                    reverse=True,
                )
            }
            # reassign all orders
            for order in weight_of_orders.values():
                candidate_batches = {}
                for key, batch in self.batches.items():
                    candidate_batches[key] = order.weight + batch.weight
                if not candidate_batches:
                    pass
                else:
                    new_batch_of_item = min(
                        candidate_batches.keys(), key=(lambda k: candidate_batches[k])
                    )
                    self.update_batches_from_new_order(
                        new_order=order.ID, batch_id=new_batch_of_item
                    )
            if all(
                    [batch.weight <= self.batch_weight for batch in self.batches.values()]
            ):
                improvement = True
            else:
                improvement = self.repair()
            if improvement:
                print(
                    f"found solution with one batch less. Number of batches now is {len(self.batches)}"
                )
                for batch in self.batches.values():
                    if len(batch.route) == 0:
                        self.greedy_cobot_tour(batch)
                        # self.simulatedAnnealing(batch=batch)
                curr_sol = copy.deepcopy(self.batches)
                item_id_pod_id_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
                # self.tests()
            else:
                self.batches = copy.deepcopy(curr_sol)
                self.item_id_pod_id_dict = copy.deepcopy(item_id_pod_id_dict_copy)
        return bool(improvement * change), destroy_batch

    def repair(self):
        improvement = True
        while (
                not all(
                    [batch.weight <= self.batch_weight for batch in self.batches.values()]
                )
                and improvement
        ):
            improvement = False
            curr_sol = copy.deepcopy(self.batches)
            item_id_pod_id_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
            inf_batch = next(
                batch for batch in self.batches.values() if batch.weight > self.batch_weight
            )
            for other_batch in [
                batch
                for batch in copy.deepcopy(self.batches).values()
                if batch.ID != inf_batch.ID
            ]:
                self.batches.pop(inf_batch.ID)
                slack = np.abs(inf_batch.weight - other_batch.weight)
                self.batches.pop(other_batch.ID)
                orders_of_batches = dict(
                    **copy.deepcopy(inf_batch.orders),
                    **copy.deepcopy(other_batch.orders),
                )
                batch1_orders, batch2_orders = self.kk(
                    [i for i in orders_of_batches.values()]
                )
                batch1_orders = [
                    i for i in orders_of_batches.values() if i.ID in batch1_orders
                ]
                batch2_orders = [
                    i for i in orders_of_batches.values() if i.ID in batch2_orders
                ]
                b1_id = str(self.get_new_batch_id())
                b1 = BatchNew(b1_id, inf_batch.pack_station)
                [b1.add_order(order) for order in batch1_orders]
                self.batches.update({b1.ID: b1})
                b2_id = str(self.get_new_batch_id())
                b2 = BatchNew(b2_id, other_batch.pack_station)
                [b2.add_order(order) for order in batch2_orders]
                self.batches.update({b2.ID: b2})
                if all(
                        [
                            batch.weight <= self.batch_weight
                            for batch in self.batches.values()
                        ]
                ):
                    return True
                elif np.abs(b1.weight - b2.weight) < slack:
                    improvement = self.repair()
                    return improvement
                else:
                    self.batches = copy.deepcopy(curr_sol)
                    self.item_id_pod_id_dict = copy.deepcopy(item_id_pod_id_dict_copy)
        return improvement

    def reduced_vns(self, max_iters, t_max, k_max):
        """
        :param max_iters:  maximum number of iterations
        :param t_max: maximum cpu time
        :param k_max: maximum number of different neighborhoods / shake operations to be performed
        """
        iters = 0
        starttime = time.time()
        t = time.time() - starttime
        time_till_best = None
        curr_fit = self.get_fitness_of_solution()
        best_fit = curr_fit
        curr_sol = copy.deepcopy(self.batches)
        best_sol = copy.deepcopy(curr_sol)
        item_id_pod_id_dict_copy = copy.deepcopy(self.item_id_pod_id_dict)
        T = curr_fit * 0.05
        while iters < max_iters and t < t_max:
            print("Start iteration {} of VNS".format(str(iters)))
            k = 1
            while k < k_max:
                self.shake(k)
                improvement = True
                while improvement:
                    fit_before_ls = self.get_fitness_of_solution()
                    self.reduce_number_of_batches(max_tries=3)
                    self.randomized_local_search(max_iters=20, k=k)
                    neighbor_fit = self.get_fitness_of_solution()
                    print("curr fit: {}; cand fit {}".format(curr_fit, neighbor_fit))
                    if neighbor_fit < fit_before_ls:
                        improvement = True
                    else:
                        improvement = False
                    if (
                        neighbor_fit < curr_fit
                    ):
                        self.tests()

                        curr_fit = neighbor_fit  # accept new solution

                        curr_sol = copy.deepcopy(self.batches)
                        item_id_pod_id_dict_copy = copy.deepcopy(
                            self.item_id_pod_id_dict
                        )
                        k = 1
                        if curr_fit < best_fit:
                            time_till_best = time.time() - starttime
                            best_sol = copy.deepcopy(self.batches)
                            best_fit = self.get_fitness_of_solution()
                            T = best_fit * 0.0001
                    elif self.acceptWithProbability(neighbor_fit, best_fit, T):
                        curr_fit = neighbor_fit  # accept new solution

                        curr_sol = copy.deepcopy(self.batches)
                        item_id_pod_id_dict_copy = copy.deepcopy(
                            self.item_id_pod_id_dict
                        )
                        k += 1
                    else:
                        self.batches = copy.deepcopy(
                            curr_sol
                        )  # don´t accept new solution and stick with the current one
                        self.item_id_pod_id_dict = copy.deepcopy(
                            item_id_pod_id_dict_copy
                        )
                        improvement = False
                        k += 1
            iters += 1
            t = time.time() - starttime

        self.batches = best_sol
        print("best fitness: ", best_fit)
        print(f"took {time_till_best} seconds")


if __name__ == "__main__":
    SKUS = ["24"]  # options: 24 and 360
    SUBSCRIPTS = [""]  # , "_a", "_b"
    NUM_ORDERSS = [10]  # [10,
    MEANS = ["5"]  # "1x6",, "5"
    instance_sols = {}
    model_sols = {}
    NUM_DEPOTS = 2
    for SKU in SKUS:
        for SUBSCRIPT in SUBSCRIPTS:
            for NUM_ORDERS in NUM_ORDERSS:
                for MEAN in MEANS:
                    # CAUTION: SCRIPT WONT RUN IF ALL SOLUTIONS ARE WRITTEN AND THIS IS NOT PUT IN COMMENTS
                    # if os.path.isfile('solutions/final/orders_{}_mean_5_sku_{}{}_{}.xml'.format(str(NUM_ORDERS), SKU, SUBSCRIPT, "mixed")):
                    #    continue  # skip iteration if the instance has been solved already
                    # try:
                    layoutFile = r"data/layout/1-1-1-2-1.xlayo"

                    podInfoFile = "data/sku{}/pods_infos.txt".format(SKU)
                    instances = {}
                    instances[24, 2] = r"data/sku{}/layout_sku_{}_{}.xml".format(
                        SKU, SKU, NUM_DEPOTS
                    )

                    storagePolicies = {}
                    # storagePolicies["dedicated"] = "data/sku{}/pods_items_dedicated_1.txt".format(SKU)
                    storagePolicies['mixed'] = 'data/sku{}/pods_items_mixed_shevels_1-5.txt'.format(SKU)
                    # storagePolicies['mixed'] = 'data/sku{}/pods_items_mixed_shevels_1-10.txt'.format(SKU)

                    orders = {}
                    orders[
                        "{}_5".format(str(NUM_ORDERS))
                    ] = r"data/sku{}/orders_{}_mean_{}_sku_{}{}.xml".format(
                        SKU, str(NUM_ORDERS), MEAN, SKU, SUBSCRIPT
                    )
                    input_files = [storagePolicies, instances, orders, layoutFile, podInfoFile]

                    sols_and_runtimes = {}
                    runtimes = [120]
                    for runtime in runtimes:
                        np.random.seed(1999851)
                        if runtime == 0:
                            ils = GreedyMixedShelves(input_files)
                            ils.apply_greedy_heuristic()
                        else:
                            ils = VariableNeighborhoodSearch(input_files)
                            STORAGE_STRATEGY = (
                                "dedicated" if ils.is_storage_dedicated else "mixed"
                            )
                            print(
                                "Now optimizing: SKU={}; Order={}; Subscript={}; Mean={}; Storage={}".format(
                                    SKU, NUM_ORDERS, SUBSCRIPT, MEAN, STORAGE_STRATEGY
                                )
                            )
                            ils.reduced_vns(max_iters=100, t_max=runtime, k_max=4)
                            # ils.perform_ils(num_iters=1500, t_max=runtime)
                            # vns = VariableNeighborhoodSearch()
                            # vns.reduced_vns(1500, runtime, 2)
                        STORAGE_STRATEGY = (
                            "dedicated" if ils.is_storage_dedicated else "mixed"
                        )
                        ils.write_solution_to_xml(
                            "solutions/orders_{}_mean_{}_sku_{}{}_{}.xml".format(
                                str(NUM_ORDERS), MEAN, SKU, SUBSCRIPT, STORAGE_STRATEGY
                            )
                        )
                        # sols_and_runtimes[runtime] = (vns.get_fitness_of_solution(), {batch.ID: batch.route for
                        #                               batch in vns.batches.values()})
                    print(sols_and_runtimes)
                    instance_sols[(SKU, SUBSCRIPT, NUM_ORDERS)] = sols_and_runtimes
                    model_sols[
                        (SKU, SUBSCRIPT, NUM_ORDERS, "ILS")
                    ] = ils.get_fitness_of_solution()
                    # model_sols[(SKU, SUBSCRIPT, NUM_ORDERS, "VNS")] = vns.get_fitness_of_solution()
                    # except Exception as e:
                    #    print(e)
                    #    continue

    # with open('../analyse_solution/solutions/mixed360_random_ls_not_random_twoopt.pickle', 'wb') as handle:
    #    pickle.dump(instance_sols, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(
        "../analyse_solution/solutions/mixed_fitness_ils_vns.pickle", "wb"
    ) as handle:
        pickle.dump(model_sols, handle, protocol=pickle.HIGHEST_PROTOCOL)
