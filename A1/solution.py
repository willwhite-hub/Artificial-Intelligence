import sys
from constants import *
from environment import *
from state import State
from queue import PriorityQueue
import heapq
from functools import lru_cache

"""
solution.py

This file is a template you should use to implement your solution.

You should implement 

COMP3702 2024 Assignment 1 Support Code
"""


class StateNode:
    """
    Class to represent the node in the frontier.

    Each entry in the fronteir stores:
        - The current state of the environment
        - The parent node for the current state
        - The action from the parent that derived this node: FORWARD, REVERSE, SPIN_LEFT, SPIN_RIGHT
        - The path cost to reach this state
    """

    def __init__(
        self, env, state, parent, action_from_parent, path_steps, path_cost: int
    ):
        self.env = env
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.path_steps = path_steps
        self.path_cost = path_cost

    def get_path(self):
        """
        Get a list of actions from the root to the current node.
        """
        path = []
        node = self
        while node.action_from_parent is not None:
            path.append(node.action_from_parent)
            node = node.parent
        return list(reversed(path))

    def get_successors(self):
        """
        Get a list of s StateNodes.
        """
        successors = []
        for action in BEE_ACTIONS:
            success, cost, next_state = self.env.perform_action(self.state, action)
            if success:
                successors.append(
                    StateNode(
                        self.env,
                        next_state,
                        self,
                        action,
                        self.path_steps + 1,
                        self.path_cost + cost,
                    )
                )
        return successors

    # Compare the path cost of two nodes.
    def __lt__(self, other):
        return self.path_cost < other.path_cost

    # Check if the state is the goal state.
    def __eq__(self, obj):
        return self.state == obj.state


class Solver:

    def __init__(self, environment, loop_counter):
        self.environment = environment
        self.loop_counter = loop_counter

    # === Uniform Cost Search ==========================================================================================
    def solve_ucs(self):
        """
        Find a path which solves the environment using Uniform Cost Search (UCS).
        :return: path (list of actions, where each action is an element of BEE_ACTIONS)

        Reference:
        Tutorial_3_solution.py, COMP3702 2024 Teaching Staff
        """
        frontier = [
            StateNode(
                self.environment, self.environment.get_init_state(), None, None, 0, 0
            )
        ]
        heapq.heapify(frontier)
        # dictionary to store the path cost to reach each state
        reached = {self.environment.get_init_state(): 0}
        n_expanded = 0
        while frontier:
            n_expanded += 1
            self.loop_counter.inc()
            # expand the node with the lowest path cost
            node = heapq.heappop(frontier)

            # check if the state is the goal state
            if self.environment.is_solved(node.state):
                print(f'Visited Nodes: {len(reached.keys())}, \t\tExpanded Nodes: {n_expanded},\t\t'
                      f'Nodes in Frontier: {len(frontier)}')
                print(f'Path Cost: {node.path_cost}')
                return node.get_path()

            # get the successors of the current node
            successors = node.get_successors()
            for s in successors:
                if s.state not in reached.keys() or s.path_cost < reached[s.state]:
                    reached[s.state] = s.path_cost
                    heapq.heappush(frontier, s)

        return None

    # === A* Search ====================================================================================================

    def preprocess_heuristic(self):
        """
        Perform pre-processing (e.g. pre-computing repeatedly used values) necessary for your heuristic,
        """

        #
        #
        # TODO: (Optional) Implement code for any preprocessing required by your heuristic here (if your heuristic
        #  requires preprocessing).
        #
        # If you choose to implement code here, you should call this method from your solve_a_star method (e.g. once at
        # the beginning of your search).
        #
        #

        pass

    @lru_cache(maxsize=None)
    def compute_heuristic(self, state):
        """
        Compute a heuristic value h(n) for the given state.
        :param state: given state (GameState object)
        :return a real number h(n)

        Reference:
        The euclidean distance equation was formulated with help from GitHub Copilot on August 16 2024.
        """
      
        bee_posit = state.BEE_posit
        widget_centres = state.widget_centres

        h = 0
        for widget_centre in widget_centres:
            # check if widget movement type is being moved by the bee
            if (
                widget_get_movement_type(
                    state.BEE_orient, state.BEE_posit, widget_centre
                )
                == TRANSLATE
            ):
                continue
                # only move in REVERSE or SPIN_LEFT/RIGHT when holding widget
                for a in BEE_ACTIONS:
                    if a == REVERSE or a == SPIN_LEFT or a == SPIN_RIGHT:
                        continue
                    # Euclidean distance heuristic
                    h += (
                        (bee_posit[0] - widget_centre[0]) ** 2
                        + (bee_posit[1] - widget_centre[1]) ** 2
                    ) ** 0.5
                    

        return h
    
        

    @lru_cache(maxsize=None)
    def solve_a_star(self):
        """
        Find a path which solves the environment using A* search.
        :return: path (list of actions, where each action is an element of BEE_ACTIONS)

        Reference:
        Tutorial_3_solution.py, COMP3702 2024 Teaching Staff
        """
        frontier = [
            (
                0 + self.compute_heuristic(self.environment.get_init_state()),
                StateNode(
                    self.environment,
                    self.environment.get_init_state(),
                    None,
                    None,
                    0,
                    0,
                ),
            )
        ]
        heapq.heapify(frontier)
        # dictionary to store the path cost to reach each state
        reached = {self.environment.get_init_state(): 0}
        n_expanded = 0
        while frontier:
            n_expanded += 1
            self.loop_counter.inc()
            # expand the node with the lowest path cost
            _, node = heapq.heappop(frontier)

            # check if the state is the goal state
            if self.environment.is_solved(node.state):
                print(f'Visited Nodes: {len(reached.keys())}, \t\tExpanded Nodes: {n_expanded},\t\t'
                      f'Nodes in Frontier: {len(frontier)}')
                print(f'Path Cost: {node.path_cost}')
                return node.get_path()

            # get the successors of the current node
            successors = node.get_successors()
            for s in successors:
                if s.state not in reached.keys() or s.path_cost < reached[s.state]:
                    reached[s.state] = s.path_cost
                    heapq.heappush(
                        frontier, (s.path_cost + self.compute_heuristic(s.state), s)
                    )

        return None

