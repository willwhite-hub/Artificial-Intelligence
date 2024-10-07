import sys
import time
import numpy as np
from constants import *
from environment import *
from state import State
import functools

"""
solution.py

This file is a template you should use to implement your solution.

You should implement each section below which contains a TODO comment.

COMP3702 2022 Assignment 2 Support Code

"""

class Solver:

    def __init__(self, environment: Environment):
        self.environment = environment
        self.states = list()  # List of all states
        self.rewards = dict()  # Dictionary of rewards
        self.probabilities = dict()  # Dictionary of transition probabilities
        self.values = None  # Value of a state
        self.policy = None  # Optimal policy
        self.gamma = environment.gamma  # Discount factor
        self.epsilon = environment.epsilon  # Convergence threshold
        self.converged = False  # Flag to check if the algorithm has converged
        self.diffs = []  # List of differences between values
        self.goal_reward = 0.0 # Reward for reaching the goal state

        # Perfrom BFS to get all possible outcomes
        frontier = [self.environment.get_init_state()]
        while frontier:
            node = frontier.pop(0)
            for action in BEE_ACTIONS:
                _, next_state = self.environment.apply_dynamics(node, action)
                if next_state not in self.states:
                    self.states.append(next_state)  # Add new state to the list of states
                    frontier.append(next_state)

        #Initialise rewards and probabilities
        for state in self.states:
            self.probabilities[state] = dict()
            self.rewards[state] = dict()
            for action in BEE_ACTIONS:
                self.probabilities[state][action], self.rewards[state][action] = self.get_transition_outcomes(state, action) 
       
    @staticmethod
    def testcases_to_attempt():
        """
        Return a list of testcase numbers you want your solution to be evaluated for.
        """
        # TODO: modify below if desired (e.g. disable larger testcases if you're having problems with RAM usage, etc)
        return [1,2,3,4,5,6]

    # === Value Iteration ==============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        """
        self.values = {state: 0.0 for state in self.states}
        self.policy = {state: np.random.choice(BEE_ACTIONS) for state in self.states}

    def vi_is_converged(self):
        """
        Check if Value Iteration has reached convergence.
        :return: True if converged, False otherwise.
        """
        return self.converged

    def vi_iteration(self):
        """
        Perform a single iteration of Value Iteration (i.e. loop over the state space once).
        """

        new_values = dict(self.values)  # Use in-place updates
        new_policy = dict(self.policy)
        for state in self.states:
            best_q = -np.inf
            best_action = None
            for action in BEE_ACTIONS:
                total = 0
                for successor in self.probabilities[state][action]:
                    for i in range(len(self.probabilities[state][action][successor])):
                        total += self.probabilities[state][action][successor][i] * \
                            (self.rewards[state][action][successor][i] + self.gamma * self.values[successor])
                if total > best_q:
                    best_q = total
                    best_action = action
            # Update the policy with the best action
            new_values[state] = best_q
            new_policy[state] = best_action
        
        # Convergence check
        diffs = [abs(self.values[state] - new_values[state]) for state in self.states]
        max_diff  = max(diffs)
        self.diffs.append(max_diff)

        if max_diff < self.epsilon:
            self.converged = True
        
        # Update values and policy
        self.values = new_values
        self.policy = new_policy
        
    def vi_plan_offline(self):
        """
        Plan using Value Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.vi_initialise()
        while True:
            self.vi_iteration()

            # NOTE: vi_iteration is always called before vi_is_converged
            if self.vi_is_converged():
                break

    def vi_get_state_value(self, state: State):
        """
        Retrieve V(s) for the given state.
        :param state: the current state
        :return: V(s)
        """
        return self.values[state]
    

    def vi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration.
        :param state: the current state
        :return: optimal action for the given state (element of BEE_ACTIONS)
        """
        return self.policy[state]

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        self.policy = {state: np.random.choice(BEE_ACTIONS) for state in self.states}
        self.values = {state: 0.0 for state in self.states}

    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        return self.converged

    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """
        v_pi = self.policy_evaluation()
        self.policy_improvement(v_pi)

    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.pi_initialise()
        while True:
            self.pi_iteration()

            # NOTE: pi_iteration is always called before PI_is_converged
            if self.pi_is_converged():
                break

    def pi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of BEE_ACTIONS)
        """
        return self.policy[state]

    # === Helper Methods ===============================================================================================
    def get_prob(self, action: int):
        """
        Get the probability of moving in the given direction.
        :param action: element of BEE_ACTIONS
        :return: each movement and its probability
        """
        dbl = self.environment.double_move_probs[action]
        cw = self.environment.drift_cw_probs[action]
        ccw = self.environment.drift_ccw_probs[action]

        # Calculate the probability of each movement
        probabilities = {
            "nominal": (1 - dbl) * (1 - cw - ccw),
            "cw, nominal": cw * (1 - dbl),
            "ccw, nominal": ccw * (1 - dbl),
            "cw, double": dbl * cw,
            "ccw, double": dbl * ccw,
            "double": dbl * (1 - cw - ccw)
        }

        # Dictionary to store all movements:
        movements = {
            "nominal": [action],
            "cw, nominal": [SPIN_RIGHT, action],
            "ccw, nominal": [SPIN_LEFT, action],
            "cw, double": [SPIN_RIGHT, action, action],
            "ccw, double": [SPIN_LEFT, action, action],
            "double": [action, action]
        }


        # Calculate the probability of each movement
        probabilities ={
            "nominal": (1 - dbl) * (1 - cw - ccw),
            "cw, nominal": cw * (1 - dbl),
            "ccw, nominal": ccw * (1 - dbl),
            "cw, double": dbl * cw,
            "ccw, double": dbl * ccw,
            "double": dbl * (1 - cw - ccw)
        }
                
        return probabilities, movements
    
    def get_transition_outcomes(self, state: State, action):
        """
        Get the possible outcomes of taking the given action in the given state.
        :param state: the current state
        :param action: element of BEE_ACTIONS
        :return: probabilities and rewards for each possible outcome
        """

        # Handle exited state
        if self.environment.is_solved(state):
            return {state: [1]}, {state: [self.goal_reward]}

        # Initialize probability and reward dictionaries
        probabilities, rewards = dict(), dict()

        # Get probabilities and movements
        prob_dict, move_dict = self.get_prob(action)

        # Apply dynamics for various cases
        for i, (move_key, prob) in enumerate(prob_dict.items()):
            self.update_prob_reward(state, move_dict[move_key], prob, probabilities, rewards)

        return probabilities, rewards

    def update_prob_reward(self, state, actions, prob, probabilities, rewards):
        """
        Helper function to apply dynamics, compute cost, and update probabilities and rewards.
        """
        cost_list = []
        new_state = state
        for act in actions:
            cost, new_state = self.environment.apply_dynamics(new_state, act)
            cost_list.append(cost)

        max_cost = np.max(cost_list)  # Take max of all applied costs

        # Update probability
        if new_state in probabilities:
            probabilities[new_state].append(prob)
        else:
            probabilities[new_state] = [prob]

        # Update reward
        if new_state in rewards:
            if self.environment.is_solved(new_state):
                rewards[new_state].append(self.goal_reward)
            else:
                rewards[new_state].append(max_cost)
        else:
            if self.environment.is_solved(new_state):
                rewards[new_state] = [self.goal_reward]
            else:
                rewards[new_state] = [max_cost]

    def policy_evaluation(self, max_iter=23):
        """
        Perform policy evaluation with a fixed number of iterations for faster convergence.
        """
        for _ in range(max_iter):
            new_values = dict(self.values)  # Use in-place updates

            for state in self.states:
                action = self.policy[state]
                Q_value = 0
                for successor in self.probabilities[state][action]:
                    for i in range(len(self.probabilities[state][action][successor])):
                        Q_value += self.probabilities[state][action][successor][i] * \
                                (self.rewards[state][action][successor][i] + self.gamma * self.values[successor])

                # Update value in place
                new_values[state] = Q_value

            # Convergence check
            diffs = [abs(self.values[state] - new_values[state]) for state in self.states]
            if max(diffs) < self.epsilon:
                break

            self.values = new_values  # Update with the newly computed values

        return self.values

    def policy_improvement(self, v_pi):
        """
        Perform a single step of policy improvement.
        """
        policy_changed = False
        # Loop over each state and improve the policy using 1-step look-ahead.
        for state in self.states:
            best_q = -np.inf
            best_action = None
            for action in BEE_ACTIONS:
                total = 0
                for successor in self.probabilities[state][action]:
                    for i in range(len(self.probabilities[state][action][successor])):
                        total += self.probabilities[state][action][successor][i] * \
                            (self.rewards[state][action][successor][i] + self.gamma * v_pi[successor])
                if total > best_q:
                    best_q = total
                    best_action = action
            # Update the policy with the best action
            if best_action != self.policy[state]:
                policy_changed = True
                self.policy[state] = best_action

        self.converged = not policy_changed
