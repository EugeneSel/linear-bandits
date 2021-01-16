#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 21, 2020
@author: Thomas Bonald <bonald@enst.fr>
"""
import numpy as np
from copy import deepcopy

from display import display_position, display_board


class Environment:
    """Generic environment.
    Reward only depends on the target state.
    """
    def __init__(self):
        self.init_state()

    def init_state(self):
        self.state = None

    @staticmethod
    def get_states():
        states = []
        return states
    
    @staticmethod
    def get_actions(state):
        actions = []
        return actions

    @staticmethod
    def encode(state):
        return state
    
    @staticmethod
    def decode(state):
        return state
    
    @staticmethod
    def get_transition(state, action):
        probs = [1]
        states = [deepcopy(state)]
        return probs, states

    @staticmethod
    def get_reward(state):
        return None

    @staticmethod
    def is_terminal(state):
        return True

    @staticmethod
    def get_model(state, action):
        probs, states = Environment.get_transition(state, action)
        rewards = [Environment.get_reward(state) for state in states]
        return probs, states, rewards
    
    def step(self, action):
        """Apply action, get reward and modify state.
        Returns reward, stop (``True`` if terminal state).
        """
        reward = None
        stop = True
        if action is not None and action in self.get_actions(self.state):
            probs, states, rewards = self.get_model(self.state, action)
            i = np.random.choice(len(probs), p=probs)
            state = states[i]
            self.state = state
            reward = rewards[i]
            stop = self.is_terminal(state)
        return reward, stop

    
class Agent:
    """Agent. Default policy is purely random."""
    def __init__(self, environment, policy=None):
        if policy is not None:
            self.policy = policy
        else:
            self.policy = self.random_policy
        self.environment = environment
            
    def random_policy(self, state):
        actions = self.environment.get_actions(state)
        probs = np.ones(len(actions)) / len(actions)
        return probs, actions

    def get_action(self, state):
        action = None
        probs, actions = self.policy(state)
        if len(actions):
            i = np.random.choice(len(actions), p=probs)
            action = actions[i]
        return action


class PolicyEvaluation:
    """Online evaluation of a policy."""
    
    def __init__(self, environment, policy=None, gamma=0.9, alpha=0.1, eps=0.5, n_steps=1000, init_value=0):
        self.environment = environment
        self.agent = Agent(environment, policy)
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.n_steps = n_steps
        self.init_value = init_value
        self.get_states()
        self.get_actions()
        self.get_rewards()
        self.init_values()
        if not len(self.states):
            print("Warning: Method 'get_states' not available in this environment.")
        
    def get_states(self):
        self.states = self.environment.get_states()
        self.state_id = {self.environment.encode(state): i for i, state in enumerate(self.states)}

    def get_actions(self):
        self.actions = self.environment.get_actions()
        self.action_id = {action: i for i, action in enumerate(self.actions)}
                
    def get_rewards(self):
        self.rewards = np.array([self.environment.get_reward(state) for state in self.states])
    
    def init_values(self):
        self.values = self.init_value * np.ones(len(self.states))

    def get_episode(self):
        """Get the states and rewards for an episode."""
        self.environment.init_state()
        states = []
        rewards = []
        for t in range(self.n_steps):
            state = deepcopy(self.environment.state)
            states.append(state)
            action = self.agent.get_action(state)
            reward, stop = self.environment.step(action)
            rewards.append(reward)
            if stop:
                break
        return states, rewards
    
    def improve_policy(self):
        """Get policy based on current estimation of values."""
        best_actions = []
        for state in self.states:
            actions = self.environment.get_actions(state)
            rewards = []
            for action in actions:
                probs, states = self.environment.get_transition(state, action)
                index = np.array([self.state_id[self.environment.encode(state)] for state in states])
                values = self.rewards[index] + self.gamma * self.values[index]
                rewards.append(np.sum(np.array(probs) * values))
            best_actions.append(actions[np.argmax(rewards)])
        # randomized policy for exploration
        def policy(state):
            actions = self.environment.get_actions(state)
            if len(actions) == 1:
                return [1], actions
            else:
                probs = np.zeros(len(self.actions))
                action_ids = np.array([self.action_id[action] for action in actions])
                probs[action_ids] = self.eps / len(actions)
                state_id = self.state_id[self.environment.encode(state)]
                best_action_id = self.action_id[best_actions[state_id]]
                probs[best_action_id] += 1 - self.eps
                return probs, self.actions
        self.agent = Agent(self.environment, policy)
        self.init_values()
        # greedy policy for exploitation
        def policy(state):
            state_id = self.state_id[self.environment.encode(state)]
            action = best_actions[state_id]
            return [1], [action]
        return policy    
    

class Walk(Environment):
    """1D Walk."""

    Length = 10
    Reward_States = [1, 8]
    Reward_Values = [1, 2]

    def __init__(self):
        super(Walk, self).__init__()    

    @classmethod
    def set_parameters(cls, length, reward_states, reward_values):
        cls.Length = length
        cls.Reward_States = reward_states
        cls.Reward_Values = reward_values

    def init_state(self):
        self.state = np.random.choice(Walk.Length)

    @staticmethod
    def get_states():
        n = Walk.Length
        states = list(range(n))
        return states
    
    @staticmethod
    def get_actions(state=None):
        actions = [1, -1]
        if state is not None:
            if state == Walk.Length - 1:
                actions = [-1]
            if state == 0:
                actions = [1]
        return actions

    @staticmethod
    def get_transition(state, action):
        state += action
        probs = [1]
        states = [state]
        return probs, states

    @staticmethod
    def get_reward(state):
        reward = 0
        if state in Walk.Reward_States:
            reward = Walk.Reward_Values[Walk.Reward_States.index(state)]
        return reward

    @staticmethod
    def is_terminal(state):
        return False

    @staticmethod
    def get_model(state, action):
        probs, states = Walk.get_transition(state, action)
        rewards = [Walk.get_reward(state) for state in states]
        return probs, states, rewards

    def display(self, states=None, marker='o', marker_size=200, marker_color='b', interval=200):
        image = 200 * np.ones((1, Walk.Length, 3)).astype(int)
        if states is not None:
            positions = [(0, state) for state in states]
        else:
            positions = None
        position = (0, self.state)
        return display_position(image, position, positions, marker, marker_size, marker_color, interval)


class Maze(Environment):
    """Maze."""

    Map = np.ones((2, 2)).astype(int)
    Init_State = (0, 0)
    Exit_States = [(1, 1)]

    def __init__(self):
        super(Maze, self).__init__()

    @classmethod
    def set_parameters(cls, maze_map, init_state, exit_states):
        cls.Map = maze_map
        cls.Init_State = init_state
        cls.Exit_States = exit_states

    def init_state(self):
        self.state = np.array(Maze.Init_State)

    @staticmethod
    def is_valid(state):
        n, m = Maze.Map.shape
        x, y = tuple(state)
        return 0 <= x < n and 0 <= y < m and Maze.Map[x, y]
    
    @staticmethod
    def get_states():
        n, m = Maze.Map.shape
        states = [np.array([x,y]) for x in range(n) for y in range(m) if Maze.is_valid(np.array([x,y]))]
        return states
    
    @staticmethod
    def encode(state):
        return tuple(state)
    
    @staticmethod
    def decode(state):
        return np.array(state)

    @staticmethod
    def get_actions(state=None):
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if state is not None:
            actions = []
            for move in moves:
                if Maze.is_valid(state + move):
                    actions.append(move)
        else:
            actions = moves.copy()
        return actions

    @staticmethod
    def get_transition(state, action):
        probs = [1]
        states = [state.copy() + action]
        return probs, states

    @staticmethod
    def get_reward(state):
        return int(tuple(state) in Maze.Exit_States)

    @staticmethod
    def is_terminal(state):
        return tuple(state) in Maze.Exit_States

    @staticmethod
    def get_model(state, action):
        probs, states = Maze.get_transition(state, action)
        rewards = [Maze.get_reward(state) for state in states]
        return probs, states, rewards

    def display(self, states=None, marker='o', marker_size=200, marker_color='b', interval=200):
        shape = (*Maze.Map.shape, 3)
        image = np.zeros(shape).astype(int)
        for i in range(3):
            image[:, :, i] = 255 * Maze.Map
        return display_position(image, self.state, states, marker, marker_size, marker_color, interval)


class TicTacToe(Environment):
    """Tic-tac-toe game."""

    def __init__(self, play_first=True, adversary_policy=None):
        self.player = 2 * play_first - 1
        self.adversary = Agent(self, adversary_policy)
        super(TicTacToe, self).__init__()

    def init_state(self):
        board = np.zeros((3, 3)).astype(int)
        self.state = [self.player, board]

    @staticmethod
    def get_actions(state=None):
        if state is not None:
            _, board = state
            x_, y_ = np.where(board == 0)
            actions = [(x, y) for x, y in zip(x_, y_)]
        else:
            actions = [(x, y) for x in range(3) for y in range(3)]
        return actions

    @staticmethod
    def get_transition(state, action):
        player, board = deepcopy(state)
        board[action] = player
        state = -player, board
        probs = [1]
        states = [state]
        return probs, states

    @staticmethod
    def get_reward(state):
        _, board = state
        sums = set(board.sum(axis=0)) | set(board.sum(axis=1))
        sums.add(board.diagonal().sum())
        sums.add(np.fliplr(board).diagonal().sum())
        if 3 in sums:
            reward = 1
        elif -3 in sums:
            reward = -1
        else:
            reward = 0
        return reward

    @staticmethod
    def is_terminal(state):
        return bool(TicTacToe.get_reward(state)) or not len(TicTacToe.get_actions(state))

    @staticmethod
    def get_model(state, action):
        probs, states = TicTacToe.get_transition(state, action)
        rewards = [TicTacToe.get_reward(state) for state in states]
        return probs, states, rewards

    def step(self, action=None):
        player, _ = self.state
        if player < 0:
            action = self.adversary.get_action(self.state)
        return Environment.step(self, action)

    def display(self, states=None, marker1='X', marker2='o', marker_size=2000, color1='b', color2='r', interval=300):
        image = 200 * np.ones((3, 3, 3)).astype(int)
        if states is not None:
            boards = [state[1] for state in states]
        else:
            boards = None
        _, board = self.state
        return display_board(image, board, boards, marker1, marker2, marker_size, color1, color2, interval)


class ConnectFour(Environment):
    """Connect Four game."""

    def __init__(self, play_first=True, adversary_policy=None):
        self.player = 2 * play_first - 1
        self.adversary = Agent(self, adversary_policy)
        super(ConnectFour, self).__init__()

    def init_state(self):
        board = np.zeros((6, 7)).astype(int)
        self.state = [self.player, board]

    @staticmethod
    def get_actions(state=None):
        if state is not None:
            _, board = state
            actions = np.where(board[0] == 0)[0]
        else:
            actions = np.arange(7)
        return actions

    @staticmethod
    def get_transition(state, action):
        player, board = deepcopy(state)
        row = np.argwhere(board[:, action] == 0).ravel()[-1]
        board[row, action] = player
        state = -player, board
        probs = [1]
        states = [state]
        return probs, states

    @staticmethod
    def get_reward(state):
        _, board = state
        sep = ','
        sequence = np.array2string(board, separator=sep)
        sequence += np.array2string(board.T, separator=sep)
        sequence += ''.join([np.array2string(board.diagonal(offset=k), separator=sep) for k in range(-2, 4)])
        sequence += ''.join([np.array2string(np.fliplr(board).diagonal(offset=k), separator=sep) for k in range(-2, 4)])
        pattern_pos = sep.join(4 * [' 1'])
        pattern_neg = sep.join(4 * ['-1'])
        if pattern_pos in sequence:
            reward = 1
        elif pattern_neg in sequence:
            reward = -1
        else:
            reward = 0
        return reward

    @staticmethod
    def is_terminal(state):
        return bool(ConnectFour.get_reward(state)) or not len(ConnectFour.get_actions(state))

    @staticmethod
    def get_model(state, action):
        probs, states = ConnectFour.get_transition(state, action)
        rewards = [ConnectFour.get_reward(state) for state in states]
        return probs, states, rewards

    def step(self, action=None):
        player, _ = self.state
        if player < 0:
            action = self.adversary.get_action(self.state)
        return Environment.step(self, action)

    def display(self, states=None, marker1='o', marker2='o', marker_size=1000, color1='gold', color2='r', interval=200):
        image = np.zeros((6, 7, 3)).astype(int)
        image[:, :, 2] = 255
        if states is not None:
            boards = [state[1] for state in states]
        else:
            boards = None
        _, board = self.state
        return display_board(image, board, boards, marker1, marker2, marker_size, color1, color2, interval)


class Nim(Environment):
    """Nim game."""

    Init_Board = [1, 3, 5, 7]

    @classmethod
    def set_init_board(cls, init_board):
        cls.Init_Board = init_board

    def __init__(self, play_first=True, adversary_policy=None):
        self.player = 2 * play_first - 1
        self.adversary = Agent(self, adversary_policy)
        super(Nim, self).__init__()

    def init_state(self):
        board = np.array(Nim.Init_Board).astype(int)
        self.state = [self.player, board]

    @staticmethod
    def get_actions(state=None):
        if state is None:
            state = Nim().state
        _, board = state
        rows = np.where(board)[0]
        actions = [(row, number + 1) for row in rows for number in range(board[row])]
        return actions

    @staticmethod
    def get_transition(state, action):
        player, board = deepcopy(state)
        row, number = action
        board[row] -= number
        state = -player, board
        probs = [1]
        states = [state]
        return probs, states

    @staticmethod
    def get_reward(state):
        player, board = state
        if np.sum(board) > 0:
            reward = 0
        else:
            reward = player
        return reward

    @staticmethod
    def is_terminal(state):
        _, board = state
        return not np.sum(board)

    @staticmethod
    def get_model(state, action):
        probs, states = Nim.get_transition(state, action)
        rewards = [Nim.get_reward(state) for state in states]
        return probs, states, rewards

    def step(self, action=None):
        player, _ = self.state
        if player < 0:
            action = self.adversary.get_action(self.state)
        return Environment.step(self, action)

    def display(self, states=None, marker='d', marker_size=500, color='gold', interval=200):
        board = np.array(Nim.Init_State).astype(int)
        image = np.zeros((len(board), np.max(board), 3)).astype(int)
        image[:, :, 1] = 135
        if states is not None:
            positions = []
            for _, board in states:
                x = []
                y = []
                for row in np.where(board)[0]:
                    for col in range(board[row]):
                        x.append(row)
                        y.append(col)
                positions.append((x, y))
        else:
            positions = None
        _, board = self.state
        x = []
        y = []
        for row in np.where(board)[0]:
            for col in range(board[row]):
                x.append(row)
                y.append(col)
        position = x, y
        return display_position(image, position, positions, marker, marker_size, color, interval)
