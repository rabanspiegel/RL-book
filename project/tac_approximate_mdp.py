
from dataclasses import dataclass
from random import random
from typing import Iterable, List, Tuple, TypeVar

import numpy as np
from tac import TacState

from project.tac_simple_mdp_with_dp import CardAction
from rl.function_approx import LinearFunctionApprox
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        MarkovDecisionProcess)
from rl.markov_process import NonTerminal

A = TypeVar('A')


@dataclass(frozen=True)
class TacState:
    position: np.ndarray  # normalized positions of player 1 and player 2
    cards_on_hand: np.ndarray  # one-hot encoded cards on hand of player 1 and player 2

    def __hash__(self):
        return hash((tuple(self.position), tuple(map(tuple, self.cards_on_hand))))

    @staticmethod
    def normalize_positions(positions: List[int], num_fields: int) -> np.ndarray:
        return np.array(positions) / num_fields

    @staticmethod
    def one_hot_encode_cards(cards: List[str], unique_cards: List[str]) -> np.ndarray:
        one_hot = np.zeros(len(unique_cards))
        for card in cards:
            index = unique_cards.index(card)
            one_hot[index] = 1
        return one_hot

    @classmethod
    def from_original_representation(
        cls, position: List[int], cards_on_hand: List[List[str]], num_fields: int, unique_cards: List[str]
    ) -> "TacState":
        normalized_positions = cls.normalize_positions(position, num_fields)
        one_hot_cards = np.array([cls.one_hot_encode_cards(
            cards, unique_cards) for cards in cards_on_hand])
        return cls(normalized_positions, one_hot_cards)


class SimpleTacGame(FiniteMarkovDecisionProcess[TacState, CardAction]):
    unique_cards = ['One', 'Two', 'Three', 'Five', 'Six']
    num_fields = 20
    num_players = 2

    def __init__(self):
        self.card_effects = {
            'One': 1,
            'Two': 2,
            'Three': 3,
            'Five': 5,
            'Six': 6
        }

    def one_hot_decode_cards(self, one_hot: np.ndarray) -> List[str]:
        card_names = []
        for i, card_presence in enumerate(one_hot):
            if card_presence:
                card_names.append(self.unique_cards[i])
        return card_names

    def get_available_actions(self, state: TacState) -> Iterable[CardAction]:
        player_1_cards_one_hot = state.cards_on_hand[0]
        player_1_cards = self.one_hot_decode_cards(player_1_cards_one_hot)
        return [CardAction(card) for card in player_1_cards]

    def is_terminal(self, state: TacState) -> bool:
        step = self.num_fields // len(state.position)
        for p in range(len(state.position)):
            if np.round(state.position[p] * self.num_fields) % self.num_fields == (step * p - 1) % self.num_fields:
                return True
        return False

    def get_initial_state(self) -> TacState:
        positions = [i * (self.num_fields // self.num_players)
                     for i in range(self.num_players)]
        cards_on_hand = [random.sample(self.unique_cards, 2)
                         for _ in range(self.num_players)]
        return TacState.from_original_representation(positions, cards_on_hand, self.num_fields, self.unique_cards)

    def get_next_state_reward(self, state: TacState, action: CardAction) \
            -> Tuple[TacState, float]:

        # Convert continuous positions to integer positions
        int_positions = (state.position * self.num_fields).astype(int)

        positions = int_positions.copy()

        # simulate action of player 0
        played_card = action
        if played_card in self.card_effects:
            positions[0] = (
                positions[0] + self.card_effects[played_card]) % self.num_fields

        # simulate actions of other players by random moves
        for p in range(1, self.players):
            positions[p] = (
                positions[p] + self.card_effects[random.choice(list(self.card_effects.keys()))]) % self.num_fields

        cards = state.cards_on_hand.copy()  # attention not deep copy

        def game_lost() -> bool:
            step = self.num_fields // self.players
            for p in range(1, self.players):
                if positions[p] == (step * p - 1) % self.num_fields:
                    return True
            return False

        reward = 1 if positions[0] == self.num_fields - \
            1 else -1 if game_lost() else 0

        # Convert the updated integer positions back to continuous positions
        new_positions = TacState.normalize_positions(
            positions, self.num_fields)

        return TacState(new_positions, cards), reward

# Define the feature function for your TacState


def tac_state_feature_function(state: TacState) -> np.ndarray:
    features = np.concatenate((state.position, state.cards_on_hand.flatten()))
    return features


# Testing State representation
def test_from_original_representation():
    unique_cards = ['One', 'Two', 'Three', 'Five', 'Six']
    num_fields = 20
    positions = [3, 7]
    cards_on_hand = [['One', 'Two'], ['Three', 'Five']]

    # Create a TacState instance using from_original_representation
    state = TacState.from_original_representation(
        positions, cards_on_hand, num_fields, unique_cards)

    # Check if the positions are normalized correctly
    expected_positions = np.array([3, 7]) / num_fields
    assert np.allclose(
        state.position, expected_positions), f"Positions don't match. Expected {expected_positions}, got {state.position}"

    # Check if the cards are one-hot encoded correctly
    expected_cards_on_hand = np.array([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0]
    ])
    assert np.array_equal(
        state.cards_on_hand, expected_cards_on_hand), f"Cards on hand don't match. Expected {expected_cards_on_hand}, got {state.cards_on_hand}"


if __name__ == '__main__':
    test_from_original_representation()

    game = SimpleTacGame()
    initial_state = game.get_initial_state()

    # Step 2: Create an instance of LinearFunctionApprox
    input_size = len(tac_state_feature_function(
        TacState(np.array([0, 0]), np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))))
    q_approximator = LinearFunctionApprox.create(input_size)

    # Set some necessary parameters
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
    num_episodes = 1000  # Number of episodes for learning

    # Q-learning algorithm
    for episode in range(num_episodes):
        # Initialize the state
        state = initial_state

        # You need to define a function to check if a state is terminal
        while not game.is_terminal(state):
            # Choose an action based on the current Q-function approximator
            # Get the available actions for the current state
            available_actions = game.get_available_actions(state)
            q_values = [q_approximator.evaluate(tac_state_feature_function(state))[
                0] for action in available_actions]
            action = available_actions[np.argmax(q_values)]

            # Take the action and observe the next state and reward
            # Get the next state and reward using the action
            next_state, reward = game.get_next_state_reward(state, action)

            # Compute the target Q-value for the (state, action) pair
            if game.is_terminal(next_state):
                target_q_value = reward
            else:
                next_q_values = [q_approximator.evaluate(tac_state_feature_function(next_state))[
                    0] for action in available_actions]
                target_q_value = reward + gamma * np.max(next_q_values)

            # Compute the current Q-value for the (state, action) pair
            current_q_value = q_approximator.evaluate(
                tac_state_feature_function(state))[0]

            # Update the Q-function approximator using the observed difference between the target and current Q-values
            q_approximator = q_approximator.update(
                [(tac_state_feature_function(state), target_q_value - current_q_value)], alpha)

            # Move on to the next state
            state = next_state
