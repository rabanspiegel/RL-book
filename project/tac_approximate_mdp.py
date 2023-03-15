
from dataclasses import dataclass
from random import random
from typing import Iterable, List, Tuple, TypeVar

import numpy as np
from tac import TacState

from project.tac_simple_mdp_with_dp import CardAction
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
    positions = [3, 7]
    cards_on_hand = [['One', 'Two'], ['Three', 'Five']]

    state = TacState.from_original_representation(
        positions, cards_on_hand, num_fields, unique_cards)

    def get_next_state_reward(self, state: TacState, action: CardAction) \
            -> Tuple[TacState, float]:

        # Convert continuous positions to integer positions
        int_positions = (state.position * self.fields).astype(int)

        positions = int_positions.copy()

        # simulate action of player 0
        played_card = action
        if played_card in self.card_effects:
            positions[0] = (
                positions[0] + self.card_effects[played_card]) % self.fields

        # simulate actions of other players by random moves
        for p in range(1, self.players):
            positions[p] = (
                positions[p] + self.card_effects[random.choice(list(self.card_effects.keys()))]) % self.fields

        cards = state.cards_on_hand.copy()  # attention not deep copy

        def game_lost() -> bool:
            step = self.fields // self.players
            for p in range(1, self.players):
                if positions[p] == (step * p - 1) % self.fields:
                    return True
            return False

        reward = 1 if positions[0] == self.fields - \
            1 else -1 if game_lost() else 0

        # Convert the updated integer positions back to continuous positions
        new_positions = TacState.normalize_positions(positions, self.fields)

        return TacState(new_positions, cards), reward

    def tac_state_feature_function(state: TacState) -> np.ndarray:
        # Concatenate the position and one-hot encoded cards information
        features = np.concatenate(
            (state.position, state.cards_on_hand.flatten()))
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
