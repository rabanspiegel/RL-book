from dataclasses import dataclass
from typing import Iterable, List, TypeVar

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
    print("All tests passed.")