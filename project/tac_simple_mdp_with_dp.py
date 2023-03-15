import itertools
import random
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from rl.distribution import Categorical
from rl.dynamic_programming import value_iteration_result
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess


class CardAction:
    def __init__(self, card: str):
        self.card = card


@dataclass(frozen=True)
class TacState:
    position: List[int]  # positions of player 1 and player 2
    # cards on hand of player 1 and player 2
    cards_on_hand: Iterable[List[str]]

    def __hash__(self):
        return hash((tuple(self.position), tuple(map(tuple, self.cards_on_hand))))


class SimpleTacGame(FiniteMarkovDecisionProcess[TacState, CardAction]):
    players = 2
    cards = {
        'One': 1,
        'Two': 1,
        'Three': 1,
        'Five': 1
    }
    card_effects = {
        'One': 1,
        'Two': 2,
        'Three': 3,
        'Five': 5,
        'Six': 6
    }
    fields = 20

    def __init__(self):
        super().__init__(self.get_action_transition_reward_map())

    def get_possible_actions_for_state(self, state: TacState) -> Iterable[CardAction]:
        return state.cards_on_hand[0]

    def get_next_state_reward(self, state: TacState, action: CardAction) \
            -> Tuple[TacState, float]:

        positions = state.position.copy()

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

        return TacState(positions, cards), reward

    def get_action_transition_reward_map(self) \
            -> Mapping[TacState, Mapping[CardAction, Categorical[Tuple[TacState, float]]]]:
        states = self.get_all_possible_states()

        transition_map: Dict[TacState, Dict[CardAction,
                                            Categorical[Tuple[TacState, float]]]] = {}
        for state in states:
            action_map = {}
            actions = self.get_possible_actions_for_state(state)
            for action in actions:
                new_state, reward = self.get_next_state_reward(state, action)
                dist: Dict[Tuple[TacState, float], float] = {
                    (new_state, reward): 1}
                action_map[action] = Categorical(dist)

            transition_map[state] = action_map

        return transition_map

    def get_all_possible_states(self) -> Iterable[TacState]:
        return [TacState(pos, cards)
                for pos in self.get_all_possible_positions()
                for cards in self.get_all_possible_cards_combinations()]

    def get_all_possible_cards_combinations(self) -> Iterable[Iterable[Iterable[str]]]:
        card_list = [val for val, count in self.cards.items()
                     for _ in range(count)]
        cards_per_player = len(card_list) // self.players

        def append_cards(round: Iterable[Iterable[str]], remaining_cards: list[str]) \
                -> Iterable[Iterable[Iterable[str]]]:
            if len(remaining_cards) < cards_per_player:
                return [round]

            # calculate all possible hands of new player
            new_player_hands = sorted(
                list(set([tuple(sorted(tup)) for tup in permutations(
                    remaining_cards, cards_per_player)])),
                key=lambda x: x)

            # calculate leftover cards
            leftover_cards = []
            for hand in new_player_hands:
                leftover_card = remaining_cards.copy()
                for card in hand:
                    leftover_card.remove(card)
                leftover_cards.append(leftover_card)

            combos = [append_cards(round + [hand], cards) for hand, cards in
                      list(zip(new_player_hands, leftover_cards))]

            return list(itertools.chain(*combos))

        return append_cards([], card_list)

    def get_all_possible_positions(self) -> Iterable[Iterable[int]]:
        def append_positions(positions: Iterable[List[int]]) -> Iterable[Iterable[int]]:
            new_positions = []
            for pos in positions:
                free_fields = sorted(list(set(range(self.fields)) - set(pos)))
                new_positions += [pos + [i] for i in free_fields]
            return new_positions

        positions = [[]]

        for _ in range(self.players):
            positions = append_positions(positions)

        return positions


if __name__ == '__main__':
    tac_mp = SimpleTacGame()

    vf, pi = value_iteration_result(tac_mp, gamma=0.99)
    print(pi)
