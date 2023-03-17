from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Tuple, List, Sequence

import numpy as np

from rl.distribution import Choose, Categorical
from rl.function_approx import LinearFunctionApprox, Weights
from rl.markov_decision_process import MarkovDecisionProcess, TransitionStep
from rl.markov_process import Terminal, NonTerminal
from rl.monte_carlo import epsilon_greedy_policy
from rl.td import q_learning

A = TypeVar('A')


@dataclass(frozen=True)
class TacGame(ABC):
    @property
    @abstractmethod
    def unique_cards(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def cards_per_player(self) -> int:
        pass

    @property
    @abstractmethod
    def num_fields(self) -> int:
        pass

    @property
    @abstractmethod
    def num_players(self) -> int:
        pass

    @property
    @abstractmethod
    def num_marbles(self) -> int:
        pass

    @abstractmethod
    def card_effects(self) -> dict:
        pass


class SimpleTacGame(TacGame):
    unique_cards = ['One', 'Two', 'Three', 'Five', 'Six']
    card_effects = {
        'One': 1,
        'Two': 2,
        'Three': 3,
        'Five': 5,
        'Six': 6
    }
    cards_per_player = 2
    num_fields = 20
    num_players = 2
    num_marbles = 1


@dataclass(frozen=True)
class TacState:
    position: np.ndarray  # normalized positions of player 1 and player 2
    cards_on_hand: List[np.ndarray]  # one-hot encoded cards on hand of players

    def __hash__(self):
        return hash((tuple(self.position), tuple(map(tuple, self.cards_on_hand))))

    @staticmethod
    def normalize_positions(positions: List[int], num_fields: int) -> np.ndarray:
        return np.array(positions) / num_fields

    @staticmethod
    def denormalize_positions(positions: np.ndarray, num_fields: int) -> List[int]:
        return [int(position * num_fields) for position in positions]

    def get_cards_of_player(self, player: int) -> np.ndarray:
        return self.cards_on_hand[player]

    @staticmethod
    def one_hot_encode_cards(cards: List[str], unique_cards: List[str]) -> np.ndarray:
        one_hot = np.zeros(len(unique_cards))
        for card in cards:
            index = unique_cards.index(card)
            one_hot[index] += 1
        return one_hot


class TacMdp(MarkovDecisionProcess[TacState, A]):

    def __init__(self, game: TacGame):
        self.game = game

    def get_start_state_distribution(self) -> Categorical[TacState]:
        # mix cards on hand
        cards_on_hand = self.mix_cards()

        starting_positions = [self.game.num_fields // self.game.num_players * i
                              for i in range(self.game.num_players) for _ in range(self.game.num_marbles)]

        return Categorical({TacState(TacState.normalize_positions(starting_positions, self.game.num_fields),
                                     cards_on_hand): 1.0})

    def game_lost(self, positions) -> bool:
        step = self.game.num_fields // self.game.num_players
        for p in range(1, self.game.num_players):
            if positions[p] == (step * p - 1) % self.game.num_fields:
                return True
        return False

    def mix_cards(self) -> List[np.ndarray]:
        cards = []
        for p in range(self.game.num_players):
            cards.append(TacState.one_hot_encode_cards(
                np.random.choice(self.game.unique_cards, self.game.cards_per_player, replace=False),
                self.game.unique_cards
            ))
        return cards

    def step(self, state: NonTerminal[TacState], action: A) -> TransitionStep[TacState, A]:
        state = state.state
        # move marble
        next_position = TacState.denormalize_positions(state.position, self.game.num_fields)
        next_position[0] += self.game.card_effects[action]

        # handle collisions
        if next_position[0] in next_position[1:]:
            hit_marble_index = next_position[1:].index(next_position[0])
            hit_player_index = hit_marble_index // self.game.num_players
            new_position = self.game.num_fields // self.game.num_players * hit_player_index
            if new_position in next_position:
                second_hit_marble_index = next_position.index(new_position)
                next_position[second_hit_marble_index] = self.game.num_fields // self.game.num_players * second_hit_marble_index
            next_position[hit_marble_index] = self.game.num_fields // self.game.num_players * hit_player_index

        # remove card from hand
        next_cards_on_hand = state.cards_on_hand.copy()
        next_cards_on_hand[0][self.game.unique_cards.index(action)] -= 1

        # simulate moves from other players
        for player_index in range(1, self.game.num_players):
            # choose card
            card_distribution = Categorical({card: state.get_cards_of_player(player_index)[self.game.unique_cards.index(card)]
                                             for card in self.game.unique_cards})
            card = card_distribution.sample()  # TODO: fix random and only allow for possible cards

            # move marble
            next_position[player_index] += self.game.card_effects[card]

            # handle collisions
            if next_position[player_index] in next_position[:player_index] + next_position[player_index + 1:]:
                hit_marble_index = (next_position[:player_index] + next_position[player_index + 1:]).index(next_position[player_index])
                hit_player_index = hit_marble_index // self.game.num_players
                new_position = self.game.num_fields // self.game.num_players * hit_player_index
                if new_position in next_position:
                    second_hit_marble_index = next_position.index(new_position)
                    next_position[second_hit_marble_index] = self.game.num_fields // self.game.num_players * second_hit_marble_index
                next_position[hit_marble_index] = self.game.num_fields // self.game.num_players * hit_player_index

            # remove card from hand
            next_cards_on_hand[player_index][self.game.unique_cards.index(card)] -= 1

        # shuffle cards on hand if empty
        if np.sum(next_cards_on_hand) == 0:
            next_cards_on_hand = self.mix_cards()

        # calculate next state
        next_state = TacState(TacState.normalize_positions(next_position, self.game.num_fields), next_cards_on_hand)

        # check if game is over
        if self.game_lost(next_position):
            return TransitionStep(
                state=NonTerminal(state),
                action=action,
                reward=-1.0,
                next_state=Terminal(next_state)
            )

        # check if game is won
        if next_position[0] == self.game.num_fields - 1:
            return TransitionStep(
                state=NonTerminal(state),
                action=action,
                reward=1.0,
                next_state=Terminal(next_state)
            )

        return TransitionStep(
            state=NonTerminal(state),
            action=action,
            reward=0.0,
            next_state=NonTerminal(next_state)
        )

    def actions(self, state: TacState) -> Sequence[A]:
        # for later: kick actions that jump over other marbles
        # for later: only allow actions that are possible (e.g. no black card if no marble on field)
        # for later: expand for multiple marbles
        # for later: expand for different effect of cards (e.g. Trickster, 7)
        cards_on_hand = state.cards_on_hand
        return [card for card in self.game.unique_cards if cards_on_hand[0][self.game.unique_cards.index(card)] > 0]



# main
if __name__ == '__main__':
    game = SimpleTacGame()
    mdp = TacMdp(game)

    # simulate game
    state = mdp.get_start_state_distribution().sample()
    print(state)
    actions = mdp.actions(state)
    print(mdp.actions(state))
    next_state = mdp.step(NonTerminal(state), actions[0]).next_state
    print(next_state)

