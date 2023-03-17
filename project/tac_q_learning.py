import itertools

import numpy as np

from project.tac_mdp import TacMdp, SimpleTacGame
from rl.distribution import Choose
from rl.function_approx import LinearFunctionApprox, Weights
from rl.monte_carlo import epsilon_greedy_policy
from rl.td import q_learning

if __name__ == '__main__':
    state_length: int = 2
    tac_mdp: TacMdp = TacMdp(SimpleTacGame())

    epsilon: float = 0.3
    gamma: float = 0.9
    max_episode_length: int = 100
    approx_0: LinearFunctionApprox = LinearFunctionApprox.create(
        weights=Weights.create(np.zeros(tac_mdp.game.num_players * tac_mdp.game.num_marbles+
                                        tac_mdp.game.num_players * len(tac_mdp.game.unique_cards))),
        feature_functions=
        [lambda s: s.position[i] for i in
         range(tac_mdp.game.num_players * tac_mdp.game.num_marbles)] +  # position features
        [lambda s: s.cards_on_hand[i][j] for i in range(tac_mdp.game.num_players)
         for j in range(len(tac_mdp.game.unique_cards))]  # cards features
    )

    vf = q_learning(
        mdp=tac_mdp,
        policy_from_q=lambda f, m: epsilon_greedy_policy(
            q=f,
            mdp=m,
            ε=epsilon
        ),
        states=tac_mdp.get_start_state_distribution(),
        approx_0=approx_0,
        γ=gamma,
        max_episode_length=max_episode_length
    )

    for vf_iter in itertools.islice(vf, 0, 100):
        print(vf_iter)
