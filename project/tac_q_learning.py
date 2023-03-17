import numpy as np

from project.tac_mdp import TacMdp
from rl.distribution import Choose
from rl.function_approx import LinearFunctionApprox, Weights
from rl.monte_carlo import epsilon_greedy_policy
from rl.td import q_learning

if __name__ == '__main__':
    state_length: int = 2
    tac_mdp: TacMdp = TacMdp()

    epsilon: float = 0.3
    gamma: float = 0.9
    max_episode_length: int = 100
    approx_0: LinearFunctionApprox = LinearFunctionApprox.create(
        weights=Weights.create(np.zeros(state_length)),
        feature_functions=[lambda s: s[i] for i in range(state_length)]
    )

    q_learning(
        mdp=tac_mdp,
        policy_from_q=lambda f, m: epsilon_greedy_policy(
            q=f,
            mdp=m,
            ε=epsilon
        ),
        states=Choose(tac_mdp.non_terminal_states),  # TODO: states
        approx_0=approx_0,
        γ=gamma,
        max_episode_length=max_episode_length
    )
