from dataclasses import dataclass
from typing import Tuple, Dict
from rl.markov_process import MarkovRewardProcess
from rl.markov_process import FiniteMarkovRewardProcess
from rl.markov_process import RewardTransition
from scipy.stats import poisson
from rl.distribution import SampledDistribution, Categorical
import numpy as np


@dataclass(frozen=True)
class SnakesAndLaddersState:
    position: int

    def position(self) -> int:
        return self.position

# special transitions
def snake_or_ladder(pos):
    # ladders
    if pos == 3:
        return 39
    elif pos ==  7:
        return 48
    elif pos == 12:
        return 51
    elif pos == 20:
        return 41
    elif pos == 25: 
        return 57
    elif pos == 28:
        return 35
    elif pos == 45:
        return 74
    elif pos == 60:
        return 85
    elif pos == 67:
        return 90
    elif pos == 69:
        return 92
    elif pos == 77:
        return 83

    # snakes
    elif pos == 31:
        return 6
    elif pos == 38:
        return 1
    elif pos == 49:
        return 8
    elif pos == 53:
        return 17
    elif pos == 65:
        return 14
    elif pos == 70:
        return 34
    elif pos == 76:
        return 37
    elif pos == 82:
        return 63
    elif pos == 88:
        return 50
    elif pos == 94:
        return 42
    elif pos == 98:
        return 54
    else:
        return np.minimum(100, pos)

def end_game_probs(current, next):
    if current == 95:
        if next == 100:
            return 2.0 / 6.0
    if current == 96: 
        if next == 100:
            return 3.0 / 6.0
    if current == 97:
        if next == 100:
            return 4.0 / 6.0
    if current == 98:
        if next == 100:
            return 5.0 / 6.0
    if current == 99:
        if next == 100:
            return 1.0
    
    return 1.0 / 6.0

class SimpleSnakesAndLaddersMRPFinite(FiniteMarkovRewardProcess[SnakesAndLaddersState]):

    def __init__(
        self,
        initial_position: int,
    ):
        self.initial_position: int = initial_position
        super().__init__(self.get_transition_reward_map())

    def get_transition_reward_map(self) -> RewardTransition[SnakesAndLaddersState]:
        d: Dict[SnakesAndLaddersState, Categorical[Tuple[SnakesAndLaddersState, float]]] = {}
        # for alpha in range(self.capacity + 1):
        #     for beta in range(self.capacity + 1 - alpha):
        #         state = SnakesAndLaddersState(alpha, beta)
        #         ip = state.position()
        #         beta1 = self.capacity - ip
        #         base_reward = - self.holding_cost * state.on_hand
        #         sr_probs_map: Dict[Tuple[SnakesAndLaddersState, float], float] =\
        #             {(SnakesAndLaddersState(ip - i, beta1), base_reward):
        #              self.poisson_distr.pmf(i) for i in range(ip)}
        #         probability = 1 - self.poisson_distr.cdf(ip - 1)
        #         reward = base_reward - self.stockout_cost *\
        #             (probability * (self.poisson_lambda - ip) +
        #              ip * self.poisson_distr.pmf(ip))
        #         sr_probs_map[(SnakesAndLaddersState(0, beta1), reward)] = probability
        #         d[state] = Categorical(sr_probs_map)
        # return d

        d: Dict[SnakesAndLaddersState, Categorical[SnakesAndLaddersState]] = {}
        for pos in range(1,100+1):
            state = SnakesAndLaddersState(pos)
            sr_probs_map: Dict[Tuple[SnakesAndLaddersState, float], float] = {}
            for j in range(1, 6+1):
                next_state = SnakesAndLaddersState(snake_or_ladder(pos + j))
                probs = end_game_probs(pos, snake_or_ladder(pos + j))
                reward = 1.0 if pos != 100 else 0.0
                sr_probs_map[(next_state, reward)] = probs
            d[state] = Categorical(sr_probs_map)
        return d


if __name__ == '__main__':
    user_gamma = 0.9

    si_mrp = SimpleSnakesAndLaddersMRPFinite(
        initial_position = 1
    )

    from rl.markov_process import FiniteMarkovProcess
    print("Transition Map")
    print("--------------")
    print(FiniteMarkovProcess(si_mrp.transition_map))

    print("Transition Reward Map")
    print("---------------------")
    print(si_mrp)

    print("Stationary Distribution")
    print("-----------------------")
    si_mrp.display_stationary_distribution()
    print()

    print("Reward Function")
    print("---------------")
    si_mrp.display_reward_function()
    print()

    print("Value Function")
    print("--------------")
    si_mrp.display_value_function(gamma=user_gamma)
    print()
