from dataclasses import dataclass
from typing import Optional, Mapping, Tuple, Dict
import numpy as np
import itertools
from operator import itemgetter
from rl.distribution import Categorical, Constant, SampledDistribution
from rl.markov_process import MarkovProcess, MarkovRewardProcess
from rl.gen_utils.common_funcs import get_logistic_func, get_unit_sigmoid_func
from rl.chapter2.stock_price_simulations import\
    plot_single_trace_all_processes
from rl.chapter2.stock_price_simulations import\
    plot_distribution_at_time_all_processes

# stock price reward function
def stock_reward(x):
    return 0.5 * (x - 10.0)**2 + 10.0 * x 

@dataclass(frozen=True)
class StateMP1:
    price: int


@dataclass
class StockPriceMP1(MarkovRewardProcess[StateMP1]):

    level_param: int  # level to which price mean-reverts
    alpha1: float = 0.25  # strength of mean-reversion (non-negative value)

    def up_prob(self, state: StateMP1) -> float:
        return get_logistic_func(self.alpha1)(self.level_param - state.price)

    def transition(self, state: StateMP1) -> Categorical[StateMP1]:
        up_p = self.up_prob(state)

        return Categorical({
            StateMP1(state.price + 1): up_p,
            StateMP1(state.price - 1): 1 - up_p
        })

    def transition_reward(
        self,
        state: StateMP1
    ) -> SampledDistribution[Tuple[StateMP1, float]]:

        def sample_next_state_reward(state=state) ->\
                Tuple[StateMP1, float]:
            next_state = self.transition(state).sample()
            reward: float = stock_reward(state.price)
            return next_state, reward

        return SampledDistribution(sample_next_state_reward)

if __name__ == '__main__':
    gamma = 0.9

    mrp = StockPriceMP1(
        level_param = 100,
        alpha1 = 0.25
    )

    state_init = StateMP1(100.0)
    dist_init = Constant(state_init)
    
    horizon = 10000
    trace = list(itertools.islice(
            mrp.simulate_reward(dist_init),
            horizon))

    cumulative_reward = sum(step.reward * gamma **t for (t, step) in enumerate(trace))
    print("cumulative reward: ", cumulative_reward)