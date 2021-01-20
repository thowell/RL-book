from dataclasses import dataclass
from typing import Optional, Mapping, Sequence, Tuple
from rl.distribution import Categorical
from rl.markov_process import Transition, FiniteMarkovProcess
from scipy.stats import poisson
import numpy as np
from collections import Counter
import itertools
from operator import itemgetter
import matplotlib.pyplot as plt
import os

@dataclass(frozen=True)
class FrogState:
    position: int

    def position(self) -> int:
        return self.position


class FrogMPFinite(FiniteMarkovProcess[FrogState]):

    def __init__(
        self,
        initial_position: int,
    ):
        self.initial_position: int = initial_position
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> Transition[FrogState]:
        d: Dict[FrogState, Categorical[FrogState]] = {}
        for pos in range(1,10+1):
            state = FrogState(pos)
            state_probs_map: Mapping[FrogState, float] = {
                FrogState(j) : 1.0 for j in range(pos+1 if pos != 10 else pos, 10+1)
            }
            d[state] = Categorical(state_probs_map)
        return d

    def next_state(self, state):
        return self.get_transition_map()[state].sample()

def simulation(process, start_state):
    state = start_state
    while True:
        yield state
        state = process.next_state(state)

def frog_traces(
    time_steps: int,
    num_traces: int
) -> np.ndarray:
    process = FrogMPFinite(initial_position = 1)
    start_state = FrogState(process.initial_position)
    return np.vstack([
        np.fromiter((s.position for s in itertools.islice(
            simulation(process, start_state),
            time_steps + 1
        )), float) for _ in range(num_traces)])

if __name__ == '__main__':
    initial_position = 1

    si_mp = FrogMPFinite(
        initial_position=initial_position
    )

    # print("Transition Map")
    # print("--------------")
    print(si_mp)

    # print("Stationary Distribution")
    # print("-----------------------")
    # si_mp.display_stationary_distribution()

    # traces
    T = 10
    num_traces = 1000
    traces = frog_traces(T, num_traces)
    # print(traces)

    len_game = []
    for i in range(num_traces):
        j = 0
        while j < T:
            if traces[i,j] == 10.0:
                len_game.append(j)
                j = T
            else:
                j += 1
    print(len_game)
    print()

    plt.hist(len_game, density=False, bins=2)  # density=False would make counts
    plt.ylabel('count')
    plt.xlabel('length of game')
    plt.show()
    # plot.savefig('snakes_and_ladders_length_histogram.png')