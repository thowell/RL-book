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


class SnakesAndLaddersMPFinite(FiniteMarkovProcess[SnakesAndLaddersState]):

    def __init__(
        self,
        initial_position: int,
    ):
        self.initial_position: int = initial_position
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> Transition[SnakesAndLaddersState]:
        d: Dict[SnakesAndLaddersState, Categorical[SnakesAndLaddersState]] = {}
        for pos in range(1,100+1):
            state = SnakesAndLaddersState(pos)
            state_probs_map: Mapping[SnakesAndLaddersState, float] = {
                SnakesAndLaddersState(snake_or_ladder(pos + j)) : end_game_probs(pos, snake_or_ladder(pos + j)) for j in range(1, 6+1)
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

def snake_traces(
    time_steps: int,
    num_traces: int
) -> np.ndarray:
    process = SnakesAndLaddersMPFinite(initial_position = 1)
    start_state = SnakesAndLaddersState(process.initial_position)
    return np.vstack([
        np.fromiter((s.position for s in itertools.islice(
            simulation(process, start_state),
            time_steps + 1
        )), float) for _ in range(num_traces)])

if __name__ == '__main__':
    initial_position = 1

    si_mp = SnakesAndLaddersMPFinite(
        initial_position=initial_position
    )

    # print("Transition Map")
    # print("--------------")
    print(si_mp)

    # print("Stationary Distribution")
    # print("-----------------------")
    # si_mp.display_stationary_distribution()

    # traces
    T = 100
    num_traces = 100
    traces = snake_traces(T, num_traces)
    # print(traces)

    len_game = []
    for i in range(num_traces):
        j = 0
        while j < T:
            if traces[i,j] == 100.0:
                len_game.append(j)
                j = T
            else:
                j += 1
    # print(len_game)

    plt.hist(len_game, density=False, bins=10)  # density=False would make counts
    plt.ylabel('count')
    plt.xlabel('length of game')
    plt.show()
    # plot.savefig('snakes_and_ladders_length_histogram.png')