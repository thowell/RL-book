from dataclasses import dataclass
from typing import Tuple, Dict
from rl.markov_decision_process import FiniteMarkovDecisionProcess
from rl.markov_decision_process import FinitePolicy, StateActionMapping
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from rl.distribution import Categorical, Constant
from scipy.stats import poisson


@dataclass(frozen=True)
class FrogEscapeState:
    pad: int

    def pad(self) -> int:
        return self.pad


PadMapping = StateActionMapping[FrogEscapeState, int]

class FrogEscapeMDP(FiniteMarkovDecisionProcess[FrogEscapeState, int]):

    def __init__(
        self,
        n: int,
        initial_pad: int
    ):
        self.n = n
        self.initial_pad = initial_pad

        super().__init__(self.get_action_transition_reward_map())

    def get_action_transition_reward_map(self) -> PadMapping:
        d: Dict[FrogEscapeState, Dict[int, Categorical[Tuple[FrogEscapeState,float]]]] = {}

        # 0 state
        state0: FrogEscapeState = FrogEscapeState(0)
        d[state0] = None

        for i in range(1,self.n):
            state: FrogEscapeState = FrogEscapeState(i)
            di: Dict[int, Categorical[Tuple[FrogEscapeState, float]]] = {}

            # action A
            sr_probs_dict_A: Dict[Tuple[FrogEscapeState, float], float] =\
                        {(FrogEscapeState(i-1), 0.0 if i-1 != 0 else -1.0): i/self.n, (FrogEscapeState(i+1), 0.0 if i+1 != self.n else 1.0): (self.n - i)/self.n}

            di[0] = Categorical(sr_probs_dict_A)

            # action B
            sr_probs_dict_B: Dict[Tuple[FrogEscapeState, float], float] =\
                        {(FrogEscapeState(j), -1.0 if j == 0 else (1.0 if j == self.n else 0.0)): 1/self.n if j != i else 0.0 for j in range(self.n+1)}

            di[1] = Categorical(sr_probs_dict_B)

            # add actions
            d[state] = di

        # n state
        staten: FrogEscapeState = FrogEscapeState(self.n)
        d[staten] = None

        return d

    # def is_terminal(self, state) -> bool:
    #     if state.pad() == 0 or state.pad() == self.n:
    #         return True

if __name__ == '__main__':
    from pprint import pprint

    user_gamma = 1.0

    fe_mdp: FiniteMarkovDecisionProcess[FrogEscapeState, int] =\
        FrogEscapeMDP(
            n = 3,
            initial_pad = 1
        )

    print("MDP Transition Map")
    print("------------------")
    print(fe_mdp)


    # setup deterministic policy
    fdp: FinitePolicy[FrogEscapeState, int] = FinitePolicy(
        {FrogEscapeState(i): Categorical({0: 0.5, 1: 0.5}) for i in range(1, fe_mdp.n)}
        # {FrogEscapeState(i):
        #  Categorial({}) if i == 0 else (Constant(1) if i == fe_mdp.n else Constant(1)) for i in range(fe_mdp.n+1)}
    )

    print("Policy Map")
    print("----------")
    print(fdp)

    implied_mrp: FiniteMarkovRewardProcess[FrogEscapeState] =\
        fe_mdp.apply_finite_policy(fdp)
    print("Implied MP Transition Map")
    print("--------------")
    print(FiniteMarkovProcess(implied_mrp.transition_map))

    print("Implied MRP Transition Reward Map")
    print("---------------------")
    print(implied_mrp)

    print("Implied MP Stationary Distribution")
    print("-----------------------")
    implied_mrp.display_stationary_distribution()
    print()

    print("Implied MRP Reward Function")
    print("---------------")
    implied_mrp.display_reward_function()
    print()

    # print("Implied MRP Value Function")
    # print("--------------")
    # implied_mrp.display_value_function(gamma=user_gamma)
    # print()

    # from rl.dynamic_programming import evaluate_mrp_result
    # from rl.dynamic_programming import policy_iteration_result
    # from rl.dynamic_programming import value_iteration_result

    # print("Implied MRP Policy Evaluation Value Function")
    # print("--------------")
    # pprint(evaluate_mrp_result(implied_mrp, gamma=user_gamma))
    # print()

    # print("MDP Policy Iteration Optimal Value Function and Optimal Policy")
    # print("--------------")
    # opt_vf_pi, opt_policy_pi = policy_iteration_result(
    #     fe_mdp,
    #     gamma=user_gamma
    # )
    # pprint(opt_vf_pi)
    # print(opt_policy_pi)
    # print()

    # print("MDP Value Iteration Optimal Value Function and Optimal Policy")
    # print("--------------")
    # opt_vf_vi, opt_policy_vi = value_iteration_result(fe_mdp, gamma=user_gamma)
    # pprint(opt_vf_vi)
    # print(opt_policy_vi)
    # print()
