from src.model_structure import BaseModel
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from src.portfolio_strategy.bond_structure import BondStructure


class InsurancePortfolioEnv(gym.Env):
    def __init__(
        self,
        initial_portfolio: BondStructure,
        liability_z: float,
        rate_list: list[float],
        liabilty_due_s0: float,
        period_T=30,
        solvability_ratio_delta=1.2,
    ):
        super().__init__()
        self.step_t = 0
        self.initial_portfolio_value = initial_portfolio.current_value(rate_list[0])

        self.period_T = period_T
        self.liability_z = liability_z
        self.solvability_ratio_delta = solvability_ratio_delta
        self.liabilty_due_s0 = liabilty_due_s0

        self.rate_list = rate_list
        self.initial_portfolio = initial_portfolio
        self.portfolio = initial_portfolio

        self.current_interest_rate_r = rate_list[self.step_t]

        self.portfolio_value_Y = self.portfolio.current_value(
            self.current_interest_rate_r
        )

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([0.0, -0.1, 0.0]),
            high=np.array([np.inf, 0.2, 1.0]),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.step_t = 0
        self.portfolio = self.initial_portfolio
        self.current_interest_rate = self.rate_list[self.step_t]

        return self._get_obs(), {}

    def _get_obs(self):
        return np.array(
            [
                self.portfolio_value_Y,
                self.current_interest_rate_r,
                self.step_t / self.period_T,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        previous_value_portfolio_Y = self.portfolio_value_Y

        limit_value = self.solvability_ratio_delta * self.liability_z

        max_value_to_reinvest = max(previous_value_portfolio_Y - limit_value, 0.0)
        buy_amount = action[0] * max_value_to_reinvest
        self.portfolio.current_buy_bonds(amount_to_buy=buy_amount)

        self.step_t += 1

        self.current_interest_rate_r = self.rate_list[self.step_t]
        self.portfolio.actuarial_compute(
            new_interest_rate=self.rate_list[self.step_t], leap_time_year=1
        )

        amount_to_sell = self.liabilty_due_s0
        value_not_paid = self.portfolio.current_forced_sell(amount_to_sell)
        curent_portfolio_value = self.portfolio_value_Y

        # reward
        reward = self.portfolio_value_Y - previous_value_portfolio_Y
        alpha = 0.5
        if self.portfolio_value_Y < limit_value:
            penalty = alpha * (limit_value - self.portfolio_value_Y)
            reward -= penalty

        # Termination
        terminated = self.step_t >= self.period_T
        truncated = False

        info = {
            "portfolio_value": self.portfolio_value_Y,
            "reward": reward,
            "step_t": self.step_t,
        }

        return self._get_obs(), reward, terminated, truncated, info


class OptimisationModel(BaseModel):
    def __init__(self):
        pass

    def training_loop(self, epochs): ...

    def save_model(self, folder_path): ...

    def generate(self, input): ...
