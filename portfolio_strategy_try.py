from src.portfolio_strategy.model_portfolio import InsurancePortfolioEnv
from src.portfolio_strategy.bond_structure import (
    BondStructure,
    initial_portfolio_generation,
)
from data.training_data import TRAINING_DATA_FOLDER_PATH
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
import time


GENERATED_DATA_PATH = (
    Path(__file__).resolve().parent
    / TRAINING_DATA_FOLDER_PATH
    / "asset_synthetic_data.csv"
)
generated_data = pd.read_csv(GENERATED_DATA_PATH)

rate_df_1y = generated_data["rate_1Y"] / 100
rate_list = rate_df_1y.values


value_required = 1_500_000

matrix = initial_portfolio_generation(
    value_required=value_required,
    current_interest_rate=rate_list[0],
)

cash_amount = 1000


initial_portfolio = BondStructure(
    matrix=matrix,
    cash_amount=cash_amount,
    interest_rate=rate_list[0],
    year=1,
    step_number_for_one_year=1 * 3 * 12,
)

liabilty_z = 1000000
liability_due_s0 = 1000

value_by_month = 3
period_T = value_by_month * 12


env = InsurancePortfolioEnv(
    initial_portfolio=initial_portfolio,
    liability_z=liabilty_z,
    liabilty_due_s0=liability_due_s0,
    period_T=period_T,
    rate_list=rate_list,
)


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    gamma=1.0,
    learning_rate=3e-4,
    batch_size=64,
    n_steps=128,
)


n_test = 10
start = time.time()
model.learn(total_timesteps=n_test)
end = time.time()

time_per_step = (end - start) / n_test
print(f"Time for steps: {time_per_step:.5f} s")

# see one trajectory
obs, info = env.reset()
terminated = False
truncated = False

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

print("Final value of the portfolio :", env.portfolio_value_Y)
