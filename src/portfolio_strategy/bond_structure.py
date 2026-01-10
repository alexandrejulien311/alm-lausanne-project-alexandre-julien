import math
import random


def zero_coupon_price(
    new_interest_rate: float, nominal_value: float, rest_time_year: float
):
    return nominal_value * math.exp(-new_interest_rate * rest_time_year)


def sell_expired_bond(interest_rate: float, nominal_value: float, year: int) -> float:
    return nominal_value * (1 + interest_rate) ** year


class BondStructure:
    def __init__(
        self,
        matrix: list[float, float, int, float],
        cash_amount: float,
        interest_rate: float,
        year: int,
        step_number_for_one_year: int,
        current_time=0,
    ):
        """
        The lines of matrix are structured, at current time as follow:
        [current_price, nominal_value, expiration date (int), initial interest rate]
        """
        self.matrix = matrix
        self.current_time = current_time
        self.cash_amount = cash_amount
        self.year = year
        self.interest_rate = interest_rate

        # How many step for one year
        self.step_number_for_one_year = step_number_for_one_year

    def actuarial_compute(self, new_interest_rate: float, leap_time_year: int) -> None:
        new_matrix = self.matrix
        self.current_time += leap_time_year
        for i in range(len(self.matrix)):
            if self.current_time < self.matrix[i][2]:
                new_matrix[i][0] = zero_coupon_price(
                    new_interest_rate=new_interest_rate,
                    nominal_value=self.matrix[i][1],
                    rest_time_year=self.current_time,
                )
            else:
                self.cash_amount += sell_expired_bond(
                    nominal_value=self.matrix[i][1],
                    interest_rate=self.matrix[i][3],
                    year=self.year,
                )

        self.matrix = new_matrix
        self.interest_rate = new_interest_rate

    def current_value(self, current_interest_rate: float) -> float:
        value = self.cash_amount
        for i in range(len(self.matrix)):

            value += zero_coupon_price(
                new_interest_rate=current_interest_rate,
                nominal_value=self.matrix[i][1],
                rest_time_year=self.current_time,
            )
        return value

    def current_forced_sell(self, amount_to_sell: float) -> float:
        """
        If there is not enough cash in this bond section, sell oldest bonds until amount_to_sell <= 0.
        Sell the cash first.
        Return the rest to pay if there is not enough value in this bond section, else return 0.
        """
        if self.cash_amount >= amount_to_sell:
            self.cash_amount = self.cash_amount - amount_to_sell
            return 0
        else:
            amount_to_sell = amount_to_sell - self.cash_amount
            while amount_to_sell > 0 and len(self.matrix) > 0:
                amount_to_sell -= self.matrix[0][0]
                self.matrix.pop(0)
            if amount_to_sell < 0:
                self.cash_amount = -1 * amount_to_sell
                return 0
            else:
                return amount_to_sell

    def current_buy_bonds(
        self, amount_to_buy: float, value_one_bond: int = 100
    ) -> None:
        """
        Buy bonds at current interest rate until there is not enough budget.
        Then put the leftovers in cash.
        """
        bonds_number = int(amount_to_buy) // value_one_bond
        cash_rest = float(int(amount_to_buy) % value_one_bond) + amount_to_buy % 1
        new_bonds = [
            [
                value_one_bond,
                value_one_bond,
                (self.current_time / self.step_number_for_one_year) + self.year,
                self.interest_rate,
            ]
            for _ in range(bonds_number)
        ]
        self.matrix = self.matrix + new_bonds
        self.cash_amount += cash_rest


def initial_portfolio_generation(
    value_required: float,
    current_interest_rate: float,
    value_by_month=3,
    nominal_value=100,
) -> list[list[float, float, int, float]]:
    matrix = []
    period_T = value_by_month * 12
    total_current_price = 0
    while total_current_price < value_required:
        new_bond = generate_one_bond(
            nominal_value, value_by_month, current_interest_rate, period_T=period_T
        )
        total_current_price += new_bond[0]
        matrix.append(new_bond)
    return matrix


def generate_one_bond(
    nominal_value: float,
    value_by_month: int,
    current_interest_rate: float,
    period_T: int,
) -> list[float, float, int, float]:
    random_rate = round(random.uniform(0, 0.06), 4)
    due_date = random.randint(1, value_by_month * 12)
    new_bond = [
        zero_coupon_price(current_interest_rate, nominal_value, due_date / period_T),
        nominal_value,
        due_date,
        random_rate,
    ]
    return new_bond
