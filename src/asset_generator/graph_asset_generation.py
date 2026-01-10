import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


def save_graphs(
    input_data: pd.DataFrame, input_data_generated: pd.DataFrame, path: str
) -> None:
    """
    input_data_generated must follows chronologically input_data
    """
    for df in [input_data, input_data_generated]:
        if "Date" in df.columns:
            df.set_index("Date", inplace=True)

    input_data.index = pd.to_datetime(input_data.index)
    input_data_generated.index = pd.to_datetime(input_data_generated.index)

    assets = input_data.columns
    n_assets = len(assets)

    # 3 graphs for each row maximum
    n_rows = math.ceil(n_assets / 3)

    plt.figure(figsize=(18, 4 * n_rows))

    for i, col in enumerate(assets, 1):
        ax = plt.subplot(n_rows, 3, i)

        # Historical
        ax.plot(input_data.index, input_data[col], color="blue", label="Historical")

        # Generated
        if col in input_data_generated.columns:
            gen_series = input_data_generated[col].dropna()
            if not gen_series.empty:
                ax.plot(
                    gen_series.index, gen_series.values, color="red", label="Generated"
                )

        ax.set_title(col)
        ax.set_xlabel("Date")
        ax.set_ylabel(col)
        ax.grid(True)
        ax.legend()

        full_index = input_data.index.union(input_data_generated.index)

        if len(full_index) > 5:
            xticks = np.linspace(0, len(full_index) - 1, 5, dtype=int)
            ax.set_xticks(full_index[xticks])
            ax.set_xticklabels(
                [full_index[j].strftime("%Y-%m-%d") for j in xticks], rotation=45
            )

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
