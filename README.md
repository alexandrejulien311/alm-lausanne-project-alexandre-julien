# ML tools for Asset-Liability Management

Insurance companies are required to perform stress tests on their portfolios, these stress tests being simulations of extreme scenarios (stock market crashes, loss of customer confidence, sudden increases in interest rates, etc.). This project proposes, through the use of machine learning, tools for the generation of coherent scenarios. For assets, it provides a tool that makes it possible to generate coherent asset values based on given asset values, enabling the creation of a set of future price scenarios that takes into account the relationships between assets. To this end, a modified LSTM was used and produced conclusive results, provided that values are not generated for a future that is too distant. For liabilities, a GAN was used to simulate changes in the customer base within the portfolio, and thus, as a result, changes in contributions and liabilities due.
A detailed description of the method is available: [alm_project_alexandre_juien](docs/alm_project_alexandre_julien.pdf).

## Installation

```bash
conda env create -f environment.yml
conda activate alm-project-ml-alexandre-julien
```
## How to use this project

Quick overview of the functionalyties by files :

*```main.py```: uses previously trained model to generate csv files for assets and liabilities, with graphs for asset.

*```train_asset_model.py```; ```train_liability_model.py```: trains new models using data stored in data/training_data.

*```portfolio_strategy_try.py```: first try to find a portfolio strategy. Problem with the reward.


For ease of use, all the files created by ```main.py```; ```train_asset_model.py```; ```train_liability_model.py```  will appear in the root of the project.


## About data
Datasets have been collected using the links in the py scripts in data/data_collect and have been clean by them after. However payroll_LA_raw.csv is not in the project because it is too large. But, one can find it via the links provided.


## About dependencies
As this project has been coded using a Mac with an Intel chip, some problem with ```pytorch``` could occur. Indeed, torch is not well supported on the Intel chip, but the use of conda seems to solve the problem.
