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
