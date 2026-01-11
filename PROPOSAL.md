Insurance companies are required to perform stress tests on their portfolios, these stress tests being simulations of extreme scenarios (stock market crashes, loss of customer confidence, sudden increases in interest rates, etc.). This project proposes, through the use of machine learning, tools for the generation of coherent scenarios.

For assets, this document provides a tool that makes it possible to generate coherent asset values based on given asset values, enabling the creation of a set of future price scenarios that takes into account the relationships between assets. To this end, a modified LSTM was used and produced conclusive results, provided that values are not generated for a future that is too distant.

For liabilities, a GAN was used to simulate changes in the customer base within the portfolio, and thus, as a result, changes in contributions and liabilities due. Improved training of the model, with particular attention to numerical values, is required for the use of the model.

Finally, a structural foundation for defining a strategy through an AI agent was studied, but further work on the machine learning component remains to be carried out.