# Trading Bot from Scratch Using Reinforcement Learning

## Overview
This project implements a **Reinforcement Learning (RL)-based trading bot** from scratch, leveraging **Q-Learning** to make market predictions and optimize trading strategies. The bot processes financial data, extracts key market indicators, and executes trades based on learned patterns.

## Features
- **ETL Pipeline**: Extracts, transforms, and loads financial data using **NumPy** and **Pandas**.
- **Technical Indicators**: Computes features like **RSI, MACD, Bollinger Bands**, and **moving averages**.
- **Q-Learning Agent**: Implements a **table-based RL algorithm** to learn and optimize trading decisions.
- **LLM-driven Sentiment Analysis**: Integrates **news and social media sentiment analysis** to enhance market predictions.
- **Backtesting Framework**: Simulates trading performance on historical data to evaluate the strategy.
- **Risk Management**: Implements position sizing, stop-loss mechanisms, and reward-based trade execution.

## Tech Stack
- **Programming Language**: Python
- **Libraries**: NumPy, Pandas, Matplotlib, Scikit-learn
- **Reinforcement Learning**: Q-Learning, OpenAI Gym (for environment simulation)
- **Data Sources**: Yahoo Finance, Alpha Vantage, News APIs
- **Sentiment Analysis**: LLMs for market sentiment extraction

![image](https://github.com/user-attachments/assets/294f0a06-bd49-47eb-bdbf-026b931e2f48)

![image](https://github.com/user-attachments/assets/6384ea41-4f08-4e0c-ab15-fbe6cba91e44)


## Results
- The bot dynamically adjusts trading strategies based on learned Q-values.
- Sentiment analysis enhances decision-making by incorporating real-time news impact.
- Achieves a **risk-adjusted return** higher than benchmark strategies.

## Future Improvements
- Implement **Deep Q-Networks (DQN)** for more complex decision-making.
- Enhance sentiment analysis with **transformer-based LLMs**.
- Optimize **hyperparameters** using Bayesian optimization.
- Extend to multi-asset portfolio management.

