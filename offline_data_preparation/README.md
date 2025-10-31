
# Offline Dataset Preparation for Crypto Trading

This folder provides code to train a DRL Agent and generate offline dataset containing (state, action, reward, episode_start), which will be used as input to the Decision Transformer.

The code is largely adapted from [FinAI Contest 2025](https://finrl-contest.readthedocs.io/en/latest/finai2025/task1.html)


### File Descriptions

- `seq_data.py`: Reads BTC's CSV data and generates Alpha101 weak factors.
  - Function `convert_csv_to_level5_csv`: Reads a CSV file into a DataFrame, then extracts the required level5 data, and saves it back as a CSV.
  - Function `convert_btc_csv_to_btc_npy`: Reads a CSV file, saves it as an array, and linearly transforms it to between ±1 using min-max values, saving as an npy. (z-score)

- `seq_run.py`: Inputs Alpha101 weak factor series to train the deep learning recurrent network, predicting future price movement labels.
  - Class `SeqData`: Prepares the input and output sequences for training the neural network, using the function `sample_for_train` to randomly cut sequences for training.
  - Function `train_model`: Uses the condition "number of steps without improvement in loss reaches the set limit" as the criterion for early stopping during training.

- `trade_simulator.py`: Contains a market replay simulator for single commodity.
  - Class `TradeSimulator`: A training-focused market replay simulator, complying with the older gym-style API requirements.
  - Class `EvalTradeSimulator`: A market replay simulator for evaluation.

- `erl_config.py`: Configuration file for reinforcement learning training.

- `erl_agent.py`: Contains the DQN class algorithm for reinforcement learning.

- `erl_replay_buffer.py`: Serves as a training dataset for the reinforcement learning market replay simulator.

- `erl_run.py`: Loads the simulator and trains the FinRL agent.

- `erl_evaluator.py`: Evaluates the performance of the reinforcement learning agent.

- `task1_ensemble.py`: This file contains code that trains multiple models and then saves them to be tested during evaluation.

- `metrics.py`: Contains some metrics for evaluation.

- `convert_replay_buffer_to_trajectories.py`: Converts DRL Agent's replay buffer data from .pth format to trajectory dataset in CSV format.


## Dataset

A dataset containing second-level Limit Order Book (LOB) data and financial news for Bitcoin. Please download the data into the dir `Crypto_Trading/offline_data_preparation/data/` folder.

| **Filename**                              | **Description**                                                                                                           | **Link**                                                                                                                    |
|-------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| `FinRL_BTC_news_signals`                      | BTC news with corresponding DeepSeek V3 engineered sentiment scores and risk scores.                                      | [Hugging Face](https://huggingface.co/datasets/SecureFinAI-Lab/FinRL_BTC_news_signals)                       |
| `BTC_1sec_with_sentiment_risk_train.csv`  | The merged and time-aligned version of the previous two datasets.                                                         |   [Google Drive](https://drive.google.com/drive/folders/1rV9tJ0T2iWNJ-g3TI4Qgqy0cVf_Zqzqp?usp=sharing)                                  |

The `BTC_1sec_with_sentiment_risk_train.csv` contains all data used to train the RNN model and FinRL agent. 

## Setup

1. Create and activate a virtual environment:  

```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download BTC datasets into data directory

4. Generate Technical Factors Alpha101 Describing the Market

Run `seq_data.py`'s convert_btc_csv_to_btc_npy:

- col1: AlphaID from Alpha1 to Alpha101
- col2: Used time (second) total
- col3: Used time (second) of single Alpha
- col4: alpha shape
- col5: number of nan `nan_rate= nan_number/alpha.shape[0]`

5. Supervised Learning Training Loop Network to Aggregate Multiple Factor Sequences into Fewer Strong Signal Factors

Run `seq_run.py`. Remember to set your GPU_ID, e.g.:

```bash
    python3 seq_run.py 0
```

6. Train FinRL Agents

```bash
   python3 erl_run.py 0         # Single agent
   python3 task1_ensemble.py 0  # Ensemble methods
```