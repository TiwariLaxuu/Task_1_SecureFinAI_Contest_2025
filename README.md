# FinAI Contest Task 1  - Crypto Trading with Decision Transformer

This task is to implement a Decision Transformer model for algorithmic cryptocurrency trading using offline reinforcement learning. The system processes high-frequency Bitcoin order book data with sentiment analysis and trains trading strategies through sequence modeling.

## Starter Kit Description

This starter kit demonstrates how to use the provided code:

1. **Data Preparation**: Process BTC order book data and generate Alpha101 technical factors
2. **Supervised Learning**: Train RNN models to aggregate factor sequences into strong signals  
3. **RL Agent Training**: Train reinforcement learning agents on market simulation
4. **Trajectory Collection**: Convert RL agent replay buffers to offline datasets
5. **Decision Transformer Training**: Train transformer model on collected trajectories
6. **Evaluation**: Test the trained model's trading performance on out-of-sample data

## Repository Structure

```
Crypto_Trading/
├── offline_data_preparation/
│   ├── README.md                                     
│   ├── requirements.txt                             
│   ├── seq_data.py                                  # BTC data processing and 
│   ├── seq_run.py                                   # RNN training for factor 
│   ├── trade_simulator.py                           # Market replay simulator
│   ├── erl_config.py                                # RL training configuration
│   ├── erl_replay_buffer.py                         # RL training dataset 
│   ├── erl_run.py                                   # Single RL agent training
│   ├── erl_agent.py                                 # DQN Algorithm
│   ├── task1_ensemble.py                            # Ensemble RL agent training
│   ├── erl_evaluator.py                             # RL agent performance 
│   ├── metrics.py                                   # Metrics for evaluation
|   └── convert_replay_buffer_to_trajectories.py     # Convert RL data to DT format
├── dt_crypto.py                                     # Decision Transformer training
└── README.md                                        
```

## Workflow

### 1. Data Requirements

Download the required datasets into `Crypto_Trading/offline_data_preparation/data/`:

| **Filename** | **Description** | **Link** |
|--------------|-----------------|----------|
| `FinRL_BTC_news_signals` | BTC news with DeepSeek V3 engineered sentiment scores and risk scores | [Hugging Face](https://huggingface.co/datasets/SecureFinAI-Lab/FinRL_BTC_news_signals) |
| `BTC_1sec_with_sentiment_risk_train.csv` | Merged and time-aligned LOB data with sentiment | [Google Drive](https://drive.google.com/drive/folders/1rV9tJ0T2iWNJ-g3TI4Qgqy0cVf_Zqzqp?usp=sharing) |

The dataset contains:

- **Second-level Limit Order Book (LOB) data** for Bitcoin
- **Financial news sentiment analysis** with risk scores
- **Time-aligned merged data** for training

### 2. Alpha101 Technical Factor Generation

Process raw BTC data and generate technical factors:

```bash
cd Crypto_Trading/offline_data_preparation
python seq_data.py
```

This script:

- Reads BTC CSV data and extracts level-5 order book information
- Generates Alpha101 weak factors describing market microstructure
- Converts data to normalized numpy arrays using z-score transformation
- Outputs technical factor sequences for model training

**Key functions**:

- `convert_csv_to_level5_csv`: Extracts required LOB data
- `convert_btc_csv_to_btc_npy`: Normalizes and saves as numpy arrays

### 3. RNN Factor Aggregation Training

Train deep learning models to aggregate multiple factor sequences:

```bash
python seq_run.py 0  # GPU ID
```

This script:

- Takes Alpha101 weak factor series as input
- Trains recurrent neural networks to predict future price movements
- Aggregates multiple weak factors into fewer strong signal factors
- Uses early stopping based on validation loss improvement


### 4. RL Agent Training

Train reinforcement learning agents using the market simulator:

#### Single Agent Training

```bash
python erl_run.py 0  # GPU ID
```

#### Ensemble Training

```bash
python task1_ensemble.py 0  # GPU ID
```

### 5. Trajectory Conversion

Convert RL agent replay buffers to Decision Transformer format:

```bash
python convert_replay_buffer_to_trajectories.py --replay_buffer_dir ./TradeSimulator-v0_D3QN_0 --output_file ../crypto_trajectories.csv
```

This script:
- Loads replay buffer data from `.pth` files
- Identifies episode boundaries from undone flags
- Converts to CSV format with columns: `state`, `action`, `reward`, `episode_start`
- Calculates return-to-go values for each trajectory step

### 6. Decision Transformer Training

Train the Decision Transformer model using `dt_crypto.py`:

```bash
cd Crypto_Trading
python dt_crypto.py \
  --epochs 100 \
  --lr 1e-3 \
  --context_length 20 \
  --model_path ./trained_models/decision_transformer.pth \
  --plots_dir plots
```

**Model architecture**:

- State dimension: Derived from processed LOB and sentiment features
- Action dimension: 3 (buy, hold, sell)
- Context length: 20 time steps
- Hidden size: 64
- Transformer layers: 3
- Attention heads: 1

**Training features**:

- Return-to-go conditioning for goal-directed behavior
- Causal action masking for proper sequential modeling
- Early stopping with validation loss monitoring
- Comprehensive loss tracking and visualization

### 7. Model Evaluation

Evaluate the trained model using `evaluation.py`:

```bash
cd Crypto_Trading
python evaluation.py \
  --model_path ./trained_models/decision_transformer.pth \
  --test_data_path ./offline_data_preparation/data/BTC_1sec_with_sentiment_risk_test.csv \
  --max_samples 35000 \
  --target_return 250.0 \
  --context_length 20 \
  --plots_dir plots
```

**Evaluation metrics**:

- Portfolio value progression
- Total and annualized returns
- Sharpe ratio
- Maximum drawdown
- Annual volatility
- Calmar ratio
- Win rate

## Configuration

### Training Parameters (dt_crypto.py)

Configurable through command line arguments:
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-3)
- `--context_length`: Sequence context length (default: 20)
- `--model_path`: Path to save trained model (default: './trained_models/decision_transformer.pth')
- `--plots_dir`: Directory for training plots (default: 'plots')

### Evaluation Parameters (evaluation.py)

- `--model_path`: Path to trained model
- `--test_data_path`: Path to test data CSV
- `--max_samples`: Maximum samples for evaluation
- `--target_return`: Target return for evaluation (default: 250.0)
- `--context_length`: Context length (default: 20)
- `--plots_dir`: Directory for evaluation plots (default: 'plots')

---

See [offline_data_preparation/README](offline_data_preparation/README.md) for details about offline data preparation.
