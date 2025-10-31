"""
Evaluation script for Crypto Decision Transformer.
Backtests the model against S&P DBM Index benchmark and calculates performance metrics.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DecisionTransformerConfig, DecisionTransformerModel
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import os
import ast
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from dt_crypto import parse_array, preprocess_raw_btc_data, CryptoDataset


class CryptoEvaluator:
    """
    Comprehensive evaluator for Crypto Decision Transformer with benchmark comparison.
    """
    
    def __init__(self, model_path, test_data_path, max_samples=1000, target_return=250.0, context_length=20):
        """
        Initialize the crypto evaluator.
        
        Args:
            model_path: Path to the trained model file
            test_data_path: Path to the test data CSV
            max_samples: Maximum number of samples for evaluation
            target_return: Target return for the model
            context_length: Context length for the model
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.max_samples = max_samples
        self.target_return = target_return
        self.context_length = context_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._load_model_config()
        
        self._load_test_data()
        
        self.model = self._load_model()
        
    def _load_model_config(self):
        """Load model configuration from training data."""
        try:
            df = pd.read_csv("crypto_decision_transformer_ready_dataset.csv", nrows=10)
            first_state = parse_array(df['state'].iloc[0])
            self.state_dim = len(first_state)
            self.act_dim = 3 
            print(f"Model Configuration - State Dimension: {self.state_dim}, Action Dimension: {self.act_dim}")
        except FileNotFoundError:
            print("Warning: Training dataset not found. Using default dimensions.")
            self.state_dim = 12
            self.act_dim = 3
    
    def _load_test_data(self):
        """Load and preprocess test data."""
        print(f"Loading test data from: {self.test_data_path}")
        try:
            self.raw_btc_df = pd.read_csv(self.test_data_path, nrows=self.max_samples)
            print(f"Loaded {len(self.raw_btc_df)} samples for evaluation")
            
            # Preprocess raw data to get state representation
            print("Converting raw BTC data to state representation...")
            print(f"Input data shape: {self.raw_btc_df.shape}")
            self.processed_states = preprocess_raw_btc_data(self.raw_btc_df)
            print(f"Processed states shape: {self.processed_states.shape}")
            
            
            if len(self.processed_states) != len(self.raw_btc_df):
                print(f" Warning: Lost {len(self.raw_btc_df) - len(self.processed_states)} samples during preprocessing")
                
                self.raw_btc_df = self.raw_btc_df.head(len(self.processed_states))
                self.timestamps = self.timestamps.head(len(self.processed_states))
                self.prices = self.prices[:len(self.processed_states)]
            
            timestamp_cols = ['system_time', 'timestamp', 'time']
            timestamp_col = None
            
            for col in timestamp_cols:
                if col in self.raw_btc_df.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col:
                self.timestamps = pd.to_datetime(self.raw_btc_df[timestamp_col])
                print(f"Using timestamps from column '{timestamp_col}': {self.timestamps.iloc[0]} to {self.timestamps.iloc[-1]}")
                
            else:
                start_time = datetime(2021, 4, 13)  
                self.timestamps = pd.date_range(start=start_time, periods=len(self.raw_btc_df), freq='1S')
                print(f"No timestamp column found, created dummy timestamps: {self.timestamps[0]} to {self.timestamps[-1]}")
            
            self.prices = self.raw_btc_df['midpoint'].values
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Test data not found at {self.test_data_path}")
    
    def _load_model(self):
        """Load the trained Decision Transformer model."""
        print(f"Loading model from: {self.model_path}")
        
        config = DecisionTransformerConfig(
            state_dim=self.state_dim, 
            act_dim=self.act_dim, 
            hidden_size=64,
            n_layer=3,
            n_head=1, 
            n_inner=4*64,
            resid_pdrop=0.3,
            attn_pdrop=0.3,
            action_tanh=False
        )
        
        model = DecisionTransformerModel(config)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        
        print("Model loaded successfully")
        return model
    
    def _fetch_benchmark_data(self, start_date, end_date):
        """
        Fetch S&P DBM Index benchmark data.
        If DBM is not available, use Bitcoin ETF (BITO) or BTC-USD as proxy.
        """
        print(f"Fetching benchmark data for period: {start_date} to {end_date}")
        
        benchmark_tickers = ['BTC-USD', 'GBTC', 'BITO', 'DBM']
        
        for ticker in benchmark_tickers:
            try:
                print(f"Trying to fetch {ticker} data...")
                
                start_with_buffer = start_date - timedelta(days=7)
                end_with_buffer = end_date + timedelta(days=7)
                
                benchmark_data = yf.download(ticker, start=start_with_buffer, end=end_with_buffer, interval='1d', progress=False)
                
                if not benchmark_data.empty and 'Close' in benchmark_data.columns:
                   
                    mask = (benchmark_data.index.date >= start_date) & (benchmark_data.index.date <= end_date)
                    filtered_data = benchmark_data[mask]
                    
                    if not filtered_data.empty:
                        print(f"Successfully fetched {ticker} data ({len(filtered_data)} data points)")
                        
                        close_data = filtered_data['Close']
                        if hasattr(close_data, 'squeeze'):
                            close_data = close_data.squeeze()
                        return close_data, ticker
                    else:
                        print(f" {ticker} data fetched but no data in date range")
                else:
                    print(f" {ticker} data empty or missing 'Close' column")
                    
            except Exception as e:
                print(f" Failed to fetch {ticker}: {e}")
                continue
        
        
        print(" No benchmark data available - proceeding without benchmark comparison")
        return None, "No Benchmark"
    
    def simulate_trading(self, initial_capital=10000, debug=False):
        """
        Simulate trading using the trained model.
        """
        print(f"Starting trading simulation with ${initial_capital:,} initial capital...")
        
        capital = initial_capital
        position = 0.0  # Number of BTC held
        portfolio_values = [capital]
        actions_taken = []
        rewards = []
        timestamps_used = []
        model_predictions = [] 
        
        with torch.no_grad():
            for t in range(len(self.processed_states)):
                current_state = self.processed_states[t]
                current_price = self.prices[t]
                current_time = self.timestamps[t]
                
                discrete_action = 1  # Default to hold
                
                if t == 0:
                    discrete_action = 1  # Start with hold
                else:
                    # Use model to predict action
                    seq_len = min(t + 1, self.context_length)
                    state_seq = self.processed_states[max(0, t-seq_len+1):t+1]
                    
                    # Pad sequence if needed
                    if len(state_seq) < self.context_length:
                        padding = np.zeros((self.context_length - len(state_seq), state_seq.shape[1]))
                        state_seq = np.vstack([padding, state_seq])
                    
                    states_tensor = torch.from_numpy(state_seq).float().unsqueeze(0).to(self.device)
                    actions_tensor = torch.zeros(1, self.context_length, 3).to(self.device)
                    
                    # Fill in previous actions if available
                    if t > 0:
                        for i in range(min(t, self.context_length)):
                            if i < len(actions_taken):
                                prev_action = int(actions_taken[-(i+1)])  # Ensure integer
                                actions_tensor[0, self.context_length-1-i, prev_action] = 1.0
                                
                    returns_to_go_tensor = torch.full((1, self.context_length, 1), self.target_return).float().to(self.device)
                    timesteps_tensor = torch.arange(self.context_length).long().unsqueeze(0).to(self.device)
                    attention_mask = torch.ones(1, self.context_length, device=self.device, dtype=torch.long)
                    
                    try:
                        actions_in = torch.zeros_like(actions_tensor)
                        actions_in[:, 1:, :] = actions_tensor[:, :-1, :]
                        
                        action_preds = self.model(
                            states=states_tensor,
                            actions=actions_in,
                            returns_to_go=returns_to_go_tensor,
                            timesteps=timesteps_tensor,
                            attention_mask=attention_mask
                        ).action_preds
                        
                        # Get the last action prediction
                        action_logits = action_preds[0, -1].cpu().numpy()
                        action_probs = np.exp(action_logits) / np.sum(np.exp(action_logits))
                        discrete_action = int(np.argmax(action_probs))  # Ensure integer
                        
                        model_predictions.append({
                            'step': t,
                            'predicted_action': discrete_action,
                            'probabilities': action_probs,
                            'logits': action_logits
                        })
                        
                        if debug and t < 10:
                            print(f"Step {t}: Model predicted action={discrete_action}")
                            print(f"  Probs: Sell={action_probs[0]:.3f}, Hold={action_probs[1]:.3f}, Buy={action_probs[2]:.3f}")
                        
                    except Exception as e:
                        if debug:
                            print(f"Model prediction error at step {t}: {e}")
                        discrete_action = 1  # Default to hold
                
                # Execute trading action
                executed_action = 1  # Default to hold
                
                if discrete_action == 2:  # Buy
                    buy_amount = min(capital * 0.1, capital)  # Buy up to 10% of capital
                    if buy_amount > 0:
                        btc_bought = buy_amount / current_price
                        position += btc_bought
                        capital -= buy_amount
                        executed_action = 2
                        if debug and t < 10:
                            print(f"  Executed: BUY {btc_bought:.6f} BTC for ${buy_amount:.2f}")
                    else:
                        executed_action = 1  # Hold if can't buy
                        if debug and t < 10:
                            print(f"  Executed: HOLD (insufficient capital)")
                        
                elif discrete_action == 0:  # Sell
                    sell_btc = min(position * 0.1, position)  # Sell up to 10% of position
                    if sell_btc > 0:
                        capital += sell_btc * current_price
                        position -= sell_btc
                        executed_action = 0
                        if debug and t < 10:
                            print(f"  Executed: SELL {sell_btc:.6f} BTC for ${sell_btc * current_price:.2f}")
                    else:
                        executed_action = 1  # Hold if nothing to sell
                        if debug and t < 10:
                            print(f"  Executed: HOLD (no position to sell)")
                        
                else:  # Hold (action == 1)
                    executed_action = 1
                    if debug and t < 10:
                        print(f"  Executed: HOLD")
                
                actions_taken.append(executed_action)
                
                # Calculate portfolio value
                portfolio_value = capital + position * current_price
                portfolio_values.append(portfolio_value)
                timestamps_used.append(current_time)
                
                # Calculate reward
                if t > 0:
                    reward = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                    rewards.append(reward)
                    self.target_return -= reward
        
        # Remove initial portfolio value to align with timestamps
        portfolio_values = portfolio_values[1:]
        
        print(f"Trading simulation completed. Final portfolio value: ${portfolio_values[-1]:,.2f}")
        
        # Analyze model predictions vs executed actions
        model_action_counts = {}
        executed_action_counts = {}
        
        if len(model_predictions) > 0:
            model_actions = [p['predicted_action'] for p in model_predictions]
            for action in [0, 1, 2]:
                model_action_counts[action] = model_actions.count(action)
                executed_action_counts[action] = actions_taken.count(action)
        
        return {
            'portfolio_values': np.array(portfolio_values),
            'actions': actions_taken,
            'rewards': rewards,
            'prices': self.prices,
            'timestamps': timestamps_used,
            'initial_capital': initial_capital,
            'model_predictions': model_predictions,
            'model_action_counts': model_action_counts,
            'executed_action_counts': executed_action_counts
        }
    
    def calculate_performance_metrics(self, portfolio_values, timestamps, benchmark_values=None):
        """
        Calculate comprehensive performance metrics.
        """
        portfolio_values = np.array(portfolio_values)
        
        # Basic metrics
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Daily returns
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Annualized metrics (assuming daily data)
        mean_daily_return = np.mean(daily_returns)
        std_daily_return = np.std(daily_returns)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = mean_daily_return / (std_daily_return + 1e-8) * np.sqrt(252)  # Annualized
        
        # Maximum drawdown
        cumulative_returns = portfolio_values / portfolio_values[0]
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Volatility (annualized)
        annual_volatility = std_daily_return * np.sqrt(252)
        
        # Calmar ratio
        calmar_ratio = (total_return * 252 / len(daily_returns)) / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = np.sum(daily_returns > 0) / len(daily_returns) if len(daily_returns) > 0 else 0
        
        metrics = {
            'Total Return': total_return,
            'Annualized Return': total_return * 252 / len(daily_returns),
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Annual Volatility': annual_volatility,
            'Calmar Ratio': calmar_ratio,
            'Win Rate': win_rate,
            'Final Value': final_value,
            'Total Days': len(portfolio_values)
        }
        
        return metrics
    
    def evaluate_with_benchmark(self, plots_dir='plots', save_plots=True):
        """
        Run complete evaluation with benchmark comparison.
        """
        print("\n" + "="*60)
        print("CRYPTO DECISION TRANSFORMER EVALUATION")
        print("="*60)
        
        # Run trading simulation
        trading_results = self.simulate_trading()
        
        # Get benchmark data n
        start_date = self.timestamps.iloc[0].date()
        end_date = self.timestamps.iloc[-1].date()
        
        try:
            benchmark_data, benchmark_name = self._fetch_benchmark_data(start_date, end_date)
            
            initial_portfolio_value = trading_results['initial_capital']
            
            if hasattr(benchmark_data, '__len__'):
                benchmark_length = len(benchmark_data)
            else:
                benchmark_data = pd.Series([benchmark_data], index=[start_date])
                benchmark_length = 1
            
            if benchmark_length > 0:
                first_value = benchmark_data.iloc[0]
                if hasattr(first_value, 'item'):  
                    first_value = first_value.item()
                benchmark_normalized = benchmark_data / first_value * initial_portfolio_value
                
                portfolio_timestamps = pd.to_datetime(trading_results['timestamps'])
                
                if len(benchmark_normalized) == 1:
                    benchmark_values = np.full(len(portfolio_timestamps), benchmark_normalized.iloc[0])
                else:
                    benchmark_df = pd.DataFrame({
                        'value': benchmark_normalized,
                        'date': benchmark_normalized.index.date
                    })
                    
                    portfolio_df = pd.DataFrame({
                        'timestamp': portfolio_timestamps,
                        'date': portfolio_timestamps.date
                    })
                    
                    merged = portfolio_df.merge(benchmark_df, on='date', how='left')
                    benchmark_values = merged['value'].ffill().bfill().values
                
                if np.isnan(benchmark_values).any():
                    benchmark_series = pd.Series(benchmark_values)
                    benchmark_values = benchmark_series.interpolate().bfill().ffill().values
                
                print(f"📊 Aligned benchmark data: {len(benchmark_values)} points (NaN count: {np.isnan(benchmark_values).sum()})")
            else:
                benchmark_values = None
                benchmark_name = "No Benchmark"
                
        except Exception as e:
            print(f"Error fetching benchmark: {e}")
            benchmark_values = None
            benchmark_name = "No Benchmark"
        
        # Calculate performance metrics
        portfolio_metrics = self.calculate_performance_metrics(
            trading_results['portfolio_values'], 
            trading_results['timestamps'],
            benchmark_values
        )
        
        if benchmark_values is not None:
            benchmark_metrics = self.calculate_performance_metrics(
                benchmark_values, 
                trading_results['timestamps']
            )
        else:
            benchmark_metrics = None
        
        # Display results
        self._display_results(portfolio_metrics, benchmark_metrics, benchmark_name, trading_results)
        
        # Create plots
        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)
            self._create_comprehensive_plots(
                trading_results, 
                benchmark_values, 
                benchmark_name, 
                portfolio_metrics, 
                benchmark_metrics,
                plots_dir
            )
        
        return {
            'trading_results': trading_results,
            'portfolio_metrics': portfolio_metrics,
            'benchmark_metrics': benchmark_metrics,
            'benchmark_name': benchmark_name
        }
    
    def _display_results(self, portfolio_metrics, benchmark_metrics, benchmark_name, trading_results):
        """Display evaluation results."""
        print(f"\n PERFORMANCE SUMMARY")
        print("-" * 50)
        
        print(f" Crypto Decision Transformer Performance:")
        for metric, value in portfolio_metrics.items():
            if metric in ['Total Return', 'Annualized Return', 'Max Drawdown', 'Annual Volatility', 'Win Rate']:
                print(f"   {metric}: {value:.2%}")
            elif metric in ['Sharpe Ratio', 'Calmar Ratio']:
                print(f"   {metric}: {value:.3f}")
            elif metric == 'Final Value':
                print(f"   {metric}: ${value:,.2f}")
            else:
                print(f"   {metric}: {value}")
        
        if benchmark_metrics:
            print(f"\n {benchmark_name} Benchmark Performance:")
            for metric, value in benchmark_metrics.items():
                if metric in ['Total Return', 'Annualized Return', 'Max Drawdown', 'Annual Volatility', 'Win Rate']:
                    print(f"   {metric}: {value:.2%}")
                elif metric in ['Sharpe Ratio', 'Calmar Ratio']:
                    print(f"   {metric}: {value:.3f}")
                elif metric == 'Final Value':
                    print(f"   {metric}: ${value:,.2f}")
        
        # Trading statistics
        print(f"\n Trading Statistics:")
        actions = trading_results['actions']
        buy_actions = sum(1 for a in actions if a == 2)
        sell_actions = sum(1 for a in actions if a == 0)
        hold_actions = sum(1 for a in actions if a == 1)
        
        print(f"   Total Actions: {len(actions)}")
        print(f"   Buy Actions: {buy_actions} ({buy_actions/len(actions):.1%})")
        print(f"   Sell Actions: {sell_actions} ({sell_actions/len(actions):.1%})")
        print(f"   Hold Actions: {hold_actions} ({hold_actions/len(actions):.1%})")
        print(f"   Total Trades: {buy_actions + sell_actions}")
        
        # Model predictions vs executed actions analysis
        if 'model_action_counts' in trading_results and 'executed_action_counts' in trading_results:
            print(f"\n Model Prediction Analysis:")
            model_counts = trading_results['model_action_counts']
            executed_counts = trading_results['executed_action_counts']
            
            if model_counts:
                total_predictions = sum(model_counts.values())
                print(f"   Model Predictions:")
                print(f"     Sell: {model_counts.get(0, 0)} ({model_counts.get(0, 0)/total_predictions:.1%})")
                print(f"     Hold: {model_counts.get(1, 0)} ({model_counts.get(1, 0)/total_predictions:.1%})")
                print(f"     Buy: {model_counts.get(2, 0)} ({model_counts.get(2, 0)/total_predictions:.1%})")
                
                print(f"   Executed Actions:")
                total_executed = len(actions)
                print(f"     Sell: {executed_counts.get(0, 0)} ({executed_counts.get(0, 0)/total_executed:.1%})")
                print(f"     Hold: {executed_counts.get(1, 0)} ({executed_counts.get(1, 0)/total_executed:.1%})")
                print(f"     Buy: {executed_counts.get(2, 0)} ({executed_counts.get(2, 0)/total_executed:.1%})")
                
                # Calculate prediction vs execution discrepancy
                if 'model_predictions' in trading_results and len(trading_results['model_predictions']) > 0:
                    model_actions = [p['predicted_action'] for p in trading_results['model_predictions']]
                    executed_actions = actions[1:] 
                    
                    discrepancies = sum([1 for m, e in zip(model_actions, executed_actions) if m != e])
                    print(f"   Prediction vs Execution discrepancies: {discrepancies}/{len(model_actions)} ({discrepancies/len(model_actions)*100:.1f}%)")
    
    def _create_comprehensive_plots(self, trading_results, benchmark_values, benchmark_name, 
                                  portfolio_metrics, benchmark_metrics, plots_dir):
        """Create comprehensive visualization plots."""
        fig = plt.figure(figsize=(20, 16))
        
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Portfolio value vs benchmark with cumulative returns on secondary y-axis
        ax1 = fig.add_subplot(gs[0, :2])
        timestamps = trading_results['timestamps']
        portfolio_values = trading_results['portfolio_values']
        
        # Plot portfolio values on primary y-axis
        ax1.plot(timestamps, portfolio_values, linewidth=2, label='Crypto DT Strategy', color='blue')
        
        if benchmark_values is not None:
            ax1.plot(timestamps, benchmark_values, linewidth=2, label=f'{benchmark_name} Benchmark', 
                    color='red', linestyle='--', alpha=0.8)
        
        initial_capital = trading_results['initial_capital']
        initial_price = trading_results['prices'][0]
        buy_hold_values = [initial_capital * (price / initial_price) for price in trading_results['prices']]

        ax1_twin = ax1.twinx()
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
        
        ax1_twin.plot(timestamps[1:], cumulative_returns, linewidth=2, color='blue', alpha=0.7, label='Cumulative Returns')
        ax1_twin.set_ylabel('Cumulative Returns', color='blue')
        ax1_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax1_twin.tick_params(axis='y', labelcolor='blue')
        
        ax1.set_title('Portfolio Performance & Cumulative Returns', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value (Million $)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        ax1_twin.legend(loc='upper right')
        

        # ax2 = fig.add_subplot(gs[0, 2])
        
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(timestamps, trading_results['prices'], alpha=0.7, color='orange', linewidth=1)
        ax3.set_ylabel('BTC Price ($)', color='orange')
        ax3.tick_params(axis='y', labelcolor='orange')
        
        ax3_twin = ax3.twinx()
        actions = trading_results['actions']
        colors = ['red' if a == 0 else 'green' if a == 2 else 'gray' for a in actions]
        ax3_twin.scatter(timestamps, actions, c=colors, alpha=0.6, s=15)
        ax3_twin.set_ylabel('Action (0: Sell, 1: Hold, 2: Buy)')
        ax3_twin.set_ylim(-0.5, 2.5)
        
        ax3.set_title('BTC Price and Trading Actions', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Returns distribution
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.hist(portfolio_returns, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Daily Return')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        ax4.axvline(np.mean(portfolio_returns), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(portfolio_returns):.3f}')
        ax4.legend()
        
        # Drawdown chart
        ax5 = fig.add_subplot(gs[2, :2])
        cumulative = portfolio_values / portfolio_values[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        
        ax5.fill_between(timestamps, drawdowns, 0, alpha=0.3, color='red')
        ax5.plot(timestamps, drawdowns, color='red', linewidth=1)
        ax5.set_title('Drawdown Chart', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Drawdown')
        ax5.set_xlabel('Date')
        ax5.grid(True, alpha=0.3)
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        # Performance metrics table
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('tight')
        ax6.axis('off')
        
        # Create performance comparison table
        metrics_data = []
        for metric in ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Annual Volatility']:
            row = [metric]
            
            # Portfolio value
            if metric in ['Total Return', 'Max Drawdown', 'Annual Volatility']:
                row.append(f"{portfolio_metrics[metric]:.2%}")
            else:
                row.append(f"{portfolio_metrics[metric]:.3f}")
            
            # Benchmark value
            if benchmark_metrics:
                if metric in ['Total Return', 'Max Drawdown', 'Annual Volatility']:
                    row.append(f"{benchmark_metrics[metric]:.2%}")
                else:
                    row.append(f"{benchmark_metrics[metric]:.3f}")
            else:
                row.append("N/A")
            
            metrics_data.append(row)
        
        headers = ['Metric', 'Crypto DT', f'{benchmark_name}']
        table = ax6.table(cellText=metrics_data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax6.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
        
        # Format x-axis for all time-based plots
        for ax in [ax1, ax3, ax5]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(timestamps)//20)))
            fig.autofmt_xdate()
        
        fig.suptitle('Decision Transformer for Crypto Trading', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Save plots
        evaluation_plot_path = os.path.join(plots_dir, 'crypto_evaluation_comprehensive.png')
        plt.savefig(evaluation_plot_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive evaluation plot saved to: {evaluation_plot_path}")
        
        plt.close()
        
        self._create_individual_plots(trading_results, benchmark_values, benchmark_name, plots_dir)
    

    def _create_individual_plots(self, trading_results, benchmark_values, benchmark_name, plots_dir):
        """Create individual plots for specific analysis."""
        
        # Portfolio comparison plot with cumulative returns
        fig, ax1 = plt.subplots(figsize=(15, 6))
        timestamps = trading_results['timestamps']
        portfolio_values = trading_results['portfolio_values']
        
        ax1.plot(timestamps, portfolio_values, linewidth=3, label='DT Strategy', color='blue')
        
        if benchmark_values is not None:
            ax1.plot(timestamps, benchmark_values, linewidth=3, label=f'{benchmark_name} Benchmark', 
                    color='red', linestyle='--', alpha=0.8)
        
        initial_capital = trading_results['initial_capital']
        initial_price = trading_results['prices'][0]
        
        ax1_twin = ax1.twinx()
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        cumulative_returns = np.cumprod(1 + portfolio_returns) - 1
        
        ax1_twin.plot(timestamps[1:], cumulative_returns, linewidth=2, color='blue', alpha=0.7, label='Cumulative Returns')
        ax1_twin.set_ylabel('Cumulative Returns', color='blue', fontsize=15)
        ax1_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax1_twin.tick_params(axis='y', labelcolor='blue')
        
        ax1.set_title('Decision Transformer vs Benchmarks', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Time', fontsize=15)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=15)
        ax1.legend(loc='upper left', fontsize=15)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        ax1_twin.legend(loc='upper right', fontsize=15)
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(30, len(timestamps)//50)))
        ax1.xaxis.set_minor_locator(mdates.MinuteLocator(interval=max(10, len(timestamps)//150)))
        ax1.grid(True, which='major', alpha=0.5)
        ax1.grid(True, which='minor', alpha=0.2)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        portfolio_plot_path = os.path.join(plots_dir, 'crypto_portfolio_comparison.png')
        plt.savefig(portfolio_plot_path, dpi=300, bbox_inches='tight')
        print(f"Portfolio comparison plot saved to: {portfolio_plot_path}")
        plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Crypto Decision Transformer with benchmark comparison')
    parser.add_argument('--model_path', type=str, 
                       default='./trained_models/decision_transformer.pth',
                       help='Path to the trained model')
    parser.add_argument('--test_data_path', type=str,
                       default='./offline_data_preparation/data/BTC_1sec_with_sentiment_risk_test.csv',
                       help='Path to test data CSV file')
    parser.add_argument('--max_samples', type=int, default=10000,
                       help='Maximum number of samples for evaluation')
    parser.add_argument('--target_return', type=float, default=250.0,
                       help='Target return for the model')
    parser.add_argument('--context_length', type=int, default=20,
                       help='Context length for the model')
    parser.add_argument('--plots_dir', type=str, default='plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create plots directory
    os.makedirs(args.plots_dir, exist_ok=True)
    
    try:
        evaluator = CryptoEvaluator(
            model_path=args.model_path,
            test_data_path=args.test_data_path,
            max_samples=args.max_samples,
            target_return=args.target_return,
            context_length=args.context_length
        )
        
        # Run evaluation
        results = evaluator.evaluate_with_benchmark(plots_dir=args.plots_dir)
        
        print(f"\n Evaluation completed successfully!")
        print(f"Plots saved to: {args.plots_dir}")
        
        return results
    except Exception as e:
        print(f" Evaluation failed: {e}")
        raise


if __name__ == '__main__':
    results = main()