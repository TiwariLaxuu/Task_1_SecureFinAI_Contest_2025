#!/usr/bin/env python3
"""
Convert Replay Buffer data from .pth format to offline trajectory dataset CSV format

The CSV format has columns: state, action, reward, episode_start
- state: array of state features (as string representation)
- action: single action value
- reward: single reward value  
- episode_start: boolean indicating if this is the start of a new episode (1/0)
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Tuple, List


def load_replay_buffer_data(replay_buffer_dir: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load replay buffer data from .pth files
    
    Args:
        replay_buffer_dir: Directory containing replay_buffer_*.pth files
        
    Returns:
        Tuple of (states, actions, rewards, undones) tensors
    """
    print(f"Loading replay buffer data from: {replay_buffer_dir}")
    
    # Load the saved replay buffer components
    states = torch.load(os.path.join(replay_buffer_dir, "replay_buffer_states.pth"))
    actions = torch.load(os.path.join(replay_buffer_dir, "replay_buffer_actions.pth"))
    rewards = torch.load(os.path.join(replay_buffer_dir, "replay_buffer_rewards.pth"))
    undones = torch.load(os.path.join(replay_buffer_dir, "replay_buffer_undones.pth"))
    
    print(f"Loaded data shapes:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Undones: {undones.shape}")
    
    return states, actions, rewards, undones


def identify_episode_boundaries(undones: torch.Tensor, max_episodes: int = None) -> List[List[Tuple[int, int]]]:
    """
    Identify episode boundaries from undones tensor.
    undones[t] = 1.0 means the episode continues, 0.0 means episode ended
    
    Args:
        undones: Tensor of shape [max_size, num_seqs] indicating episode continuation
        max_episodes: Maximum number of episodes to return (for memory efficiency)
        
    Returns:
        List of episodes, where each episode is a list of (step_idx, seq_idx) tuples
    """
    max_size, num_seqs = undones.shape
    all_episodes = []
    
    print(f"Processing {num_seqs} sequences with {max_size} steps each...")
    
    for seq_idx in range(num_seqs):
        if seq_idx % 100 == 0:
            print(f"Processing sequence {seq_idx}/{num_seqs}")
            
        episodes_in_seq = []
        current_episode = []
        
        for step_idx in range(max_size):
            current_episode.append((step_idx, seq_idx))
            
            # If undone is 0, this step ends the episode
            if undones[step_idx, seq_idx] == 0.0:
                if len(current_episode) > 1:  # Only keep episodes with more than 1 step
                    episodes_in_seq.append(current_episode)
                current_episode = []
        
        # Add any remaining episode
        if len(current_episode) > 1:
            episodes_in_seq.append(current_episode)
            
        all_episodes.extend(episodes_in_seq)
        
        # Early stopping if we have enough episodes
        if max_episodes and len(all_episodes) >= max_episodes:
            all_episodes = all_episodes[:max_episodes]
            print(f"Reached max episodes limit ({max_episodes}), stopping early")
            break
    
    print(f"Found {len(all_episodes)} episodes across {min(seq_idx + 1, num_seqs)} sequences")
    if all_episodes:
        episode_lengths = [len(ep) for ep in all_episodes]
        print(f"Episode length stats: min={min(episode_lengths)}, max={max(episode_lengths)}, avg={np.mean(episode_lengths):.1f}")
    
    return all_episodes


def convert_to_csv_format(states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, 
                         episodes: List[List[Tuple[int, int]]]) -> pd.DataFrame:
    """
    Convert replay buffer data to CSV format
    
    Args:
        states: State tensor [max_size, num_seqs, state_dim]
        actions: Action tensor [max_size, num_seqs, action_dim] 
        rewards: Reward tensor [max_size, num_seqs]
        episodes: List of episodes, each episode is list of (step_idx, seq_idx) tuples
        
    Returns:
        DataFrame with columns: state, action, reward, episode_start
    """
    data_rows = []
    
    for episode_idx, episode in enumerate(episodes):
        if episode_idx % 1000 == 0:
            print(f"Processing episode {episode_idx}/{len(episodes)}")
            
        for step_num, (step_idx, seq_idx) in enumerate(episode):
            # Get state, action, reward for this step
            state = states[step_idx, seq_idx].cpu().numpy()
            action = actions[step_idx, seq_idx].cpu().numpy()
            reward = rewards[step_idx, seq_idx].cpu().numpy()
            
            # Handle action dimension - if discrete, take the single value
            if action.ndim > 0 and len(action) == 1:
                action = action[0]
            elif action.ndim > 0:
                # For multi-dimensional actions, take the first dimension
                action = action[0]
            
            # Mark episode start
            episode_start = 1 if step_num == 0 else 0
            
            # Convert state to string format matching the CSV
            state_str = str(state.tolist())
            
            data_rows.append({
                'state': state_str,
                'action': float(action),
                'reward': float(reward),
                'episode_start': episode_start
            })
    
    print(f"Created {len(data_rows)} trajectory steps")
    return pd.DataFrame(data_rows)


def calculate_returns_to_go(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates return-to-go for each step in each episode.
    
    Args:
        df: DataFrame with columns: state, action, reward, episode_start
        
    Returns:
        DataFrame with additional return_to_go column
    """
    df = df.copy()
    df['return_to_go'] = 0.0
    
    # Reset index so we can use safe row positions
    df = df.reset_index(drop=True)
    
    # Get the integer positions where each episode starts
    episode_starts = df.index[df['episode_start'] == True].tolist()
    
    for i in range(len(episode_starts)):
        start = episode_starts[i]
        end = episode_starts[i + 1] if i + 1 < len(episode_starts) else len(df)
        
        rewards = df.loc[start:end - 1, 'reward'].values
        rtg = np.cumsum(rewards[::-1])[::-1]
        
        # Use .loc with the correct slice, which matches rtg length exactly
        df.loc[start:end - 1, 'return_to_go'] = rtg
    
    return df


def create_crypto_trajectories_dataset(replay_buffer_dir: str, output_file: str, max_episodes: int = None):
    """
    Main function to convert replay buffer to trajectory dataset
    
    Args:
        replay_buffer_dir: Directory containing replay buffer .pth files
        output_file: Output CSV file path
        max_episodes: Maximum number of episodes to include (optional)
    """
    # Check if replay buffer files exist
    required_files = ['replay_buffer_states.pth', 'replay_buffer_actions.pth', 
                     'replay_buffer_rewards.pth', 'replay_buffer_undones.pth']
    
    for file in required_files:
        file_path = os.path.join(replay_buffer_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    print("Starting replay buffer to CSV conversion...")
    
    # Load replay buffer data
    states, actions, rewards, undones = load_replay_buffer_data(replay_buffer_dir)
    
    # Identify episode boundaries  
    episodes = identify_episode_boundaries(undones, max_episodes=max_episodes)
    
    # Convert to CSV format
    df = convert_to_csv_format(states, actions, rewards, episodes)
    
    # Save to CSV
    print(f"Saving dataset to: {output_file}")
    df.to_csv(output_file, index=False)
    
    print(f"Conversion complete!")
    print(f"Final dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Episode starts: {df['episode_start'].sum()} episodes")
    
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay_buffer_dir", type=str, required=True, default="./TradeSimulator-v0_D3QN_0")
    parser.add_argument("--output_file", type=str, required=True, default="../crypto_trajectories.csv")
        parser.add_argument("--dt_ready_output_file", type=str, default="../crypto_decision_transformer_ready_dataset.csv",
                        help="Path to save the decision transformer ready dataset with return-to-go")
    parser.add_argument("--max_episodes", type=int, default=100, help="Maximum number of episodes to include (optional)")
    args = parser.parse_args()
    
    # Convert replay buffer to trajectory dataset
    df = create_crypto_trajectories_dataset(
        replay_buffer_dir=args.replay_buffer_dir,
        output_file=args.output_file,
        max_episodes=args.max_episodes
    )
    
    # Calculate return-to-go and save decision transformer ready dataset
    print(f"\nCalculating return-to-go values...")
    df_for_dt = calculate_returns_to_go(df)
    print("Saving decision transformer ready dataset to Crypto_Trading/crypto_decision_transformer_ready_dataset.csv")
    df_for_dt.to_csv("../crypto_decision_transformer_ready_dataset.csv", index=False)