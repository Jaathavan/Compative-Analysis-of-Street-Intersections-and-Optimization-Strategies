"""
Train PPO Agent for Adaptive Signal Control

Uses Stable-Baselines3 to train a PPO agent that learns to optimize
traffic signal timing based on real-time traffic conditions.

Features:
- Training with multiple parallel environments
- Evaluation callback for best model selection
- TensorBoard logging
- Checkpointing
- Curriculum learning (optional)

Usage:
    python train_ppo.py --timesteps 500000 --eval-freq 10000
    python train_ppo.py --timesteps 1000000 --curriculum --use-gpu
"""

import argparse
import os
import sys
from datetime import datetime
import numpy as np

# Stable-Baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Import our environment
from ppo_environment import SignalControlEnv


def make_env(config_path: str, sumo_cfg_path: str, demand_multiplier: float = 1.0, rank: int = 0):
    """
    Create a single environment instance.
    
    Args:
        config_path: Path to config.yaml
        sumo_cfg_path: Path to SUMO config
        demand_multiplier: Demand scaling factor
        rank: Environment rank (for parallel envs)
    
    Returns:
        Environment creation function
    """
    def _init():
        env = SignalControlEnv(
            config_path=config_path,
            sumo_cfg_path=sumo_cfg_path,
            use_gui=False,
            demand_multiplier=demand_multiplier,
            episode_length=3600  # 1 hour episodes
        )
        env = Monitor(env)
        return env
    
    return _init


def train_ppo(
    config_path: str,
    sumo_cfg_path: str,
    total_timesteps: int = 500000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    eval_freq: int = 10000,
    save_freq: int = 50000,
    curriculum: bool = False,
    use_gpu: bool = False,
    output_dir: str = '../models',
    log_dir: str = '../logs'
):
    """
    Train PPO agent for signal control.
    
    Args:
        config_path: Path to config.yaml
        sumo_cfg_path: Path to SUMO config
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        learning_rate: Learning rate
        n_steps: Steps per environment per update
        batch_size: Minibatch size
        n_epochs: Optimization epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        eval_freq: Evaluation frequency (timesteps)
        save_freq: Checkpoint save frequency (timesteps)
        curriculum: Use curriculum learning
        use_gpu: Use GPU if available
        output_dir: Model output directory
        log_dir: TensorBoard log directory
    """
    print("\n" + "="*70)
    print("PPO TRAINING FOR ADAPTIVE SIGNAL CONTROL")
    print("="*70)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"ppo_signal_{timestamp}"
    
    print(f"\nTraining Configuration:")
    print(f"   Run name: {run_name}")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Parallel environments: {n_envs}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Device: {'GPU' if use_gpu else 'CPU'}")
    print(f"   Curriculum learning: {curriculum}")
    
    # Create training environments
    print(f"\n🏗️  Creating {n_envs} parallel environments...")
    
    if curriculum:
        # Start with lower demand, gradually increase
        demand_schedule = [0.5, 0.75, 1.0, 1.25]
        current_demand_idx = 0
        demand_multiplier = demand_schedule[current_demand_idx]
        print(f"Curriculum: Starting with {demand_multiplier}× demand")
    else:
        demand_multiplier = 1.0
        print(f"   Demand multiplier: {demand_multiplier}×")
    
    train_env = SubprocVecEnv([
        make_env(config_path, sumo_cfg_path, demand_multiplier, i) 
        for i in range(n_envs)
    ])
    
    # Create evaluation environment
    print("   Creating evaluation environment...")
    eval_env = DummyVecEnv([
        make_env(config_path, sumo_cfg_path, 1.0, 0)  # Always evaluate at baseline demand
    ])
    
    # Create PPO model
    print("\nInitializing PPO model...")
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, run_name),
        device='cuda' if use_gpu else 'cpu'
    )
    
    print("   Model architecture:")
    print(f"   - Policy: MLP (Multi-Layer Perceptron)")
    print(f"   - Observation space: {train_env.observation_space.shape}")
    print(f"   - Action space: {train_env.action_space}")
    print(f"   - Parameters: ~{sum(p.numel() for p in model.policy.parameters()):,}")
    
    # Create callbacks
    print("\nSetting up callbacks...")
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(output_dir, run_name),
        log_path=os.path.join(log_dir, run_name, 'eval'),
        eval_freq=eval_freq // n_envs,  # Adjust for parallel envs
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=os.path.join(output_dir, run_name, 'checkpoints'),
        name_prefix='ppo_checkpoint',
        verbose=1
    )
    
    # Combine callbacks
    callback_list = CallbackList([eval_callback, checkpoint_callback])
    
    # Curriculum learning callback (if enabled)
    if curriculum:
        class CurriculumCallback:
            def __init__(self, demand_schedule, update_freq):
                self.demand_schedule = demand_schedule
                self.update_freq = update_freq
                self.current_idx = 0
            
            def __call__(self, locals_, globals_):
                if locals_['self'].num_timesteps % self.update_freq == 0:
                    if self.current_idx < len(self.demand_schedule) - 1:
                        self.current_idx += 1
                        new_demand = self.demand_schedule[self.current_idx]
                        print(f"\n📚 Curriculum: Increasing to {new_demand}× demand")
                        # Note: Would need to recreate environments with new demand
                return True
    
    # Start training
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"\nTensorBoard: tensorboard --logdir {log_dir}")
    print(f"Monitor training in real-time!\n")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            progress_bar=True
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        
        # Save final model
        final_model_path = os.path.join(output_dir, f'{run_name}_final.zip')
        model.save(final_model_path)
        print(f"\nFinal model saved: {final_model_path}")
        
        # Save best model with standard name
        best_model_src = os.path.join(output_dir, run_name, 'best_model.zip')
        best_model_dst = os.path.join(output_dir, 'ppo_signal_best.zip')
        if os.path.exists(best_model_src):
            import shutil
            shutil.copy(best_model_src, best_model_dst)
            print(f"Best model saved: {best_model_dst}")
        
        # Print training summary
        print("\n" + "="*70)
        print("TRAINING SUMMARY")
        print("="*70)
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Training time: {model.num_timesteps / total_timesteps * 100:.1f}% complete")
        print(f"Final learning rate: {model.learning_rate}")
        print(f"\nModel files:")
        print(f"  - Final: {final_model_path}")
        print(f"  - Best: {best_model_dst}")
        print(f"  - Checkpoints: {os.path.join(output_dir, run_name, 'checkpoints')}")
        print(f"\nLogs:")
        print(f"  - TensorBoard: {os.path.join(log_dir, run_name)}")
        print(f"  - Evaluation: {os.path.join(log_dir, run_name, 'eval')}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        print("Saving current model...")
        interrupted_path = os.path.join(output_dir, f'{run_name}_interrupted.zip')
        model.save(interrupted_path)
        print(f"📦 Model saved: {interrupted_path}")
    
    finally:
        # Cleanup
        train_env.close()
        eval_env.close()
        print("\nEnvironments closed")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train PPO agent for signal control')
    
    # Paths
    parser.add_argument('--config', type=str, 
                       default='../config/config.yaml',
                       help='Path to config.yaml')
    parser.add_argument('--sumo-cfg', type=str,
                       default='../quickstart_output/sumo_configs/webster/intersection.sumocfg',
                       help='Path to SUMO config')
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel environments')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='Steps per environment per update')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Minibatch size')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='Optimization epochs per update')
    
    # PPO hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE lambda')
    parser.add_argument('--clip-range', type=float, default=0.2,
                       help='PPO clip range')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                       help='Entropy coefficient')
    
    # Callbacks
    parser.add_argument('--eval-freq', type=int, default=10000,
                       help='Evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=50000,
                       help='Checkpoint save frequency')
    
    # Options
    parser.add_argument('--curriculum', action='store_true',
                       help='Use curriculum learning')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU if available')
    
    # Directories
    parser.add_argument('--output-dir', type=str, default='../models',
                       help='Model output directory')
    parser.add_argument('--log-dir', type=str, default='../logs',
                       help='TensorBoard log directory')
    
    args = parser.parse_args()
    
    # Check if SUMO config exists
    if not os.path.exists(args.sumo_cfg):
        print(f"❌ SUMO config not found: {args.sumo_cfg}")
        print("   Run quickstart.py first to generate SUMO configuration!")
        sys.exit(1)
    
    # Train
    train_ppo(
        config_path=args.config,
        sumo_cfg_path=args.sumo_cfg,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        curriculum=args.curriculum,
        use_gpu=args.use_gpu,
        output_dir=args.output_dir,
        log_dir=args.log_dir
    )


if __name__ == '__main__':
    main()
