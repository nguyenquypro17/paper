# MiniGrid-DoorKey-6x6-v0:

## Stage 1: Train & Test PPO Model + Rollout data
python train_doorkey_6x6.py

python test_ppo_doorkey_6x6.py \
    --model_path ppo_doorkey_6x6 \
    --n_episodes 1000 \
    --multi-seed

python feature_collect.py \
    --model_path ppo_doorkey_6x6 \
    --env_name MiniGrid-DoorKey-6x6-v0 \
    --n_episodes 2000 \
    --save_dir ./stage1_outputs_doorkey

## Stage 2: SAE + Logic training
python train_joint.py \
    --features_path ./stage1_outputs_doorkey/collected_data.pt \
    --stage1_path ./stage1_outputs_doorkey/stage1_outputs.pt \
    --hidden_dim 300 \
    --k 50 \
    --n_clauses_per_action 15 \
    --n_epochs 600 \
    --max_grad_norm 5.0 \
    --save_training_data \
    --no_ica_init \
    --seed 42 \
    --threshold 0.66 \
    --entropy_weight 0.015 \
    --bimodal_ramp 120 \
    --save_dir ./outputs/lucid_model_doorkey_42
## Stage 3: Evaluation & Fidelity Metrics
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_doorkey_42/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_doorkey/collected_data.pt \
    --hard_threshold 0.66 \
    --threshold 0.66 \
    --env_name MiniGrid-DoorKey-6x6-v0 \
    --ppo_path ./ppo_doorkey_6x6 \
    --multi-seed \
    --episodes 1000

## Step 4: Verify Theorem
python collect_with_observations.py \
    --model_path ppo_doorkey_6x6.zip \
    --env_name MiniGrid-DoorKey-6x6-v0 \
    --n_episodes 2000 \
    --save_path held_out_doorkey_data.pt \
    --seed 42

python experiments/theorem/verify_fidelity.py \
    --model_path ./outputs/lucid_model_doorkey_42/sae_logic_joint_model.pt \
    --data_path ./held_out_doorkey_data.pt \
    --output_dir ./experiments/theorem/results_doorkey

python experiments/theorem/plot_fidelity_results.py \
    --results_file ./experiments/theorem/results_doorkey/theorem2_analytics.pt \
    --output_folder ./experiments/theorem/results_doorkey/figures

## Step 5: Baseline Comparisons (Decision Trees)
python experiments/dt/sa_dt_baseline.py \
    --ppo_path ppo_doorkey_6x6.zip \
    --env_name MiniGrid-DoorKey-6x6-v0 \
    --data_path stage1_outputs_doorkey/collected_data.pt \
    --save_dir ./experiments/dt/results_doorkey_6x6 \
    --n_eval_episodes 1000 \
    --max_depth 4 \
    --multi-seed

python experiments/dt/viper_baseline.py \
    --ppo_path ppo_doorkey_6x6.zip \
    --env_name MiniGrid-DoorKey-6x6-v0 \
    --max_depth 4 \
    --save_dir experiments/dt/results_doorkey_6x6 \
    --data_path stage1_outputs_doorkey/collected_data.pt \
    --n_dagger_iters 75 \
    --episodes_per_iter 50 \
    --max_steps 100 \
    --n_eval_episodes 1000 \
    --multi-seed \
    --min_samples_leaf 4

python experiments/soft_dt/soft_dt_baseline.py \
    --data_path stage1_outputs_doorkey/collected_data.pt \
    --env_name MiniGrid-DoorKey-6x6-v0 \
    --n_eval_episodes 1000 \
    --max_depth 4 \
    --multi-seed \
    --epochs 1000 \
    --save_dir experiments/soft_dt/results_doorkey_6x6/

# MiniGrid-Dynamic-Obstacles-5x5-v0

## Stage 1: Train & Test PPO Model + Rollout data
python train_dynamic_obstacles_5x5.py

python test_ppo_dynamic_obs_5x5.py \
    --n_episodes 1000 \
    --multi-seed

python feature_collect.py \
    --model_path ppo_dynamic_obs_5x5.zip \
    --env_name MiniGrid-Dynamic-Obstacles-5x5-v0 \
    --n_episodes 3000 \
    --save_dir ./stage1_outputs_dyobs

## Stage 2: SAE + Logic training
python train_joint.py \
    --features_path ./stage1_outputs_dyobs/collected_data.pt \
    --stage1_path ./stage1_outputs_dyobs/stage1_outputs.pt \
    --hidden_dim 200 \
    --k 20 \
    --n_clauses_per_action 10 \
    --n_epochs 300 \
    --max_grad_norm 5.0 \
    --save_training_data \
    --no_ica_init \
    --env_name MiniGrid-Dynamic-Obstacles-5x5-v0 \
    --seed 42 \
    --threshold 0.66 \
    --l0_penalty 4.5 \
    --bimodal_max 0.4 \
    --entropy_weight 0.0 \
    --bimodal_ramp 60 \
    --save_dir ./outputs/lucid_model_dyobs_42

python train_joint.py \
    --features_path ./stage1_outputs_dqn/dqn_dyobs/collected_data.pt \
    --stage1_path ./stage1_outputs_dqn/dqn_dyobs/stage1_outputs.pt \
    --hidden_dim 200 \
    --k 20 \
    --n_clauses_per_action 10 \
    --n_epochs 300 \
    --max_grad_norm 5.0 \
    --save_training_data \
    --no_ica_init \
    --env_name MiniGrid-Dynamic-Obstacles-5x5-v0 \
    --seed 42 \
    --threshold 0.66 \
    --l0_penalty 20.0 \
    --bimodal_max 1.5 \
    --entropy_weight 0.05 \
    --bimodal_ramp 60 \
    --save_dir ./outputs/lucid_model_dyobs_dqn_421
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_dyobs_dqn_421/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_dqn/dqn_dyobs/collected_data.pt \
    --threshold 0.66 \
    --hard_threshold 0.66 \
    --ppo_path dqn_dynamic_obs_5x5.zip \
    --algo dqn \
    --env_name MiniGrid-Dynamic-Obstacles-5x5-v0 \
    --episodes 1000 \
    --multi-seed
## Stage 3: Evaluation & Fidelity Metrics
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_dyobs_42/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_dyobs/collected_data.pt \
    --threshold 0.66 \
    --hard_threshold 0.66 \
    --ppo_path ppo_dynamic_obs_5x5.zip \
    --env_name MiniGrid-Dynamic-Obstacles-5x5-v0 \
    --episodes 1000 \
    --multi-seed

## Step 4: Verify fidelity (Theorem)
python collect_with_observations.py \
    --model_path ppo_dynamic_obs_5x5.zip \
    --env_name MiniGrid-Dynamic-Obstacles-5x5-v0 \
    --n_episodes 2000 \
    --save_path held_out_dyobs_data.pt  

python experiments/theorem/verify_fidelity.py \
    --model_path ./outputs/lucid_model_dyobs_42/sae_logic_joint_model.pt \
    --data_path ./held_out_dyobs_data.pt \
    --output_dir ./experiments/theorem/results_dyobs

python experiments/theorem/plot_fidelity_results.py \
    --results_file ./experiments/theorem/results_dyobs/theorem2_analytics.pt \
    --output_folder ./experiments/theorem/results_dyobs/figures

## Step 5: Baseline Comparisons (Decision Trees)
python experiments/dt/sa_dt_baseline.py \
    --ppo_path ppo_dynamic_obs_5x5.zip \
    --env_name MiniGrid-Dynamic-Obstacles-5x5-v0 \
    --data_path stage1_outputs_dyobs/collected_data.pt \
    --save_dir ./experiments/dt/results_dynamic_obs_5x5 \
    --n_eval_episodes 1000 \
    --max_depth 2 \
    --multi-seed \
    --seed 42

python experiments/dt/viper_baseline.py \
    --ppo_path ppo_dynamic_obs_5x5.zip \
    --env_name MiniGrid-Dynamic-Obstacles-5x5-v0 \
    --max_depth 2 \
    --seed 42 \
    --save_dir experiments/dt/results_dynamic_obs_5x5 \
    --data_path stage1_outputs_dyobs/collected_data.pt \
    --n_dagger_iters 15 \
    --episodes_per_iter 20 \
    --max_steps 100 \
    --multi-seed \
    --n_eval_episodes 1000

python experiments/soft_dt/soft_dt_baseline.py \
    --ppo_path ppo_dynamic_obs_5x5.zip \
    --env_name MiniGrid-Dynamic-Obstacles-5x5-v0 \
    --data_path stage1_outputs_dyobs/collected_data.pt \
    --n_eval_episodes 1000 \
    --max_depth 2 \
    --multi-seed \
    --save_dir experiments/soft_dt/results_dynamic_obs_5x5/ \
    --epochs 300

# PixelCartPole-v0:

## Stage 1: Train & Test PPO Model + Rollout data
python train_pixel_cartpole.py

python test_pixel_cartpole.py \
    --n_episodes 200 \
    --multi-seed

python feature_collect.py \
    --model_path ppo_pixel_cartpole.zip \
    --env_name PixelCartPole-v0 \
    --n_episodes 100 \
    --save_dir ./stage1_outputs_cartpole

## Stage 2: SAE + Logic training
python train_joint.py \
    --features_path ./stage1_outputs_cartpole/collected_data.pt \
    --stage1_path ./stage1_outputs_cartpole/stage1_outputs.pt \
    --env_name PixelCartPole-v0 \
    --hidden_dim 512 --k 50 \
    --n_clauses_per_action 20 \
    --n_epochs 900 \
    --threshold 0.66 \
    --entropy_weight 0.005 \
    --bimodal_ramp 200 \
    --max_grad_norm 5.0 \
    --save_training_data \
    --no_ica_init \
    --save_dir ./outputs/lucid_model_cartpole_42

## Stage 3: Evaluation & Fidelity Metrics
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_cartpole_42/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_cartpole/collected_data.pt \
    --threshold 0.66 \
    --hard_threshold 0.66 \
    --ppo_path ppo_pixel_cartpole.zip \
    --env_name PixelCartPole-v0 \
    --multi-seed \
    --episodes 200

## Step 4: Verify fidelity (Theorem)
python collect_with_observations.py \
    --model_path ppo_pixel_cartpole.zip \
    --env_name PixelCartPole-v0 \
    --n_episodes 50 \
    --save_path held_out_cartpole_data.pt \
    --save_obs_mode grid \
    --seed 42

python experiments/theorem/verify_fidelity.py \
    --model_path ./outputs/lucid_model_cartpole_42/sae_logic_joint_model.pt \
    --data_path ./held_out_cartpole_data.pt \
    --output_dir ./experiments/theorem/results_cartpole

python experiments/theorem/plot_fidelity_results.py \
    --results_file ./experiments/theorem/results_cartpole/theorem2_analytics.pt \
    --output_folder ./experiments/theorem/results_cartpole/figures

## Step 5: Baseline Comparisons (Decision Trees)
python experiments/dt/viper_baseline.py \
    --ppo_path ppo_pixel_cartpole.zip \
    --env_name PixelCartPole-v0 \
    --save_dir ./experiments/dt/results_cartpole \
    --data_path stage1_outputs_cartpole/collected_data.pt \
    --n_eval_episodes 200 \
    --max_depth 4 \
    --multi-seed \
    --n_dagger_iters 20 \
    --episodes_per_iter 15

python experiments/dt/sa_dt_baseline.py \
    --ppo_path ppo_pixel_cartpole.zip \
    --env_name PixelCartPole-v0 \
    --data_path stage1_outputs_cartpole/collected_data.pt \
    --save_dir ./experiments/dt/results_cartpole \
    --n_eval_episodes 200 \
    --max_depth 4 \
    --multi-seed

python experiments/soft_dt/soft_dt_baseline.py \
    --ppo_path ppo_pixel_cartpole.zip \
    --env_name PixelCartPole-v0 \
    --data_path stage1_outputs_cartpole/collected_data.pt \
    --n_eval_episodes 200 \
    --max_depth 4 \
    --multi-seed \
    --save_dir ./experiments/soft_dt/results_cartpole \
    --epoch 500

## Extend for DQN_LUCID:
D=./stage1_outputs_dqn/dqn_cartpole; T=dqn_pixel_cartpole.zip; SEED=42

python feature_collect.py --features_path $D/collected_data.pt --env_name PixelCartPole-v0 --save_dir $D

python -c "import torch; d=torch.load('$D/collected_data.pt', mmap=True, weights_only=False); torch.save({'features': d['features'].clone(), 'actions': d['actions'].clone()}, '$D/features_only.pt')"

python compute_margin_weights.py --algo dqn --model_path $T --env_name PixelCartPole-v0 \
    --features_path $D/features_only.pt --out $D/margin_weights_w100.pt --wmax 100

python -u train_joint.py \
    --features_path $D/features_only.pt \
    --stage1_path $D/stage1_outputs.pt \
    --sample_weights $D/margin_weights_w100.pt \
    --hidden_dim 512 --k 50 \
    --n_clauses_per_action 20 \
    --env_name PixelCartPole-v0 \
    --no_ica_init \
    --n_epochs 250 \
    --threshold 0.5 \
    --entropy_weight 0.1 \
    --l0_penalty 20.0 \
    --bimodal_max 3.0 \
    --bimodal_ramp 100 \
    --bias_max 1.0 \
    --max_grad_norm 5.0 \
    --seed $SEED \
    --save_training_data \
    --save_dir ./outputs/lucid_model_cartpole_dqn_$SEED
python -u experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_cartpole_dqn_$SEED/sae_logic_joint_model.pt \
    --features_path $D/features_only.pt \
    --threshold 0.66 --hard_threshold 0.66 \
    --ppo_path $T --algo dqn \
    --env_name PixelCartPole-v0 \
    --episodes 100 --multi-seed \
    --save_dir ./outputs/lucid_model_cartpole_dqn_$SEED/eval

# BoxingNoFrameskip-v4:

## Stage 1: Train & Test PPO Model + Rollout data
python train_boxing.py

python test_boxing.py \
    --model_path boxing_sb3_290.zip \
    --n_episodes 100 \
    --multi-seed

python feature_collect.py \
    --model_path boxing_sb3_290.zip \
    --env_name BoxingNoFrameskip-v4 \
    --n_episodes 300 \
    --save_dir ./stage1_outputs_boxing

## Stage 2: SAE + Logic training
python train_joint.py \
    --features_path ./stage1_outputs_boxing/collected_data.pt \
    --stage1_path ./stage1_outputs_boxing/stage1_outputs.pt \
    --hidden_dim 512 --k 50 \
    --n_clauses_per_action 10 \
    --no_ica_init \
    --n_epochs 600 \
    --threshold 0.5 \
    --entropy_weight 0.3 \
    --l0_penalty 100.0 \
    --bimodal_ramp 200 \
    --max_grad_norm 5.0 \
    --save_training_data \
    --save_dir ./outputs/lucid_model_boxing_42

## Stage 3: Evaluation & Fidelity Metrics
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_boxing_42/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_boxing/collected_data.pt \
    --threshold 0.66 \
    --ppo_path boxing_sb3_290.zip \
    --env_name BoxingNoFrameskip-v4 \
    --episodes 100 \
    --multi-seed

## Step 4: Verify fidelity (Theorem)
python collect_with_observations.py \
    --model_path boxing_sb3_290.zip \
    --env_name BoxingNoFrameskip-v4 \
    --n_episodes 100 \
    --save_path held_out_boxing_data.pt

python experiments/theorem/verify_fidelity.py \
    --model_path ./outputs/lucid_model_boxing_42/sae_logic_joint_model.pt \
    --data_path ./held_out_boxing_data.pt \
    --output_dir ./experiments/theorem/results_boxing

python experiments/theorem/plot_fidelity_results.py \
    --results_file ./experiments/theorem/results_boxing/theorem2_analytics.pt \
    --output_folder ./experiments/theorem/results_boxing/figures

## Step 5: Baseline Comparisons (Decision Trees)
python experiments/dt/sa_dt_baseline.py \
    --ppo_path boxing_sb3_290.zip \
    --env_name BoxingNoFrameskip-v4 \
    --data_path stage1_outputs_boxing/collected_data.pt \
    --save_dir ./experiments/dt/results_boxing \
    --n_eval_episodes 100 \
    --max_depth 6 \
    --multi-seed \
    --seed 42

python experiments/dt/viper_baseline.py \
    --ppo_path boxing_sb3_290.zip \
    --env_name BoxingNoFrameskip-v4 \
    --max_depth 6 \
    --seed 42 \
    --save_dir experiments/dt/results_boxing \
    --data_path stage1_outputs_boxing/collected_data.pt \
    --n_dagger_iters 15 \
    --episodes_per_iter 20 \
    --max_steps 27000 \
    --multi-seed \
    --n_eval_episodes 100

python experiments/soft_dt/soft_dt_baseline.py \
    --ppo_path boxing_sb3_290.zip \
    --env_name BoxingNoFrameskip-v4 \
    --data_path stage1_outputs_boxing/collected_data.pt \
    --n_eval_episodes 100 \
    --max_depth 6 \
    --multi-seed \
    --epochs 200 \
    --save_dir experiments/soft_dt/results_boxing/

## Extend for DQN_LUCID:
D=./stage1_outputs_dqn/dqn_boxing; T=teachers/boxing_dqn_sb3_290.zip; SEED=42

python -c "import torch; d=torch.load('$D/collected_data.pt', mmap=True, weights_only=False); torch.save({'features': d['features'].clone(), 'actions': d['actions'].clone()}, '$D/features_only.pt')"
for cfg in "1000 7" "800 11" "800 12"; do set -- $cfg
  python feature_collect.py --model_path $T --algo dqn --env_name BoxingNoFrameskip-v4 \
      --n_episodes $1 --seed $2 --save_dir ${D}_tmp
  python -c "
import torch
a=torch.load('$D/features_only.pt', mmap=True, weights_only=False); b=torch.load('${D}_tmp/collected_data.pt', mmap=True, weights_only=False)
torch.save({'features': torch.cat([a['features'], b['features']]), 'actions': torch.cat([a['actions'], b['actions']])}, '$D/features_only.pt')
print('total', len(a['actions'])+len(b['actions']))"
  rm -rf ${D}_tmp
done

python feature_collect.py --features_path $D/features_only.pt --env_name BoxingNoFrameskip-v4 --save_dir $D

python compute_margin_weights.py --algo dqn --model_path $T --env_name BoxingNoFrameskip-v4 \
    --features_path $D/features_only.pt --out $D/margin_weights_w500.pt --wmax 500

python -u train_joint.py \
    --features_path $D/features_only.pt \
    --stage1_path $D/stage1_outputs.pt \
    --sample_weights $D/margin_weights_w500.pt \
    --hidden_dim 512 --k 50 \
    --n_clauses_per_action 10 \
    --env_name BoxingNoFrameskip-v4 \
    --no_ica_init \
    --n_epochs 400 \
    --threshold 0.5 \
    --entropy_weight 0.3 \
    --l0_penalty 50.0 \
    --bimodal_max 15.0 \
    --bimodal_ramp 150 \
    --bias_max 1.5 \
    --max_grad_norm 5.0 \
    --seed $SEED \
    --save_training_data \
    --save_dir ./outputs/lucid_model_boxing_dqn_$SEED
python -u experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_boxing_dqn_$SEED/sae_logic_joint_model.pt \
    --features_path $D/features_only.pt \
    --threshold 0.66 --hard_threshold 0.66 \
    --ppo_path $T --algo dqn \
    --env_name BoxingNoFrameskip-v4 \
    --episodes 100 --multi-seed \
    --save_dir ./outputs/lucid_model_boxing_dqn_$SEED/eval

# PongNoFrameskip-v4:

## Stage 1: Train & Test PPO Model + Rollout data
python train_pong.py

python test_pong.py \
    --model_path pong_sb3_290.zip \
    --n_episodes 100 \
    --multi-seed

python feature_collect.py \
    --model_path pong_sb3_290.zip \
    --env_name PongNoFrameskip-v4 \
    --n_episodes 300 \
    --save_dir ./stage1_outputs_pong

## Stage 2: SAE + Logic training
python train_joint.py \
    --features_path ./stage1_outputs_pong/collected_data.pt \
    --stage1_path ./stage1_outputs_pong/stage1_outputs.pt \
    --hidden_dim 512 --k 50 \
    --n_clauses_per_action 10 \
    --env_name PongNoFrameskip-v4 \
    --no_ica_init \
    --n_epochs 600 \
    --threshold 0.5 \
    --entropy_weight 0.3 \
    --l0_penalty 100.0 \
    --bimodal_ramp 200 \
    --max_grad_norm 5.0 \
    --save_training_data \
    --save_dir ./outputs/lucid_model_pong_42

## Stage 3: Evaluation & Fidelity Metrics
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_pong_42/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_pong/collected_data.pt \
    --threshold 0.66 \
    --ppo_path pong_sb3_290.zip \
    --env_name PongNoFrameskip-v4 \
    --episodes 100 \
    --multi-seed

## Step 4: Verify fidelity (Theorem)
python collect_with_observations.py \
    --model_path pong_sb3_290.zip \
    --env_name PongNoFrameskip-v4 \
    --n_episodes 100 \
    --save_path held_out_pong_data.pt

python experiments/theorem/verify_fidelity.py \
    --model_path ./outputs/lucid_model_pong_42/sae_logic_joint_model.pt \
    --data_path ./held_out_pong_data.pt \
    --output_dir ./experiments/theorem/results_pong

python experiments/theorem/plot_fidelity_results.py \
    --results_file ./experiments/theorem/results_pong/theorem2_analytics.pt \
    --output_folder ./experiments/theorem/results_pong/figures

## Step 5: Baseline Comparisons (Decision Trees)
python experiments/dt/sa_dt_baseline.py \
    --ppo_path pong_sb3_290.zip \
    --env_name PongNoFrameskip-v4 \
    --data_path stage1_outputs_pong/collected_data.pt \
    --save_dir ./experiments/dt/results_pong \
    --n_eval_episodes 100 \
    --max_depth 6 \
    --multi-seed \
    --seed 42
python experiments/dt/viper_baseline.py \
    --ppo_path pong_sb3_290.zip \
    --env_name PongNoFrameskip-v4 \
    --max_depth 6 \
    --seed 42 \
    --save_dir experiments/dt/results_pong \
    --data_path stage1_outputs_pong/collected_data.pt \
    --n_dagger_iters 15 \
    --episodes_per_iter 20 \
    --max_steps 27000 \
    --multi-seed \
    --n_eval_episodes 100
python experiments/soft_dt/soft_dt_baseline.py \
    --ppo_path pong_sb3_290.zip \
    --env_name PongNoFrameskip-v4 \
    --data_path stage1_outputs_pong/collected_data.pt \
    --n_eval_episodes 100 \
    --max_depth 6 \
    --multi-seed \
    --epochs 300 \
    --save_dir experiments/soft_dt/results_pong/

## Extend for DQN_LUCID:
D=./stage1_outputs_dqn/dqn_pong; T=teachers/hf/dqn_pong/dqn-PongNoFrameskip-v4.zip; SEED=42

python feature_collect.py --features_path $D/collected_data.pt --env_name PongNoFrameskip-v4 --save_dir $D

python -c "import torch; d=torch.load('$D/collected_data.pt', mmap=True, weights_only=False); torch.save({'features': d['features'].clone(), 'actions': d['actions'].clone()}, '$D/features_only.pt')"

python compute_margin_weights.py --algo dqn --model_path $T --env_name PongNoFrameskip-v4 \
    --features_path $D/features_only.pt --out $D/margin_weights_w100.pt --wmax 100

python -u train_joint.py \
    --features_path $D/features_only.pt \
    --stage1_path $D/stage1_outputs.pt \
    --sample_weights $D/margin_weights_w100.pt \
    --hidden_dim 512 --k 50 \
    --n_clauses_per_action 10 \
    --env_name PongNoFrameskip-v4 \
    --no_ica_init \
    --n_epochs 250 \
    --threshold 0.5 \
    --entropy_weight 0.3 \
    --l0_penalty 100.0 \
    --bimodal_ramp 100 \
    --bias_max 1.5 \
    --max_grad_norm 5.0 \
    --seed $SEED \
    --save_training_data \
    --save_dir ./outputs/lucid_model_pong_dqn_$SEED
python -u experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_pong_dqn_$SEED/sae_logic_joint_model.pt \
    --features_path $D/collected_data.pt \
    --threshold 0.66 --hard_threshold 0.66 \
    --ppo_path $T --algo dqn \
    --env_name PongNoFrameskip-v4 \
    --episodes 100 --multi-seed \
    --save_dir ./outputs/lucid_model_pong_dqn_$SEED/eval