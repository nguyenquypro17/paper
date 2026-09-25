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
    
## Step 4: Semantic Concept Grounding (VLM)
python auto_label_concepts.py \
    --model_path ./outputs/lucid_model_doorkey/sae_logic_joint_model.pt \
    --ppo_path ./ppo_doorkey_6x6.zip \
    --env_name MiniGrid-DoorKey-6x6-v0 \
    --episodes 100 \
    --top_k 8

python gemini_concept_labeler.py \
    --img_dir ./concept_grounding/MiniGrid_DoorKey_6x6_v0 \
    --env_name "MiniGrid DoorKey 6x6" \
    --top_k 8 \
    --rules_path ./outputs/lucid_model_doorkey/learned_rules.json \
    --model_name gemini-2.5-pro \
    --api_key YOUR_API_KEY

## Step 5: Verify Theorem
python collect_with_observations.py \
    --model_path ppo_doorkey_6x6.zip \
    --env_name MiniGrid-DoorKey-6x6-v0 \
    --n_episodes 2000 \
    --save_path held_out_doorkey_data.pt \
    --seed 42

python experiments/theorem/verify_fidelity.py \
    --model_path ./outputs/lucid_model_doorkey/sae_logic_joint_model.pt \
    --data_path ./held_out_doorkey_data.pt \
    --output_dir ./experiments/theorem/results_doorkey

python experiments/theorem/plot_fidelity_results.py \
    --results_file ./experiments/theorem/results_doorkey/theorem2_analytics.pt \
    --output_folder ./experiments/theorem/results_doorkey/figures

## Step 6: Ablation Studies (Threshold Sensitivity)
Analyze how the binarization threshold $\tau$ affects the model's performance on held-out data. This helps verify the "Schelling optimality" of $\tau = 0.5$.
python experiments/tau/run_tau_ablation.py \
    --model_path ./outputs/lucid_model_doorkey/sae_logic_joint_model.pt \
    --features_path ./held_out_doorkey_data.pt \
    --env_name MiniGrid-DoorKey-6x6-v0 \
    --output_dir ./experiments/tau/results_doorkey \
    --episodes 1000

## Step 7: Baseline Comparisons (Decision Trees)
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
    --max_depth 8 \
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
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_dyobs_42/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_dyobs/collected_data.pt \
    --threshold 0.66 \
    --hard_threshold 0.66 \
    --ppo_path ppo_dynamic_obs_5x5.zip \
    --env_name MiniGrid-Dynamic-Obstacles-5x5-v0 \
    --episodes 1000 \
    --multi-seed

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
    --seed 43 \
    --threshold 0.66 \
    --l0_penalty 4.5 \
    --bimodal_max 0.4 \
    --entropy_weight 0.0 \
    --bimodal_ramp 60 \
    --save_dir ./outputs/lucid_model_dyobs_43
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_dyobs_43/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_dyobs/collected_data.pt \
    --threshold 0.66 \
    --hard_threshold 0.66 \
    --ppo_path ppo_dynamic_obs_5x5.zip \
    --env_name MiniGrid-Dynamic-Obstacles-5x5-v0 \
    --episodes 1000 \
    --multi-seed
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
    --seed 44 \
    --threshold 0.66 \
    --l0_penalty 4.5 \
    --bimodal_max 0.4 \
    --entropy_weight 0.0 \
    --bimodal_ramp 60 \
    --save_dir ./outputs/lucid_model_dyobs_44
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_dyobs_44/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_dyobs/collected_data.pt \
    --threshold 0.66 \
    --hard_threshold 0.66 \
    --ppo_path ppo_dynamic_obs_5x5.zip \
    --env_name MiniGrid-Dynamic-Obstacles-5x5-v0 \
    --episodes 1000 \
    --multi-seed

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
    --seed 42 \
    --threshold 0.5 \
    --l0_penalty 4.5 \
    --bimodal_max 0.4 \
    --entropy_weight 0.06 \
    --bimodal_ramp 60 \
    --save_dir ./outputs/lucid_model_dyobs

## Stage 3: Evaluation & Fidelity Metrics
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_dyobs/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_dyobs/collected_data.pt \
    --threshold 0.66 \
    --hard_threshold 0.5 \
    --ppo_path ppo_dynamic_obs_5x5.zip \
    --env_name MiniGrid-Dynamic-Obstacles-5x5-v0 \
    --multi-seed \
    --episodes 1000
    
## Step 4: Semantic Concept Grounding (VLM)
python auto_label_concepts.py \
    --model_path ./outputs/lucid_model_dyobs/sae_logic_joint_model.pt \
    --ppo_path ppo_dynamic_obs_5x5.zip \
    --env_name MiniGrid-Dynamic-Obstacles-5x5-v0 \
    --episodes 50 \
    --top_k 8

python gemini_concept_labeler.py \
    --img_dir ./concept_grounding/MiniGrid_Dynamic_Obstacles_5x5_v0 \
    --env_name "MiniGrid-Dynamic-Obstacles-5x5-v0" \
    --top_k 8 \
    --model_name gemini-2.5-pro \
    --rules_path ./outputs/lucid_model_dyobs/learned_rules.json \
    --api_key YOUR_API_KEY

## Step 5: Verify fidelity (Theorem)
python collect_with_observations.py \
    --model_path ppo_dynamic_obs_5x5.zip \
    --env_name MiniGrid-Dynamic-Obstacles-5x5-v0 \
    --n_episodes 2000 \
    --save_path held_out_dyobs_data.pt  

python experiments/theorem/verify_fidelity.py \
    --model_path ./outputs/lucid_model_dyobs/sae_logic_joint_model.pt \
    --data_path ./held_out_dyobs_data.pt \
    --output_dir ./experiments/theorem/results_dyobs

python experiments/theorem/plot_fidelity_results.py \
    --results_file ./experiments/theorem/results_dyobs/theorem2_analytics.pt \
    --output_folder ./experiments/theorem/results_dyobs/figures

## Step 6: Ablation Studies (Threshold Sensitivity)
Analyze how the binarization threshold $\tau$ affects the model's performance on held-out data. This helps verify the "Schelling optimality" of $\tau = 0.5$.
python experiments/tau/run_tau_ablation.py \
    --model_path ./outputs/lucid_model_dyobs/sae_logic_joint_model.pt \
    --features_path ./held_out_dyobs_data.pt \
    --env_name MiniGrid-Dynamic-Obstacles-5x5-v0 \
    --output_dir ./experiments/tau/results_dyobs \
    --episodes 1000

## Step 7: Baseline Comparisons (Decision Trees)
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
    --max_depth 4 \
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
    --epochs 1000

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
python train_joint.py \
    --features_path ./stage1_outputs_cartpole/collected_data.pt \
    --stage1_path ./stage1_outputs_cartpole/stage1_outputs.pt \
    --hidden_dim 512 --k 50 \
    --n_clauses_per_action 30 \
    --n_epochs 1000 \
    --threshold 0.66 \
    --entropy_weight 0.0005 \
    --bimodal_ramp 200 \
    --max_grad_norm 5.0 \
    --save_training_data \
    --no_ica_init \
    --save_dir ./outputs/lucid_model_cartpole
python train_joint.py \
    --features_path ./stage1_outputs_cartpole/collected_data.pt \
    --stage1_path ./stage1_outputs_cartpole/stage1_outputs.pt \
    --env_name PixelCartPole-v0 \
    --hidden_dim 512 --k 50 \
    --n_clauses_per_action 30 \
    --n_epochs 900 \
    --threshold 0.66 \
    --entropy_weight 0.0004 \
    --bimodal_ramp 200 \
    --max_grad_norm 5.0 \
    --save_training_data \
    --no_ica_init \
    --save_dir ./outputs/lucid_model_cartpole2
python train_joint.py \
    --features_path ./stage1_outputs_cartpole/collected_data.pt \
    --stage1_path ./stage1_outputs_cartpole/stage1_outputs.pt \
    --env_name PixelCartPole-v0 \
    --hidden_dim 512 --k 50 \
    --n_clauses_per_action 20 \
    --n_epochs 900 \
    --seed 43 \
    --threshold 0.66 \
    --entropy_weight 0.005 \
    --bimodal_ramp 200 \
    --max_grad_norm 5.0 \
    --save_training_data \
    --no_ica_init \
    --save_dir ./outputs/lucid_model_cartpole_43
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_cartpole_43/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_cartpole/collected_data.pt \
    --threshold 0.66 \
    --hard_threshold 0.66 \
    --ppo_path ppo_pixel_cartpole.zip \
    --env_name PixelCartPole-v0 \
    --multi-seed \
    --episodes 200
python train_joint.py \
    --features_path ./stage1_outputs_cartpole/collected_data.pt \
    --stage1_path ./stage1_outputs_cartpole/stage1_outputs.pt \
    --env_name PixelCartPole-v0 \
    --hidden_dim 512 --k 50 \
    --n_clauses_per_action 20 \
    --n_epochs 900 \
    --seed 44 \
    --threshold 0.66 \
    --entropy_weight 0.005 \
    --bimodal_ramp 200 \
    --max_grad_norm 5.0 \
    --save_training_data \
    --no_ica_init \
    --save_dir ./outputs/lucid_model_cartpole_44
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_cartpole_44/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_cartpole/collected_data.pt \
    --threshold 0.66 \
    --hard_threshold 0.66 \
    --ppo_path ppo_pixel_cartpole.zip \
    --env_name PixelCartPole-v0 \
    --multi-seed \
    --episodes 200
## Stage 2: SAE + Logic training
python train_joint.py \
    --features_path ./stage1_outputs_cartpole/collected_data.pt \
    --stage1_path ./stage1_outputs_cartpole/stage1_outputs.pt \
    --env_name PixelCartPole-v0 \
    --hidden_dim 512 --k 50 \
    --n_clauses_per_action 30 \
    --n_epochs 1000 \
    --threshold 0.66 \
    --entropy_weight 0.0001 \
    --bimodal_ramp 200 \
    --max_grad_norm 5.0 \
    --save_training_data \
    --no_ica_init \
    --save_dir ./outputs/lucid_model_cartpole2

## Stage 3: Evaluation & Fidelity Metrics
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_cartpole/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_cartpole/collected_data.pt \
    --threshold 0.66 \
    --hard_threshold 0.66 \
    --ppo_path ppo_pixel_cartpole.zip \
    --env_name PixelCartPole-v0 \
    --multi-seed \
    --episodes 200
    
## Step 4: Semantic Concept Grounding (VLM)
python auto_label_concepts.py \
    --model_path ./outputs/lucid_model_cartpole/sae_logic_joint_model.pt \
    --ppo_path ./ppo_pixel_cartpole.zip \
    --env_name PixelCartPole-v0 \
    --episodes 30 \
    --top_k 6

python gemini_concept_labeler.py \
    --img_dir ./concept_grounding/PixelCartPole_v0 \
    --env_name "PixelCartPole-v0" \
    --top_k 6 \
    --model_name gemini-2.5-pro \
    --rules_path ./outputs/lucid_model_cartpole/learned_rules.json \
    --model_name gemini-2.5-pro \
    --api_key YOUR_API_KEY

## Step 5: Verify fidelity (Theorem)
python collect_with_observations.py \
    --model_path ppo_pixel_cartpole.zip \
    --env_name PixelCartPole-v0 \
    --n_episodes 50 \
    --save_path held_out_cartpole_data.pt \
    --save_obs_mode grid \
    --seed 42

python experiments/theorem/verify_fidelity.py \
    --model_path ./outputs/lucid_model_cartpole/sae_logic_joint_model.pt \
    --data_path ./held_out_cartpole_data.pt \
    --output_dir ./experiments/theorem/results_cartpole

python experiments/theorem/plot_fidelity_results.py \
    --results_file ./experiments/theorem/results_cartpole/theorem2_analytics.pt \
    --output_folder ./experiments/theorem/results_cartpole/figures

## Step 6: Ablation Studies (Threshold Sensitivity)
Analyze how the binarization threshold $\tau$ affects the model's performance on held-out data. This helps verify the "Schelling optimality" of $\tau = 0.5$.
python experiments/tau/run_tau_ablation.py \
    --model_path ./outputs/lucid_model_cartpole/sae_logic_joint_model.pt \
    --features_path ./held_out_cartpole_data.pt \
    --env_name PixelCartPole-v0 \
    --ppo_path ./ppo_pixel_cartpole.zip \
    --output_dir ./experiments/tau/results_cartpole \
    --episodes 200 \
    --seed 42

## Step 7: Baseline Comparisons (Decision Trees)
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
    --save_dir ./outputs/lucid_model_boxing
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_boxing/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_boxing/collected_data.pt \
    --threshold 0.66 \
    --ppo_path boxing_sb3_290.zip \
    --env_name BoxingNoFrameskip-v4 \
    --episodes 100 \
    --multi-seed

python train_joint.py \
    --features_path ./stage1_outputs_boxing/collected_data.pt \
    --stage1_path ./stage1_outputs_boxing/stage1_outputs.pt \
    --hidden_dim 512 --k 50 \
    --n_clauses_per_action 10 \
    --no_ica_init \
    --seed 43 \
    --n_epochs 600 \
    --threshold 0.5 \
    --entropy_weight 0.3 \
    --l0_penalty 100.0 \
    --bimodal_ramp 200 \
    --max_grad_norm 5.0 \
    --save_training_data \
    --save_dir ./outputs/lucid_model_boxing_43
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_boxing_43/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_boxing/collected_data.pt \
    --threshold 0.66 \
    --ppo_path boxing_sb3_290.zip \
    --env_name BoxingNoFrameskip-v4 \
    --episodes 100 \
    --multi-seed
python train_joint.py \
    --features_path ./stage1_outputs_boxing/collected_data.pt \
    --stage1_path ./stage1_outputs_boxing/stage1_outputs.pt \
    --hidden_dim 512 --k 50 \
    --n_clauses_per_action 10 \
    --no_ica_init \
    --seed 44 \
    --n_epochs 600 \
    --threshold 0.5 \
    --entropy_weight 0.3 \
    --l0_penalty 100.0 \
    --bimodal_ramp 200 \
    --max_grad_norm 5.0 \
    --save_training_data \
    --save_dir ./outputs/lucid_model_boxing_44
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_boxing_44/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_boxing/collected_data.pt \
    --threshold 0.66 \
    --ppo_path boxing_sb3_290.zip \
    --env_name BoxingNoFrameskip-v4 \
    --episodes 100 \
    --multi-seed
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
    --save_dir ./outputs/lucid_model_boxing

## Stage 3: Evaluation & Fidelity Metrics
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_boxing/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_boxing/collected_data.pt \
    --threshold 0.66 \
    --ppo_path boxing_sb3_290.zip \
    --env_name BoxingNoFrameskip-v4 \
    --episodes 100 \
    --multi-seed

## Step 7: Baseline Comparisons (Decision Trees)
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
    --save_dir ./outputs/lucid_model_pong
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_pong/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_pong/collected_data.pt \
    --threshold 0.66 \
    --ppo_path pong_sb3_290.zip \
    --env_name PongNoFrameskip-v4 \
    --episodes 100 \
    --multi-seed

python train_joint.py \
    --features_path ./stage1_outputs_pong/collected_data.pt \
    --stage1_path ./stage1_outputs_pong/stage1_outputs.pt \
    --hidden_dim 512 --k 50 \
    --n_clauses_per_action 10 \
    --env_name PongNoFrameskip-v4 \
    --no_ica_init \
    --seed 43 \
    --n_epochs 600 \
    --threshold 0.5 \
    --entropy_weight 0.3 \
    --l0_penalty 100.0 \
    --bimodal_ramp 200 \
    --max_grad_norm 5.0 \
    --save_training_data \
    --save_dir ./outputs/lucid_model_pong_43
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_pong_43/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_pong/collected_data.pt \
    --threshold 0.66 \
    --ppo_path pong_sb3_290.zip \
    --env_name PongNoFrameskip-v4 \
    --episodes 100 \
    --multi-seed
python train_joint.py \
    --features_path ./stage1_outputs_pong/collected_data.pt \
    --stage1_path ./stage1_outputs_pong/stage1_outputs.pt \
    --hidden_dim 512 --k 50 \
    --n_clauses_per_action 10 \
    --env_name PongNoFrameskip-v4 \
    --no_ica_init \
    --seed 44 \
    --n_epochs 600 \
    --threshold 0.5 \
    --entropy_weight 0.3 \
    --l0_penalty 100.0 \
    --bimodal_ramp 200 \
    --max_grad_norm 5.0 \
    --save_training_data \
    --save_dir ./outputs/lucid_model_pong_44
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_pong_44/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_pong/collected_data.pt \
    --threshold 0.66 \
    --ppo_path pong_sb3_290.zip \
    --env_name PongNoFrameskip-v4 \
    --episodes 100 \
    --multi-seed    
## Stage 2: SAE + Logic training
python train_joint.py \
    --features_path ./stage1_outputs_pong/collected_data.pt \
    --stage1_path ./stage1_outputs_pong/stage1_outputs.pt \
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
    --save_dir ./outputs/lucid_model_pong

## Stage 3: Evaluation & Fidelity Metrics
python experiments/lucid/evaluate_lucid_metrics.py \
    --model_path ./outputs/lucid_model_pong/sae_logic_joint_model.pt \
    --features_path ./stage1_outputs_pong/collected_data.pt \
    --threshold 0.66 \
    --ppo_path pong_sb3_290.zip \
    --env_name PongNoFrameskip-v4 \
    --episodes 100 \
    --multi-seed

## Step 7: Baseline Comparisons (Decision Trees)
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
    --epochs 200 \
    --save_dir experiments/soft_dt/results_pong/