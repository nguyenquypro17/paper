#!/usr/bin/env bash
# Run from ~/XRL with the 4 .py files copied there.  conda activate lucid
# One-time: pip install "transformers>=4.52" accelerate bitsandbytes sentence-transformers captum
#           pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
#           huggingface-cli login   (Gemma-3 needs the licence accepted on HF)
set -e
declare -A ENV=( [doorkey]=MiniGrid-DoorKey-6x6-v0 [dyobs]=MiniGrid-Dynamic-Obstacles-5x5-v0
                 [cartpole]=PixelCartPole [pong]=PongNoFrameskip-v4 [boxing]=BoxingNoFrameskip-v4 )
declare -A PPO=( [doorkey]=ppo_doorkey_6x6.zip [dyobs]=ppo_dynamic_obs_5x5.zip
                 [cartpole]=ppo_pixel_cartpole.zip [pong]=pong_sb3_290.zip [boxing]=boxing_sb3_290.zip )
declare -A EPS=( [doorkey]="--n_episodes 2000" [dyobs]="--n_episodes 2000" [cartpole]="--n_episodes 20 --stride 5"
                 [pong]="--n_episodes 6 --stride 4" [boxing]="--n_episodes 6 --stride 4" )

for e in ${@:-doorkey dyobs cartpole pong boxing}; do
  D=experiments/vlm/${e}_42
  rm -rf $D
  python render_concepts.py --model_path outputs/lucid_model_${e}_42/sae_logic_joint_model.pt \
      --ppo_path ${PPO[$e]} --env_name ${ENV[$e]} ${EPS[$e]} --seed 2000 --out_dir $D
  python vlm_name.py --img_dir $D/naming --env_name ${ENV[$e]} --model qwen     --out $D/labels_qwen.json
  python vlm_name.py --img_dir $D/naming --env_name ${ENV[$e]} --model internvl --out $D/labels_internvl.json
  python vlm_predict.py --test_dir $D/test --labels $D/labels_qwen.json $D/labels_internvl.json \
      --env_name ${ENV[$e]} --model gemma --load_4bit --out $D/predict_gemma.json
  python label_agreement.py --labels_a $D/labels_qwen.json --labels_b $D/labels_internvl.json \
      --out $D/agreement.json
  python label_factor.py --meta $D/meta.json --labels $D/labels_qwen.json $D/labels_internvl.json \
      --env_name ${ENV[$e]} --out $D/label_factor.json
done
