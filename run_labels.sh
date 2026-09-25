#!/usr/bin/env bash
# run_labels.sh -- VLM labelling protocol with a configurable namer, scorer DeepSeek, hard USD caps.
# Run from ~/XRL, conda activate lucid, source ~/.vlm_keys  (OPENAI_API_KEY, DEEPSEEK_API_KEY)
#
#   NAMER=gpt-5.5-2026-04-23 TAG=gpt55 WORKERS=32 bash run_labels.sh dyobs
#
# Budget: every API call is logged in experiments/vlm/spend_<model>.json (shared by all envs and
# windows). A run stops as soon as the cap is reached; completed work is kept in the caches.
# Caps are set a little under the real budget because calls already in flight can overshoot.
set -e
NAMER=${NAMER:?set NAMER, e.g. gpt-5.5-2026-04-23}
TAG=${TAG:?set TAG, e.g. gpt55}
SCORER=${SCORER:-deepseek-flash}
WORKERS=${WORKERS:-8}
RUNS=${RUNS:-a}                 # naming runs; "a b" adds a second run for label stability
MAXC=${MAXC:-10}                # at most MAXC concepts per env, random subset with seed 0
THINK=${THINK:-1024}            # OpenAI reasoning effort: 1024 = medium, 4096 = high
BUDGET_NAMER=${BUDGET_NAMER:-9}
BUDGET_SCORER=${BUDGET_SCORER:-9}
declare -A ENV=( [doorkey]=MiniGrid-DoorKey-6x6-v0 [dyobs]=MiniGrid-Dynamic-Obstacles-5x5-v0
                 [cartpole]=PixelCartPole [pong]=PongNoFrameskip-v4 [boxing]=BoxingNoFrameskip-v4 )
declare -A PPO=( [doorkey]=ppo_doorkey_6x6.zip [dyobs]=ppo_dynamic_obs_5x5.zip
                 [cartpole]=ppo_pixel_cartpole.zip [pong]=pong_sb3_290.zip [boxing]=boxing_sb3_290.zip )
declare -A EPS=( [doorkey]="--n_episodes 2000" [dyobs]="--n_episodes 2000" [cartpole]="--n_episodes 20 --stride 5"
                 [pong]="--n_episodes 6 --stride 4" [boxing]="--n_episodes 6 --stride 4" )

for e in ${@:-doorkey dyobs cartpole pong boxing}; do
  D=experiments/vlm/${e}_42
  echo "=================== $e ($NAMER) ==================="
  if [ ! -f $D/meta.json ]; then
    python render_concepts.py --model_path outputs/lucid_model_${e}_42/sae_logic_joint_model.pt \
        --ppo_path ${PPO[$e]} --env_name ${ENV[$e]} ${EPS[$e]} --seed 2000 --out_dir $D
  fi
  for s in $RUNS; do
    [ -f $D/labels_${TAG}_$s.json ] || python vlm_name.py --img_dir $D/naming --env_name ${ENV[$e]} \
        --model $NAMER --thinking_budget $THINK --max_concepts $MAXC --seed 0 \
        --budget_usd $BUDGET_NAMER --out $D/labels_${TAG}_$s.json
  done
  python vlm_predict.py --test_dir $D/test --labels $D/labels_${TAG}_a.json \
      --env_name ${ENV[$e]} --model $SCORER --workers $WORKERS --budget_usd $BUDGET_SCORER \
      --cache $D/cache_$SCORER.json --out $D/predict_${SCORER}_${TAG}.json
  if [ -f $D/labels_${TAG}_b.json ]; then
    python label_agreement.py --labels_a $D/labels_${TAG}_a.json --labels_b $D/labels_${TAG}_b.json \
        --out $D/stability_${TAG}.json || true
  fi
  python label_factor.py --meta $D/meta.json --labels $D/labels_${TAG}_a.json \
      --env_name ${ENV[$e]} --out $D/label_factor_${TAG}.json || true
done
cat experiments/vlm/spend_*.json 2>/dev/null; echo
