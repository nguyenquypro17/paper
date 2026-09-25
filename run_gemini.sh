#!/usr/bin/env bash
# run_gemini.sh -- VLM labelling protocol: namer Gemini 3.1 Pro, scorer DeepSeek (other vendor).
# Run from ~/XRL, conda activate lucid, export GEMINI_API_KEY=... DEEPSEEK_API_KEY=...
#
#   bash run_gemini.sh doorkey                # one env
#   bash run_gemini.sh                        # all envs
#   SCORER=deepseek-flash WORKERS=8 bash run_gemini.sh doorkey
#
# Scorer settings are chosen from the positive control ONLY, never from label results.
# Every step caches its API responses: re-running after a crash costs nothing for done work.
set -e
SCORER=${SCORER:-deepseek-flash}
WORKERS=${WORKERS:-8}
declare -A ENV=( [doorkey]=MiniGrid-DoorKey-6x6-v0 [dyobs]=MiniGrid-Dynamic-Obstacles-5x5-v0
                 [cartpole]=PixelCartPole [pong]=PongNoFrameskip-v4 [boxing]=BoxingNoFrameskip-v4 )
declare -A PPO=( [doorkey]=ppo_doorkey_6x6.zip [dyobs]=ppo_dynamic_obs_5x5.zip
                 [cartpole]=ppo_pixel_cartpole.zip [pong]=pong_sb3_290.zip [boxing]=boxing_sb3_290.zip )
declare -A EPS=( [doorkey]="--n_episodes 2000" [dyobs]="--n_episodes 2000" [cartpole]="--n_episodes 20 --stride 5"
                 [pong]="--n_episodes 6 --stride 4" [boxing]="--n_episodes 6 --stride 4" )

for e in ${@:-doorkey dyobs cartpole pong boxing}; do
  D=experiments/vlm/${e}_42
  echo "=================== $e ==================="
  # 0) render (skipped if already done; delete $D to re-render)
  if [ ! -f $D/meta.json ]; then
    python render_concepts.py --model_path outputs/lucid_model_${e}_42/sae_logic_joint_model.pt \
        --ppo_path ${PPO[$e]} --env_name ${ENV[$e]} ${EPS[$e]} --seed 2000 --out_dir $D
  fi
  # 1) naming: two independent samples (temperature 1.0) -> label stability
  for s in a b; do
    [ -f $D/labels_gemini31_$s.json ] || python vlm_name.py --img_dir $D/naming --env_name ${ENV[$e]} \
        --model gemini-3.1-pro-preview --out $D/labels_gemini31_$s.json
  done
  # 2) positive control + 3) blind scoring of sample a (one call, shared cache)
  python vlm_predict.py --test_dir $D/test --labels $D/labels_gemini31_a.json \
      --env_name ${ENV[$e]} --model $SCORER --workers $WORKERS \
      --cache $D/cache_$SCORER.json --out $D/predict_$SCORER.json
  # 4) stability of names across the two samples, 5) label <-> simulator factor
  python label_agreement.py --labels_a $D/labels_gemini31_a.json --labels_b $D/labels_gemini31_b.json \
      --out $D/stability_gemini31.json || true          # needs >= 3 labelled concepts
  python label_factor.py --meta $D/meta.json --labels $D/labels_gemini31_a.json \
      --env_name ${ENV[$e]} --out $D/label_factor_gemini31.json || true   # Atari: no factors
done
