#!/usr/bin/env bash
set -euo pipefail

# Example end-to-end run after data/eval_notes.csv and data/cluster_summary_k16.csv exist.

python -m multicom.personas \
  --cluster-summary data/cluster_summary_k16.csv \
  --out artifacts/personas.csv \
  --expected-clusters "${MULTICOM_NUM_PERSONA_AGENTS:-16}"

mkdir -p artifacts/main_run
cp data/eval_notes.csv artifacts/main_run/pilot_notes.csv

python -m multicom.agent_rating \
  --notes data/eval_notes.csv \
  --personas artifacts/personas.csv \
  --out artifacts/main_run/agent_votes.csv \
  --model "${OPENAI_MODEL:-gpt-5.4-nano}" \
  --temperature "${MULTICOM_TEMPERATURE:-0.7}" \
  --max-tokens "${MULTICOM_MAX_TOKENS:-420}" \
  --concurrency "${MULTICOM_CONCURRENCY:-8}" \
  --max-retries "${MULTICOM_MAX_RETRIES:-3}"

python -m multicom.oof_aggregation \
  --run-dir artifacts/main_run \
  --out-dir artifacts/main_run/oof_aggregation \
  --folds "${MULTICOM_OUTER_FOLDS:-5}" \
  --inner-folds "${MULTICOM_INNER_FOLDS:-4}" \
  --seed "${MULTICOM_SEED:-42}"
