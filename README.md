# MultiCom

This repository contains the release code for the main MultiCom experiment: persona-guided multi-agent Community Notes rating and leakage-safe out-of-fold aggregation.

It does not include the ComRate dataset, raw Community Notes data, post text, model outputs, API keys, or paper-only temporary artifacts.

## What Is Included

- Persona construction from rater-cluster summaries.
- Multi-agent structured note rating with the official Community Notes rating schema.
- Feature construction from `agent_votes.csv`.
- Nested out-of-fold aggregation with logistic calibration.
- Final hard ensemble and conservative promotion rule.
- Minimal data-preparation utilities and run scripts.

## Data

Download the official Community Notes data from X:

https://communitynotes.x.com/guide/en/under-the-hood/download-data

The official release is expected to include files such as:

- `notes/*.tsv`
- `noteStatusHistory/*.tsv`
- `noteRatings/*.tsv`
- `userEnrollment/*.tsv`

The original post text is not fully available in the official Community Notes release. To run the same type of note-evaluation experiment, prepare a local CSV with:

```text
noteId,tweetId,post_text
```

## Expected Input Format

The multi-agent runner expects an evaluation CSV with at least:

```text
noteId,tweetId,currentStatus,true_label_3way,true_label_text,post_text,note_text
```

Labels use:

```text
0 = NOT_HELPFUL
1 = NEEDS_MORE_RATINGS
2 = HELPFUL
```

You can build a starter evaluation file from official notes/status files and your post-text table:

```bash
python -m multicom.data_prep make-eval \
  --notes-tsv data/extracted_communitynotes/notes/notes-00000.tsv \
  --status-tsv data/extracted_communitynotes/noteStatusHistory/noteStatusHistory-00000.tsv \
  --posts-csv data/posts.csv \
  --out data/eval_notes.csv
```

If the official files are still zipped:

```bash
python -m multicom.data_prep extract \
  --raw-dir data/raw_communitynotes \
  --out-dir data/extracted_communitynotes
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running The Main Pipeline

Prepare:

- `data/eval_notes.csv`
- `data/cluster_summary_k16.csv`

The cluster summary should contain one row per rater cluster. The release script accepts common fields such as `cluster`, `share_helpful`, `share_not_helpful`, `bw_rater_agree_ratio`, and `bw_mean_note_score`.

The main paper configuration uses a fixed 16-persona panel. We use 16 rater clusters because it performed best after jointly considering three clustering diagnostics in the final `K=2..32` search: Silhouette, Calinski-Harabasz, and Davies-Bouldin. To reproduce the reported main experiment, use a 16-row cluster summary or rerun the rater-clustering step with `n_clusters=16`. See [docs/cluster_selection.md](docs/cluster_selection.md) for the clustering rationale.

Set your model API credentials:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5.1
```

The main release defaults are:

| Parameter | Default | Override | Meaning |
|---|---:|---|---|
| `num_persona_agents` | `16` | `MULTICOM_NUM_PERSONA_AGENTS` | Number of rater clusters/persona agents. The paper's main setting fixes this at 16. |
| `model` | `gpt-5.1` | `OPENAI_MODEL` | Model used by every persona agent. |
| `base_url` | `https://api.openai.com/v1` | `OPENAI_BASE_URL` | Chat-completions compatible API endpoint. |
| `temperature` | `0.7` | `MULTICOM_TEMPERATURE` | Agent decoding temperature. |
| `max_tokens` | `420` | `MULTICOM_MAX_TOKENS` | Maximum output tokens for each agent prediction. |
| `concurrency` | `8` | `MULTICOM_CONCURRENCY` | Number of parallel agent API calls. |
| `max_retries` | `3` | `MULTICOM_MAX_RETRIES` | Retry count for failed agent API calls. |
| `outer_folds` | `5` | `MULTICOM_OUTER_FOLDS` | Outer folds for leakage-safe OOF aggregation. |
| `inner_folds` | `5` | `MULTICOM_INNER_FOLDS` | Inner folds used inside each outer training split for regularization, class-weight, and decision-threshold selection. |
| `seed` | `42` | `MULTICOM_SEED` | Random seed for OOF splits and calibration models. |

The aggregation follows the paper setting: a 5-fold stratified outer out-of-fold evaluation, with an inner 5-fold split inside each outer training fold for selecting logistic regularization strength, class weights, and decision thresholds. Each held-out note is predicted by models that were not trained on that note.

Then run:

```bash
bash scripts/run_pipeline.sh
```

This creates:

```text
artifacts/personas.csv
artifacts/main_run/pilot_notes.csv
artifacts/main_run/agent_votes.csv
artifacts/main_run/oof_aggregation/oof_predictions.csv
artifacts/main_run/oof_aggregation/summary.csv
```

## Running Steps Manually

Build personas:

```bash
python -m multicom.personas \
  --cluster-summary data/cluster_summary_k16.csv \
  --out artifacts/personas.csv \
  --expected-clusters 16
```

Run multi-agent ratings:

```bash
mkdir -p artifacts/main_run
cp data/eval_notes.csv artifacts/main_run/pilot_notes.csv

python -m multicom.agent_rating \
  --notes data/eval_notes.csv \
  --personas artifacts/personas.csv \
  --out artifacts/main_run/agent_votes.csv \
  --model gpt-5.1 \
  --temperature 0.7 \
  --max-tokens 420 \
  --concurrency 8 \
  --max-retries 3
```

Run OOF aggregation:

```bash
python -m multicom.oof_aggregation \
  --run-dir artifacts/main_run \
  --out-dir artifacts/main_run/oof_aggregation \
  --folds 5 \
  --inner-folds 5 \
  --seed 42
```

## Algorithm Defaults

The released main pipeline is configured as a 16-cluster rater simulation problem. `multicom.personas` checks that `cluster_summary_k16.csv` has exactly 16 rows by default, then creates one persona agent per row. If you intentionally run an ablation with another number of agents, pass `--expected-clusters 0` and report that setting separately.

## Rater Clustering And Prompt Construction

MultiCom builds one persona prompt for each rater cluster. Raters are clustered from MFCore-style rater representations and behavioral features, then each row in `cluster_summary_k16.csv` is converted into one cluster-specific persona. The generated persona tells the agent to follow the official Community Notes rating rubric while letting the cluster profile shape its strictness, evidence expectations, and sensitivity to missing context.

For each post-note pair, the agent receives two chat messages:

- `system`: declares the model as a simulated Community Notes rater, inserts the cluster-specific persona, and requires exactly one JSON object with no markdown or extra prose.
- `user`: provides the task instruction, allowed helpfulness labels, 0/1 reason-field format, 0-100 confidence/diagnostic scores, the required JSON keys, the original post text, and the community note text.

The rater clustering method, prompt construction rules, and 16 cluster-specific persona blocks are documented in [docs/prompt_construction.md](docs/prompt_construction.md). The release prompt template is implemented in `multicom.agent_rating.build_prompt`, and persona construction is implemented in `multicom.personas.build_system_prompt`.

Each persona agent returns one structured JSON rating with:

- one helpfulness label: `HELPFUL`, `SOMEWHAT_HELPFUL`, or `NOT_HELPFUL`;
- agreement signals: `agree`, `disagree`;
- helpfulness reasons: `helpfulClear`, `helpfulGoodSources`, `helpfulAddressesClaim`, `helpfulImportantContext`, `helpfulUnbiasedLanguage`;
- not-helpfulness reasons: `notHelpfulIncorrect`, `notHelpfulSourcesMissingOrUnreliable`, `notHelpfulMissingKeyPoints`, `notHelpfulHardToUnderstand`, `notHelpfulArgumentativeOrBiased`, `notHelpfulIrrelevantSources`, `notHelpfulOpinionSpeculation`, `notHelpfulNoteNotNeeded`;
- diagnostic scores: `confidence` and `changes_reader_understanding`, each on a 0-100 scale.

The OOF aggregator converts the 16 agent outputs into note-level vote, confidence, reason, failure-mode, disagreement, and metadata features. It then trains nested out-of-fold logistic calibration models and combines the paper-style hard-ensemble anchors `oof_ensemble_weighted`, `oof_ensemble_gated`, `xstyle_rescue_gate`, `blend`, and `full_meta`, followed by a conservative promotion rule for selected `NEEDS_MORE_RATINGS` cases.

## Notes On Reproducibility

The paper experiments used persona clusters estimated from historical Community Notes rater behavior and a 16-agent main configuration. The release code exposes the same algorithmic structure but leaves data collection, post-text availability, and model choice to the user.

The OOF aggregator is leakage-safe: held-out predictions are generated by models that did not train on the held-out notes. For the paper setting, use `--folds 5 --inner-folds 5`. The final `multicom_final` prediction combines multiple OOF views and applies a conservative promotion rule for selected `NEEDS_MORE_RATINGS` predictions.

If you re-download the official Community Notes data and independently run clustering, you may obtain a different statistically preferred number of clusters depending on the date range, filtering thresholds, features, initialization seed, and selection metric. That does not invalidate reproduction of the main MultiCom setting. For the paper setting, keep the downstream persona panel fixed at 16 agents.

## Repository Hygiene

The `.gitignore` excludes data, artifacts, model outputs, virtual environments, and local secrets.
