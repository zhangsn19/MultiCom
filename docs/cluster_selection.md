# Why The Main Pipeline Uses 16 Persona Agents

The main MultiCom experiment uses a fixed 16-persona panel by default. In the release code, this corresponds to a 16-row `data/cluster_summary_k16.csv`, where each row is converted into one persona prompt and therefore one simulated rater agent.

This value comes from the final rater-clustering diagnostic used by the main pipeline. We jointly considered Silhouette, Calinski-Harabasz, and Davies-Bouldin scores and found `K=16` to be the best default cluster count.

## Determination Process

1. Learn a rater representation from historical Community Notes ratings.

   We first fit a biased rank-one matrix-factorization model over contributor-note ratings:

   ```text
   r_ij ~= mu + alpha_i + beta_j + u_i v_j
   ```

   The rater-side representation combines the learned contributor intercept/factor with behavioral statistics such as agreement tendency and mean rated-note score.

2. Search candidate cluster counts with three internal clustering metrics.

   The final pipeline searched `K=2..32` using the official MFCore rater output and the following feature columns:

   ```text
   internalRaterIntercept
   internalRaterFactor1
   internalFirstRoundRaterIntercept
   internalFirstRoundRaterFactor1
   crhCrnhRatioDifference
   meanNoteScore
   raterAgreeRatio
   aboveHelpfulnessThreshold
   ```

   The three diagnostics were:

   - Silhouette coefficient: larger is better.
   - Calinski-Harabasz score: larger is better.
   - Davies-Bouldin score: smaller is better.

   Based on these three diagnostics, we used `K=16` as the final rater-cluster count. In particular, `K=16` achieved the strongest separation under the primary silhouette criterion in the final `K=2..32` search, while also preserving enough behavioral granularity for persona construction. We therefore fixed 16 as the default MultiCom persona panel size.

3. Fix the main persona panel to 16 agents.

   After selecting `K=16`, we generated `cluster_summary_k16.csv`. Each cluster summary row becomes one agent persona. The release code fixes this value so that downstream persona generation, agent voting, and OOF aggregation reproduce the paper's main configuration.

4. Check agent-count sensitivity.

   We compared larger panels in the agent-count analysis:

   | Agent panel | Accuracy | Balanced accuracy | Macro-F1 |
   |---|---:|---:|---:|
   | 16 agents | 84.7 | 68.3 | 60.1 |
   | 32 agents | 85.1 | 64.5 | 58.9 |
   | 48 agents | 88.5 | 59.5 | 60.2 |

   The 16-agent panel was selected as the main default because it gave strong balanced accuracy and macro-F1 while being much cheaper than 32- or 48-agent inference.

## What To Do When Reproducing

For the main-paper setting:

```text
Use n_clusters = 16
Use a 16-row cluster_summary_k16.csv
Generate one persona per row
Run the OOF aggregator on the resulting agent_votes.csv
```

If you rerun clustering after downloading a newer Community Notes release, a metric such as silhouette may choose another value. That is expected because clustering metrics depend on the data snapshot, filtering rules, feature set, and random seed. To reproduce MultiCom's main reported results, keep the downstream persona panel fixed at 16.

If your goal is a new ablation rather than exact reproduction, you can try other values of `K`. In that case, report the agent count explicitly and treat it as a different MultiCom configuration.
