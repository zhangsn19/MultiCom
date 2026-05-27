# Data Directory

This directory is intentionally empty.

Download the official Community Notes data from:

https://communitynotes.x.com/guide/en/under-the-hood/download-data

The main files used by this pipeline are:

- `notes/*.tsv`
- `noteStatusHistory/*.tsv`
- `noteRatings/*.tsv` if you want to rebuild rater clusters/personas
- a local `posts.csv` containing `noteId`, `tweetId`, and `post_text`

The official Community Notes release does not always include complete original post text. For the paper experiments, post text was collected separately through platform API access where available.

For the main MultiCom setting, prepare a 16-row `cluster_summary_k16.csv`. We use 16 rater clusters because it performed best after jointly considering Silhouette, Calinski-Harabasz, and Davies-Bouldin diagnostics; see `docs/cluster_selection.md` for the clustering rationale.
