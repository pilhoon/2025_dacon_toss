# External Research Topics for Plan4

| Priority | Topic | Why We Need It | Desired Output | Potential Sources |
|----------|-------|----------------|----------------|-------------------|
| P0 | Official competition metric definition (AP/WLL weights, WLL formula) | Notes conflict between Plan1 (0.5/0.5) and Plan3 (0.7/0.3). Without clarity we cannot align offline/online scores. | Verified formula + example calculation mirroring leaderboard | DACON competition overview/FAQ, forum announcements |
| P0 | Weighted LogLoss implementation details | Need to know how positives/negatives are re-weighted (50:50? sample weights?) to match LB. | Reference implementation / pseudocode | DACON metric documentation, previous competitions, community repos |
| P1 | Feature semantics for `l_feat_*`, `feat_[a-e]_*`, `history_*` | Better understanding may drive targeted feature engineering/aggregation. | Feature dictionary or discussion posts clarifying meaning and ranges. | DACON Q&A, organizers' data description, discussion boards |
| P1 | Successful CTR calibration strategies in recent competitions | Helps balance AP vs WLL without sacrificing leaderboard score. | Short list of methods (e.g., beta calibration, isotonic for imbalanced CTR). | Academic papers, Kaggle/Dacon discussions |
| P1 | Sequence-based CTR architectures handling long categorical sequences | To refine DIN-lite and other models using `seq`. Need proven techniques for truncation, attention, embedding initialization. | Best practices + sample configs | Research papers (DIN, DIEN, BST), blog posts, open-source repos |
| P2 | Ensemble strategies optimizing precision-recall trade-offs | Score weights AP heavily; need ensemble methods that improve ranking while keeping calibrated probabilities. | Techniques like rank averaging, stacking, power mean weighting. | Kaggle solution write-ups, ML blogs |
| P2 | Efficient generation of time-window CTR aggregates on 10M+ rows | Performance considerations for feature engineering (Plan3 pipeline cost). | Recipes for PyArrow/Polars/Spark pipelines | Big-data feature engineering articles |
| P3 | Monitoring frameworks for offline vs online metric drift | Ensure we detect shifts when new submissions behave differently. | Tooling suggestions (e.g., EvidentlyAI, Great Expectations) | MLOps blogs, open-source docs |

> Note: Network access is restricted in the current environment. Capture findings offline and cite sources in future documentation when available.
