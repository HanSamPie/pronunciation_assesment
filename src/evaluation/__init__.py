"""
Evaluation package for computing metrics, caching results, and analyzing fairness.

Public API
----------
- :mod:`src.evaluation.evaluate`          — PCC, RMSE, SRC metrics
- :mod:`src.evaluation.cache`             — SQLite MD5 result cache
- :mod:`src.evaluation.fairness`          — stratified fairness analysis
- :mod:`src.evaluation.score_comparison`  — cross-metric prediction-accuracy comparison
"""

#from src.evaluation.score_comparison import (  # noqa: F401
#    compare_score_predictions,
#    summarise_by_level,
#    format_report,
#)
