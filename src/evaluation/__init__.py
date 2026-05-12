"""
Evaluation package for computing metrics, caching results, and analyzing fairness.

Public API
----------
- :mod:`src.evaluation.evaluate`          — PCC, RMSE metrics + orchestration
- :mod:`src.evaluation.cache`             — SQLite MD5 result cache (model/split identifiable)
- :mod:`src.evaluation.fairness`          — stratified fairness analysis
- :mod:`src.evaluation.score_comparison`  — cross-metric prediction-accuracy comparison
- :mod:`src.evaluation.charts`            — publication-quality evaluation charts
"""
