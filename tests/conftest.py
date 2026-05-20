# Test suite configuration for the Live AAPL Forecasting system.
#
# Structure:
#   tests/
#     conftest.py          — shared fixtures and pytest configuration (this file)
#     test_live_fetcher.py — unit + property tests for data/live_fetcher.py
#     test_feature_builder.py — unit + property tests for features/feature_builder.py
#     test_sequence_builder.py — unit + property tests for models/sequence_builder.py
#     test_scaler_manager.py  — unit + property tests for models/scaler_manager.py
#     test_attention.py       — unit + property tests for models/attention.py
#     test_forecaster.py      — unit + property tests for inference/forecaster.py
#     test_comparator.py      — unit + property tests for evaluation/comparator.py
#
# Property-based tests use the `hypothesis` library (>=6,<7).
# Unit tests use pytest.
