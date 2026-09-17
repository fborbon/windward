"""Demonstrative time-series forecasting showcase: the naive-to-foundation-model taxonomy of
wind production forecasting techniques, evaluated on the same Kelmarsh series/split as
forecasting/ and demo_notebook/. See taxonomy.py for scope and README.md §13 for context.

This is deliberately kept separate from forecasting/ (the production regression-on-weather
model) - that module answers "what will this farm produce", tracked/served/registered like any
production ML system. This module answers a different, complementary question a lot of wind/
energy DS roles specifically screen for: "which time-series forecasting paradigm fits this kind
of signal, and why" - a breadth showcase across the field, not a second production candidate.
"""
