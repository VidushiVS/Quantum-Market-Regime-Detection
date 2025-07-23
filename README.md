# Quantum-Market-Regime-Detection
Quantum machine learning classifier to detect market regimes (bull, bear, sideways, high volatility) using variational quantum circuits.

Market Regimes

Bull: Rising markets with positive momentum
Bear: Falling markets with negative trend
Sideways: Flat markets with low directional movement
High Vol: High volatility periods regardless of direction

Features
Uses 8 market features:

Average returns and volatility
Trend strength and momentum
Correlation breakdown
Maximum drawdown
Return skewness
Volatility clustering

Quantum Circuit

Feature Map: ZZ feature encoding
Ansatz: Real amplitude variational circuit
Optimizer: COBYLA for parameter optimization
Backend: Qiskit Aer simulator

Model Comparison
Compares quantum classifier against:

Random Forest
Support Vector Machine
Classical optimization benchmarks

Analysis Features

Regime transition probability matrix
Quantum advantage metrics
Circuit depth and parameter analysis
Prediction confidence scores
