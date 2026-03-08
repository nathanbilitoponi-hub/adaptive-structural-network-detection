# Structural Network Engine — Benchmark Report

## Overview

This document reports the first benchmark evaluation of the Structural Network Engine, a system designed to extract structural backbones, trunks and topology from noisy spatial point clouds.

The goal of the benchmark is to evaluate how well the engine performs compared to a simple baseline approach when increasing levels of noise are introduced.

The baseline used is a simple density-based neighbor graph method.

---

# Benchmark Setup

## Dataset

A synthetic dataset with known ground-truth structure was generated.  
The structure contains:

- a main trunk
- multiple branching segments
- spatial jitter

Additional uniform noise points were progressively added to simulate real-world conditions.

Noise levels tested:

- 100 noise points
- 300 noise points
- 600 noise points
- 900 noise points
- 1500 noise points
- 2500 noise points

---

# Evaluation Metrics

The following metrics were used:

- Precision — how many detected points belong to the real structure
- Recall — how much of the real structure was recovered
- F1 score — harmonic mean of precision and recall
- False positives — points incorrectly classified as structure
- Execution time

---

# Results

## Average Performance

| Metric | Engine | Baseline |
|------|------|------|
| Average F1 | 0.923 | 0.898 |
| Average Precision | 0.924 | 0.859 |
| Recall | 1.0 | 1.0 |

The Structural Network Engine demonstrates higher precision and F1 score on average compared to the baseline.

---

## Behavior Under Increasing Noise

When the noise level increases, the advantage of the Structural Network Engine becomes more significant.

Example at high noise:

### Noise = 2500

| Metric | Engine | Baseline |
|------|------|------|
| Precision | 0.687 | 0.459 |
| F1 score | 0.815 | 0.630 |
| False positives | 311 | 810 |

The engine significantly reduces false positives compared to the baseline method.

---

# Interpretation

The benchmark suggests that the Structural Network Engine is particularly effective at:

- extracting structural skeletons from noisy spatial data
- maintaining structural continuity under high noise
- reducing false positives in dense point clouds

This behavior is especially relevant for applications involving:

- geospatial networks
- biological vascular systems
- cosmological filament detection
- noisy point-cloud structural analysis

---

# Conclusion

The Structural Network Engine shows stronger robustness than a simple baseline method under increasing noise levels.

The results suggest that the engine's structural filtering and topology reconstruction pipeline provides improved performance in challenging noisy environments.

Further benchmarking against additional algorithms and real-world datasets will be conducted in future evaluations.
