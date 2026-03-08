# Benchmark Results

This document summarizes the first benchmark evaluation of the Structural Network Engine.

The goal of the benchmark is to evaluate the robustness of the algorithm in detecting filament-like structures under increasing levels of noise.

---

# Benchmark Setup

A synthetic branching structure is generated and progressively corrupted with increasing amounts of random noise points.

Two methods are compared:

• Structural Network Engine  
• Baseline graph reconstruction method

The evaluation metrics are:

• Precision  
• Recall  
• F1 Score  
• Execution Time

---

# Baseline vs Engine Example

Baseline:

Precision: 0.980  
Recall: 1.0  
F1 Score: 0.989  

Engine:

Precision: 0.992  
Recall: 1.0  
F1 Score: 0.996  

Result:

The Structural Network Engine produces higher precision while maintaining perfect recall.

---

# Noise Stress Test

The following experiment increases the amount of random noise points.

Noise Levels Tested:

100  
300  
600  
900  
1500  
2500

Results show that:

• both algorithms perform similarly under very low noise  
• the Structural Network Engine becomes significantly more robust as noise increases

At high noise levels the baseline method produces many false positives, while the engine maintains a cleaner structural backbone.

---

# Key Result

Under heavy noise conditions the engine achieves:

Higher precision  
Higher F1 score  
Lower false positive rate

This indicates stronger robustness for detecting structural patterns in noisy spatial datasets.

---

# Interpretation

The benchmark suggests that the Structural Network Engine is particularly well suited for:

• noisy spatial point clouds  
• filament-like structures  
• fragmented networks

Example application domains include:

• cosmology (cosmic filaments)  
• urban road networks  
• vascular structures  
• spatial infrastructure networks

---

# Current Status

These results represent the first validation stage of the Structural Network Engine.

Future work will include:

• additional benchmarks on real datasets  
• scalability tests on larger point clouds  
• comparisons with additional baseline algorithms
