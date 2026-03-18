# A Novel Hybrid KAN-BiLSTM Framework for Remaining Useful Life Prediction of Aircraft Engines

[![Paper](https://img.shields.io/badge/Paper-Quality_and_Reliability_Engineering_International-blue.svg)](https://doi.org/DOI)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the official PyTorch implementation of the **KAN-BiLSTM** framework for aero-engine Remaining Useful Life (RUL) prediction, as described in our research paper.

---

## 🌟 Overview

Predicting the RUL of aircraft engines is critical for operational safety and cost-effective maintenance. Our proposed **KAN-BiLSTM** model leverages:
1. **Kolmogorov-Arnold Network (KAN)**: Utilizing learnable B-spline activation functions on edges to capture complex, fine-grained nonlinear features from multi-channel sensor data.
2. **Bidirectional LSTM (BiLSTM)**: Modeling long-range temporal dependencies from both past and future contexts.
3. **Advanced Mechanisms**: Incorporating **Temporal Positional Embeddings**, **Residual Skip Connections**, and **Layer Normalization** to ensure fast convergence and high prediction stability.

---

## 📊 Experimental Results (C-MAPSS)

To ensure statistical reliability, the model was evaluated over **10 independent runs with different random seeds**. The results (Mean ± Std) on the NASA C-MAPSS benchmark are as follows:

| Dataset | Operating Conditions | Fault Types | RMSE (Our Model) | Score (Our Model) |
| :--- | :---: | :---: | :---: | :---: |
| **FD001** | 1 | 1 | **11.85 ± 0.20** | 258.91 ± 15.55 |
| **FD002** | 6 | 1 | **12.24 ± 0.26** | 505.69 ± 29.00 |
| **FD003** | 1 | 2 | **10.52 ± 0.20** | 179.61 ± 15.27 |
| **FD004** | 6 | 2 | **17.50 ± 0.69** | 1968.32 ± 512.78 |

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/momojiang329/A-Novel-Hybrid-KAN-BiLSTM-Framework-for-Remaining-Useful-Life-Prediction-of-Aircraft-Engines.git](https://github.com/momojiang329/A-Novel-Hybrid-KAN-BiLSTM-Framework-for-Remaining-Useful-Life-Prediction-of-Aircraft-Engines.git)
cd A-Novel-Hybrid-KAN-BiLSTM-Framework-for-Remaining-Useful-Life-Prediction-of-Aircraft-Engines
