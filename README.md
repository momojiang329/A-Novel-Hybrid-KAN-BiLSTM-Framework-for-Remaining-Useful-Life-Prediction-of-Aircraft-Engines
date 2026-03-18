# A Novel Hybrid KAN-BiLSTM Framework for Remaining Useful Life Prediction of Aircraft Engines

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/DOI)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the official PyTorch implementation of the **KAN-BiLSTM** framework for aero-engine Remaining Useful Life (RUL) prediction.

---

## 🌟 Overview
Predicting the RUL of aircraft engines is critical for operational safety. Our proposed **KAN-BiLSTM** model leverages:
1. **Kolmogorov-Arnold Network (KAN)**: For fine-grained nonlinear feature extraction.
2. **Bidirectional LSTM (BiLSTM)**: For capturing complex temporal degradation patterns.
3. **Optimized Mechanisms**: Includes Temporal Positional Embeddings and Residual Skip Connections.

---

## 📊 Experimental Results (C-MAPSS)
Evaluated over **10 independent runs** (Mean ± Std):

| Dataset | RMSE | Score |
| :--- | :---: | :---: |
| **FD001** | 11.85 ± 0.20 | 258.91 ± 15.55 |
| **FD002** | 12.24 ± 0.26 | 505.69 ± 29.00 |
| **FD003** | 10.52 ± 0.20 | 179.61 ± 15.27 |
| **FD004** | 17.50 ± 0.69 | 1968.32 ± 512.78 |

---

## 🛠️ Setup & Usage
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Train model**: `python train.py --dataset FD001`
3. **Evaluate**: `python evaluate.py --model_path ./models/best_model.pt`

---

## 📝 Citation
```bibtex
@article{mo2026kanbilstm,
  title={A Novel Hybrid KAN-BiLSTM Framework for Remaining Useful Life Prediction of Aircraft Engines},
  author={Mo, Wenqi and Tao, Jiyuan and Wang, Guoqiang and Zhang, Xia and Liu, Xintian},
  journal={Quality and Reliability Engineering International},
  year={2026}
}
