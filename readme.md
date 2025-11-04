# 🎯 ID Card Detector using Machine Learning

A computer vision project that detects whether a person is wearing an ID card using webcam input.

## 🚀 How it Works
1. `generate_dataset.py` extracts features from images of people **with** and **without** ID cards.
2. `train_model.py` trains a machine learning model using Gradient Boosting.
3. `predict_test.py` uses your webcam to classify whether the person has an ID card in real-time.

## 🧩 Installation
```bash
git clone https://github.com/yourusername/id-card-detector.git
cd id-card-detector
pip install -r requirements.txt
