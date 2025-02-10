# AttenGluco: Transformer-Based Blood Glucose Forecasting

## Overview
This repository contains the source code for **AttenGluco**, a Transformer-based framework for blood glucose level (BGL) forecasting using multimodal wearable sensor data. The model is designed to handle time-series data efficiently, leveraging **cross-attention** and **multi-scale attention** mechanisms to enhance predictive performance.

## Features
- **Multimodal Data Fusion**: Integrates CGM readings, activity data, and physiological signals.
- **Transformer-Based Forecasting**: Utilizes attention mechanisms to capture long-term dependencies.
- **Cohort-Wise Learning**: Supports **person-wise**, **subject-wise fine-tuning**, and **continual learning** scenarios.
- **Baseline Comparisons**: Evaluates performance against LSTM and 1D-CNN-based models.
- **Flexible Prediction Horizons**: Supports forecasting for **5-minute, 30-minute, and 60-minute** intervals.

## Installation
To set up the environment and run the code, install the required dependencies:

```bash
git clone https://github.com/yourusername/AttenGluco.git
cd AttenGluco
pip install -r requirements.txt
