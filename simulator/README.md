# Computer Vision Edge Detection Simulator

## Overview
Interactive Streamlit app comparing Sobel, Prewitt, LoG, and Canny edge detection methods with adjustable parameters.

## Features
- Upload image or use built-in sample
- Adjustable blur, noise, and thresholds
- Side-by-side comparison (clean vs noisy)
- Parameter sensitivity insights
- Method comparison table + explanations

## Setup (Run in CLI from project root folder)
conda create -n CV-Project python=3.10
conda activate CV-Project
pip install -r requirements.txt

## Run (Run in CLI from project root folder)
streamlit run simulator/app.py

## Notes
- Noise slider demonstrates failure modes
- Canny generally produces the cleanest edges
- LoG is highly noise-sensitive