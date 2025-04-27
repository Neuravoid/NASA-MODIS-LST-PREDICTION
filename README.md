# Marmara LST Prediction

This project focuses on predicting Land Surface Temperature (LST) across the Marmara Region by utilizing NASA’s MODIS products (MOD11A1 and MCD43A3). Through comprehensive data preparation and advanced machine learning techniques, we aimed to model LST behavior accurately at the pixel level.

## Project Overview

- Goal: Predict daily LST values across the Marmara Region with high precision.
- Data Sources: 
  - MODIS MOD11A1 (Daily LST and Emissivity)
  - MODIS MCD43A3 (Albedo and BRDF parameters)
- Study Area: Marmara Region, Turkey

## Dataset Preparation

A rich geospatial dataset was created, including the following variables:

- Vegetation Indices: NDVI, EVI, NDWI
- Surface Parameters: Emissivity, Albedo
- Sun/View Geometry: Sun angle, View angle
- Topographic Features: Elevation, Slope, Aspect (derived from DEM)

Additional derivative variables were engineered:

- LST_Diff: Difference in LST across consecutive days
- Albedo_Diff: Difference in albedo across consecutive days

The final dataset was preprocessed with pixel-level alignment, scaling, and missing data handling.

## Machine Learning Models

Various algorithms were trained and evaluated:

- Linear Models: Linear Regression, Ridge Regression, Lasso Regression, Partial Least Squares (PLS)
- Tree-Based Models: K-Nearest Neighbors (KNN), Decision Tree, Random Forest
- Ensemble and Advanced Models: Gradient Boosting, XGBoost, Support Vector Machine (SVM), Artificial Neural Networks (ANN)

## Feature Engineering and Dimensionality Reduction

- Dimensionality Reduction: Principal Component Analysis (PCA), Partial Least Squares (PLS)
- Statistical Significance Tests: Performed to validate feature relevance
- Feature Importance Analysis: Conducted to interpret model results

## Model Evaluation

Each model was evaluated based on:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Visual inspection of predicted versus observed LST maps

## Tools and Technologies

- Python
- Pandas, NumPy
- scikit-learn
- MODIS HDF and GeoTIFF Processing
- Remote Sensing Techniques
- Machine Learning and Statistical Analysis
- PCA and PLS modeling
- Artificial Neural Networks (ANN)

## Repository Structure

```
├── data/              # Raw and processed geospatial datasets
├── notebooks/         # Jupyter notebooks for data processing and modeling
├── models/            # Saved models and evaluation reports
├── scripts/           # Python scripts for preprocessing and feature engineering
├── README.md          # Project documentation
├── requirements.txt   # Required Python packages
└── results/           # Visualizations and analysis outputs
```

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/Neuravoid/NASA-MODIS-LST-PREDICTION.git
cd marmara-lst-prediction
pip install -r requirements.txt
```

## Usage

Example to train and evaluate a model:

```python
from models.train import train_model
train_model(model_type="RandomForest", dataset_path="data/processed/marmara_lst.csv")
```

## Acknowledgements

- NASA LP DAAC for MODIS data products
- Open-source Python community for remote sensing and machine learning libraries
