# ML-Based-Energy-Prediction-and-Economic-Model
Project Overview
This repository houses a comprehensive predictive optimization system designed for a bike-sharing network. The framework utilizes Time Series Forecasting (LSTM) integrated with a Micro-Economic Simulation to determine optimal, profit-maximizing dynamic pricing strategies for individual stations.

The core methodology relies on DBSCAN Clustering to segment stations based on rich behavioral profiles, allowing for the development of specialized and highly accurate predictive models that significantly improve upon a single global model benchmark.

Core Objectives
Hierarchical Forecasting: Train and evaluate specialized LSTM models for behaviorally-clustered stations versus a global benchmark.

Model Specialization: Implement Transfer Learning (fine-tuning) to achieve superior predictive accuracy with efficient training.

Profit Maximization: Use the high-accuracy forecasts as inputs for a scipy.optimize solver to maximize revenue under defined cost and demand elasticity constraints.

Business Intelligence (BI): Generate final reports and geographical visualizations to analyze economic performance and profit lift.

 Execution Workflow (3 Sequential Stages)
The project is structured into three mandatory, sequential execution stages. Each stage's output is required as the input for the subsequent stage.

Stage	Script	Purpose	Output Directory
I: Baseline Training	python main.py	Trains the Global and initial Clustered LSTM models. This stage generates the weights needed for the transfer learning starting point.	./final_run_models_no_lag/
II: Transfer Learning	python transfer_learning.py	Loads the Global model weights and fine-tunes them for each cluster using the cluster-specific training data and hyperparameters.	./fine_tuned_models/
III: Simulation & Report	python simulator.py	Loads the final fine-tuned models, executes the 24-hour economic simulation (heuristic vs. optimized pricing), and generates the BI Dashboard outputs.	./fine_tuned_reports/

Export to Sheets
 Setup and Installation
1. Data Preparation
All raw input data files must be placed within the data/raw/ directory, as defined in config.py.

your-project-repo/
└── data/
    └── raw/
        ├── information.csv
        ├── volume.csv
        ├── ... (all time series and static files)
        └── SZ_districts.shp
2. Environment Setup
The project requires several deep learning and scientific libraries.

Create and activate a virtual environment:

Bash

python -m venv venv
source venv/bin/activate
Install dependencies: Use the provided list of packages.

Bash

pip install -r requirements.txt
Note: If you encounter issues with torch-geometric dependencies (torch-sparse, torch-scatter), please consult the official PyTorch Geometric documentation to install the wheels corresponding to your specific PyTorch and CUDA version.

 Running the Scripts
Execute the scripts directly from the repository root, ensuring your virtual environment is active.

Bash

# 1. Train Base Models and Generate Scalers
python main.py

# 2. Execute Transfer Learning Fine-Tuning
python transfer_learning.py

# 3. Run the Final Economic Simulation and Dashboard Generation
python simulator.py
 Repository Structure
The project follows a modular structure to ensure maintainability and reproducibility.

File/Folder Name	Category	Description
config.py	Configuration	Centralized source of truth for all constants: model hyperparameters, file paths, and economic simulation parameters.
utils.py	Library	Reusable functions including Model Definitions (SimpleLSTM), Data Pipeline (loading, clustering), and Economic Formulae.
main.py	Execution	Stage I training script (Global and Clustered benchmark).
transfer_learning.py	Execution	Stage II fine-tuning script.
simulator.py	Execution	Stage III simulation and reporting script.
requirements.txt	Documentation	List of all required dependencies.
data/raw/	Input	Directory for all source data files.
final_run_models_no_lag/	Output/Artifact	Stores the saved models (.pth) and normalization scalers (.joblib) from main.py.
fine_tuned_models/	Output/Artifact	Stores the final, optimized models from transfer_learning.py.
fine_tuned_reports/	Output/Report	Stores the final simulation metrics and reports (e.g., 04_transfer_learning_summary.csv).
