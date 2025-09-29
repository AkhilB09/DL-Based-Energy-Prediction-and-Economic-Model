# DL-Based-Energy-Prediction-and-Economic-Model
## Project Overview

This repository houses a comprehensive predictive optimization system designed for a bike-sharing network. It integrates advanced **Time Series Forecasting** (LSTM) with a **Micro-Economic Simulation** to determine optimal, profit-maximizing dynamic pricing strategies for individual stations.

The primary methodology involves **DBSCAN Clustering** to segment stations based on rich behavioral profiles. This leads to the development of specialized (clustered) and fine-tuned predictive models that demonstrate superior accuracy and economic performance compared to a single global model benchmark.

### Core Objectives
* **Hierarchical Forecasting:** Train and evaluate specialized LSTM models for behaviorally-clustered stations.
* **Model Specialization:** Implement **Transfer Learning** (fine-tuning) to adapt a pre-trained Global model for each cluster.
* **Economic Simulation:** Utilize high-accuracy forecasts as inputs for a `scipy.optimize` solver to maximize revenue under defined economic constraints.
* **Business Intelligence (BI):** Generate final reports and geographical visualizations to analyze economic profit lift.

---

## 🚀 Execution Workflow (3 Sequential Stages)

The project must be executed in three mandatory stages. The output (trained models/scalers) of each stage is the required input for the next.

| Stage | Script | Purpose | Output Directory |
| :--- | :--- | :--- | :--- |
| **I: Baseline Training** | `python main.py` | Trains the Global and initial Clustered LSTM models. | `./final_run_models_no_lag/` |
| **II: Transfer Learning** | `python transfer_learning.py` | Loads Global model weights from Stage I and fine-tunes them for each cluster. | `./fine_tuned_models/` |
| **III: Simulation & Report** | `python simulator.py` | Loads the final fine-tuned models, executes the 24-hour economic optimization, and generates all dashboard outputs. | `./fine_tuned_reports/` |

---

## 🛠️ Setup and Installation
### 2. Environment Setup

1.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:** Use the provided list of packages.
    ```bash
    pip install -r requirements.txt
    ```

    *Note: For complex dependencies like `torch-geometric`, users may need to manually install specific wheels for their environment (e.g., CUDA version).*

---

## 💻 Running the Scripts

Execute the scripts directly from the repository root in the following order:

```bash
# 1. Train Base Models and Generate Scalers
python main.py

# 2. Execute Transfer Learning Fine-Tuning
python transfer_learning.py

# 3. Run the Final Economic Simulation and Dashboard Generation
python simulator.py

### 1. Data Preparation

All raw input data files must be placed within the **`data/raw/`** director
