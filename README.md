# MLOps Premium Prediction Project

![MLOps Pipeline](https://img.shields.io/badge/MLOps-Pipeline-blue)
![Kubernetes](https://img.shields.io/badge/Orchestration-Kubernetes-blue)
![Segmented Regression](https://img.shields.io/badge/Model-Segmented_Regression-brightgreen)
![Docker](https://img.shields.io/badge/Container-Docker-blue)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-brightgreen)
![Monitoring](https://img.shields.io/badge/Monitoring-Prometheus_Grafana-orange)

A complete end-to-end MLOps solution for insurance premium prediction using segmented regression models with Kubernetes orchestration.

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Development](#-development)
- [Deployment](#-deployment)
- [Monitoring](#-monitoring)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## 🔍 Overview

This project implements a comprehensive MLOps workflow for predicting insurance premiums. The system uses a segmented approach, training separate models for different premium ranges to improve prediction accuracy. The entire pipeline is containerized and can be deployed using Kubernetes, with monitoring, experiment tracking, and model versioning built-in.

### ✨ Why Segmented Regression?

Insurance premiums often exhibit different patterns across price ranges. By training specialized models for each segment (e.g., very low, low, medium, high, very high premiums), we can achieve better prediction accuracy compared to a single model approach.

## 🏗️ Architecture

The system consists of several interconnected components forming a complete MLOps pipeline:

```
                     ┌─────────────┐
                     │             │
                     │  Raw Data   │
                     │             │
                     └──────┬──────┘
                            │
                            ▼
┌─────────────────────────────────────────────┐
│           Data Processing Pipeline           │
│  ┌─────────┐    ┌───────────┐    ┌────────┐ │
│  │   DVC    │───►Validation &│───►Feature  │ │
│  │(Version  │    │Preprocessing│   │Engineering│
│  │ Control) │    └───────────┘    └────────┘ │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐       ┌────────────────┐
│           Model Training Pipeline            │       │                │
│  ┌─────────┐    ┌───────────┐    ┌────────┐ │       │     MLflow     │
│  │ Segment │───►│Feature    │───►│ Model  │ │◄──────┤  Experiment    │
│  │ Creation│    │Selection  │    │Training│ │       │   Tracking     │
│  └─────────┘    └───────────┘    └────────┘ │       │                │
└──────────────────────┬──────────────────────┘       └────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│               Model Registry                 │
│  ┌─────────┐    ┌───────────┐    ┌────────┐ │
│  │ Version │───►│Validation &│───►│Artifact│ │
│  │ Control │    │Comparison  │    │Storage │ │
│  └─────────┘    └───────────┘    └────────┘ │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│           Deployment Pipeline                │
│  ┌─────────┐    ┌───────────┐    ┌────────┐ │
│  │Container │───►│Kubernetes │───►│FastAPI │ │
│  │  Build   │    │Deployment │    │Service │ │
│  └─────────┘    └───────────┘    └────────┘ │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│           Monitoring & Alerting              │
│  ┌─────────┐    ┌───────────┐    ┌────────┐ │
│  │Prometheus│───►│  Grafana  │───►│Alerting│ │
│  │ Metrics  │    │Dashboards │    │System  │ │
│  └─────────┘    └───────────┘    └────────┘ │
└─────────────────────────────────────────────┘
```

## 🌟 Features

- **Segmented Modeling**: Trains specialized models for different premium ranges to improve accuracy
- **Automated Pipeline**: Complete data-to-deployment workflow with minimal manual intervention
- **Experiment Tracking**: MLflow integration for tracking model parameters, metrics, and artifacts
- **Containerized Deployment**: Docker containers for consistent, reproducible deployment
- **Kubernetes Orchestration**: Scalable and resilient deployment with Kubernetes
- **Monitoring & Alerting**: Prometheus and Grafana for real-time monitoring of model and system performance
- **Model Registry**: Versioning for models with comparison and approval workflows
- **CI/CD Ready**: Infrastructure for continuous integration and deployment

## 🚀 Installation

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Kubernetes (Docker Desktop with Kubernetes enabled or a separate cluster)
- Git

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/your-username/premium-prediction-mlops.git
cd premium-prediction-mlops

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source venv/bin/activate
```

### Manual Setup

If you prefer to set up the project manually:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create required directories
mkdir -p data/{raw,processed} models logs artifacts

# Initialize DVC
dvc init
```

## 📊 Usage

### Data Preparation

```bash
# Place training data in data/raw directory
# Track it with DVC
dvc add data/raw/train_data.csv
git add data/raw/train_data.csv.dvc
git commit -m "Add training data"
```

### Training Models

```bash
# Edit configuration in config/config.yaml if needed
# Run training
python -m src.train

# View experiment results in MLflow
mlflow ui
# Open http://localhost:5000 in your browser
```

### Local Deployment

```bash
# Build and start services with Docker Compose
docker-compose up -d

# API documentation available at
# http://localhost:8000/docs
```

### Making Predictions

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
        "features": {
            "Age": 35,
            "Vehicle_Age": 5,
            "Credit_Score": 720,
            "Annual_Income": 65000,
            "Previous_Claims": 1,
            "Insurance_Duration": 3
        }
     }'
```

## 📁 Project Structure

```
End_to_End_MLOPs_regression/
├── config/                # Configuration files
│   └── config.yaml        # Main configuration
├── data/                  # Data directory (managed by DVC)
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── kubernetes/            # Kubernetes deployment manifests
├── models/                # Trained model storage
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── api/               # API service
│   │   └── app.py         # FastAPI application
│   ├── data/              # Data processing code
│   │   └── data_processor.py
│   └── models/            # Model-related code
│       ├── model_trainer.py
│       └── model_predictor.py
├── tests/                 # Unit and integration tests
├── Dockerfile             # Docker container definition
├── docker-compose.yml     # Local deployment configuration
├── requirements.txt       # Python dependencies
├── setup.sh               # Setup script
└── README.md              # Project documentation
```

## 🧑‍💻 Development

### Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and run tests:
   ```bash
   pytest tests/
   ```

3. Format and lint your code:
   ```bash
   make format
   make lint
   ```

4. Commit your changes:
   ```bash
   git commit -m "Add meaningful commit message"
   ```

5. Push to your fork and create a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all functions and classes
- Include unit tests for new functionality

## 🌐 Deployment

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/ingress.yaml
kubectl apply -f kubernetes/monitoring.yaml

# Check deployment status
kubectl get all -n mlops-premium
```

### Production Considerations

- Use proper secrets management
- Configure TLS certificates for HTTPS
- Implement authentication for the API
- Set appropriate resource limits
- Configure backup and recovery procedures

## 📊 Monitoring

### Key Metrics to Monitor

- **Model Performance**: RMSE, MAE, R² for each segment
- **Prediction Latency**: Response time for predictions
- **Data Drift**: Changes in feature distributions
- **System Metrics**: CPU, memory, network usage

### Accessing Monitoring Dashboards

```bash
# Port-forward Prometheus
kubectl port-forward -n mlops-premium svc/prometheus-service 9090:9090

# Port-forward Grafana
kubectl port-forward -n mlops-premium svc/grafana-service 3000:3000

# Access in browser
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (default: admin/admin)
```

### Model Retraining

Trigger model retraining when:

1. Data drift is detected
2. Model performance degrades
3. On a regular schedule (e.g., monthly)

```bash
python -m src.train --log-artifacts --register-model
```

## 🔧 Troubleshooting

### Common Issues

- **API Issues**: Check logs with `docker-compose logs premium-api` or `kubectl logs -n mlops-premium deployment/premium-model-api`
- **Training Issues**: Check logs in the `logs/` directory
- **Kubernetes Issues**: Use `kubectl describe pod -n mlops-premium <pod-name>` for detailed information

For more detailed troubleshooting, refer to the [CHECKLIST.md](CHECKLIST.md) and [QUICKSTART.md](QUICKSTART.md) files.

## 📜 License

MIT License

## 🙏 Acknowledgments

- [List any acknowledgments, libraries, or resources here]