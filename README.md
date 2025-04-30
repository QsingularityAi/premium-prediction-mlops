# MLOps Premium Prediction Project

A complete end-to-end MLOps solution for insurance premium prediction using segmented regression models with Kubernetes orchestration.

## 1. Project Description

This project implements a comprehensive MLOps (Machine Learning Operations) workflow for predicting insurance premiums. The system uses a segmented approach, training separate models for different premium ranges to improve prediction accuracy. The entire pipeline is containerized and can be deployed using Kubernetes, with monitoring, experiment tracking, and model versioning built-in.

### Key Features

- **Segmented Modeling**: Splits premium data into segments and trains specialized models for each segment
- **Automated Data Pipeline**: Data versioning, validation, and preprocessing
- **Experiment Tracking**: MLflow integration for tracking model parameters, metrics, and artifacts
- **Containerized Deployment**: Docker containers for reproducible deployment
- **Kubernetes Orchestration**: Scalable deployment with Kubernetes
- **Monitoring & Alerting**: Prometheus and Grafana for real-time monitoring
- **CI/CD Ready**: Infrastructure for continuous integration and deployment

### Architecture Overview

The MLOps pipeline consists of several interconnected components:

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

## 2. Installation and Setup

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Kubernetes (Docker Desktop with Kubernetes enabled or a separate cluster)
- Git

### Quick Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/premium-prediction-mlops.git
   cd premium-prediction-mlops
   ```

2. Run the setup script to create the environment and install dependencies:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

4. Place your training data in the `data/raw/` directory.

### Manual Setup

If you prefer to set up the project manually:

1. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create the required directories:
   ```bash
   mkdir -p data/{raw,processed} models logs artifacts
   ```

4. Initialize DVC:
   ```bash
   dvc init
   ```

5. Set up monitoring infrastructure:
   ```bash
   mkdir -p prometheus grafana/provisioning/{dashboards,datasources}
   ```

## 3. Usage Instructions

### Data Preparation

1. Place your training data CSV file in the `data/raw/` directory.
2. Track it with DVC:
   ```bash
   dvc add data/raw/train_sampled2.csv
   git add data/raw/train_sampled2.csv.dvc
   git commit -m "Add training data"
   ```

### Training Models

1. Edit configuration in `config/config.yaml` if needed.

2. Run the training script:
   ```bash
   python -m src.train
   ```

3. Track the results with MLflow:
   ```bash
   mlflow ui
   ```
   Then open http://localhost:5000 in your browser.

### Local Deployment

1. Build and start the services with Docker Compose:
   ```bash
   docker-compose up -d
   ```

2. Check the API documentation:
   ```
   http://localhost:8000/docs
   ```

3. Monitor the application:
   - MLflow: http://localhost:5000
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000

### Kubernetes Deployment

1. Apply Kubernetes manifests:
   ```bash
   kubectl apply -f kubernetes/namespace.yaml
   kubectl apply -f kubernetes/configmap.yaml
   kubectl apply -f kubernetes/deployment.yaml
   kubectl apply -f kubernetes/service.yaml
   kubectl apply -f kubernetes/ingress.yaml
   kubectl apply -f kubernetes/monitoring.yaml
   ```

2. Check deployment status:
   ```bash
   kubectl get all -n mlops-premium
   ```

### Making Predictions

Using the API:

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

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
        "instances": [
            {
                "Age": 35,
                "Vehicle_Age": 5,
                "Credit_Score": 720,
                "Annual_Income": 65000
            },
            {
                "Age": 25,
                "Vehicle_Age": 2,
                "Credit_Score": 650,
                "Annual_Income": 45000
            }
        ]
     }'
```

## 4. MLOps Pipeline Components

### Data Processing (`src/data/data_processor.py`)

- **Data Loading**: Reads raw data from CSV files
- **Preprocessing**: Handles missing values, date conversions, outlier detection
- **Feature Engineering**: Creates derived features and interactions
- **Data Segmentation**: Divides the data into premium segments for specialized modeling

### Model Training (`src/models/model_trainer.py`)

- **Segmented Training**: Trains different models for each premium segment
- **Hyperparameter Tuning**: Optimizes model hyperparameters using RandomizedSearchCV
- **Feature Selection**: Selects important features for each segment
- **Evaluation**: Computes metrics (RMSE, MAE, R²) for model performance assessment
- **Model Serialization**: Saves trained models with metadata

### Model Prediction (`src/models/model_predictor.py`)

- **Model Loading**: Loads trained segment models
- **Preprocessing**: Applies the same preprocessing pipeline to new data
- **Segment Assignment**: Determines which segment a new data point belongs to
- **Prediction**: Makes predictions using the appropriate segment model

### API Service (`src/api/app.py`)

- **REST Endpoints**: Provides endpoints for single and batch predictions
- **Validation**: Validates input data format
- **Metrics**: Collects metrics for monitoring
- **Documentation**: Auto-generated API documentation

### Monitoring Configuration

- **Prometheus**: Collects metrics from the API service
- **Grafana**: Visualizes metrics with customizable dashboards
- **Alerts**: Configurable alerting based on performance metrics

## 5. Monitoring and Maintenance

### Key Metrics to Monitor

- **Model Performance**: RMSE, MAE, R² for each segment
- **Prediction Latency**: Response time for predictions
- **Data Drift**: Changes in feature distributions
- **System Metrics**: CPU, memory, network usage

### Model Retraining

Trigger model retraining when:

1. **Data Drift Detected**: When input feature distributions change significantly
2. **Performance Degradation**: When model metrics fall below thresholds
3. **Scheduled Updates**: On a regular cadence (e.g., monthly)

To retrain:

```bash
python -m src.train --log-artifacts --register-model
```

### Viewing Metrics

1. Access Grafana dashboards at http://localhost:3000 (or your Kubernetes ingress)
2. Default username/password: admin/admin
3. Check the "Premium Model Dashboard" for key metrics

### Troubleshooting

- **API Issues**: Check logs with `docker-compose logs premium-api` or `kubectl logs -n mlops-premium deployment/premium-model-api`
- **Training Issues**: Check logs in the `logs/` directory
- **Kubernetes Issues**: Use `kubectl describe pod -n mlops-premium <pod-name>` for detailed information

## 6. Contribution Guidelines

### Development Workflow

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit:
   ```bash
   git commit -m "Add your meaningful commit message"
   ```
4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Create a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type annotations
- Write docstrings for all functions and classes
- Add unit tests for new functionality

### Testing

Run tests with:
```bash
pytest tests/
```

### Adding New Features

When adding new functionality:
1. Update the appropriate module in `src/`
2. Add tests in `tests/`
3. Update documentation in docstrings and README
4. Ensure CI/CD pipeline passes

## License

[Insert license information here]

## Acknowledgments

- [List any acknowledgments, libraries, or resources here]

# MLOps Premium Prediction Project

This project demonstrates a complete MLOps (Machine Learning Operations) workflow for a premium prediction regression task using segmented models. The system is built with scalability, reproducibility, and production-readiness in mind.

## Project Overview

This MLOps pipeline implements a premium prediction system that segments data into 5 different categories and trains specialized models for each segment. The approach includes:

- **Data Pipeline**: Versioning, validation, and preprocessing of insurance data
- **Segmented Models**: Specialized models (LGBM and XGBoost) for different premium segments
- **Model Registry**: Versioning and tracking of model artifacts
- **Model Serving**: REST API for real-time predictions
- **Monitoring**: Comprehensive monitoring of data drift and model performance

### Architecture

```
                                            ┌───────────────┐
                                            │  Data Sources │
                                            └───────┬───────┘
                                                   │
┌───────────────────────────────────────────────┐ │ ┌───────────────────────┐
│                  Data Pipeline                 │ │ │                       │
│ ┌─────────────┐  ┌────────────┐  ┌──────────┐ │◄┘ │    MLflow Tracking    │
│ │ Data Ingest │─►│Validation & │─►│ Feature  │ │   │ ┌──────────────────┐ │
│ │  (DVC)      │  │Preprocessing│  │Engineering│ │   │ │  Experiments     │ │
│ └─────────────┘  └────────────┘  └──────────┘ │   │ └──────────────────┘ │
└─────────────────────┬─────────────────────────┘   │ ┌──────────────────┐ │
                      │                             │ │  Parameters       │ │
                      ▼                             │ └──────────────────┘ │
┌────────────────────────────────────────────┐     │ ┌──────────────────┐ │
│           Model Training Pipeline           │     │ │  Metrics         │ │
│ ┌────────┐  ┌────────┐  ┌────────────────┐ │     │ └──────────────────┘ │
│ │Segment │  │ Model  │  │Feature Selection│ │────► ┌──────────────────┐ │
│ │Creation│─►│Training│─►│& Evaluation     │ │     │ │  Artifacts       │ │
│ └────────┘  └────────┘  └────────────────┘ │     │ └──────────────────┘ │
└────────────────────────┬───────────────────┘     └───────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────┐    ┌───────────────────────┐
│             Model Registry                   │    │                       │
│ ┌─────────────┐  ┌──────────┐  ┌──────────┐ │    │      Monitoring       │
│ │ Model       │  │Versioning│  │Deployment│ │    │ ┌──────────────────┐ │
│ │ Artifacts   │─►│& Metadata│─►│Approval  │ │    │ │  Data Drift      │ │
│ └─────────────┘  └──────────┘  └──────────┘ │    │ └──────────────────┘ │
└────────────────────────┬────────────────────┘    │ ┌──────────────────┐ │
                         │                         │ │  Model            │ │
                         ▼                         │ │  Performance      │ │
┌────────────────────────────────────────────┐     │ └──────────────────┘ │
│             Serving Infrastructure          │     │ ┌──────────────────┐ │
│ ┌───────────┐  ┌──────────┐  ┌───────────┐ │     │ │  System          │ │
│ │ FastAPI   │  │Kubernetes│  │Prometheus │ │────► │  Metrics          │ │
│ │ Endpoints │─►│Deployment│─►│& Grafana  │ │     │ └──────────────────┘ │
│ └───────────┘  └──────────┘  └───────────┘ │     └───────────────────────┘
└────────────────────────────────────────────┘
```

## Setup Instructions

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Kubernetes (enabled in Docker Desktop or a separate cluster)
- Git LFS (for data versioning)

### Initial Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd End_to_End_MLOPs_regression
   ```

2. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

### Manual Setup (alternative to setup.sh)

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Initialize DVC:
   ```bash
   dvc init
   ```

4. Create necessary directories:
   ```bash
   mkdir -p data/raw data/processed models logs artifacts
   ```

## Directory Structure

```
End_to_End_MLOPs_regression/
├── config/                  # Configuration files
│   └── config.yaml          # Main configuration
├── data/                    # Data directory (managed by DVC)
│   ├── raw/                 # Raw data
│   └── processed/           # Processed data
├── kubernetes/              # Kubernetes manifests
│   ├── configmap.yaml       # ConfigMap for non-sensitive configuration
│   ├── deployment.yaml      # Deployment specification
│   ├── ingress.yaml         # Ingress routing
│   ├── namespace.yaml       # Namespace definition
│   ├── monitoring.yaml      # Prometheus and Grafana setup
│   ├── secrets.yaml         # Secret management template
│   └── service.yaml         # Service definition
├── models/                  # Model storage
├── notebooks/               # Jupyter notebooks
├── src/                     # Source code
│   ├── api/                 # API service
│   │   └── app.py           # FastAPI application
│   ├── data/                # Data processing code
│   │   └── data_processor.py # Data processing module
│   └── models/              # Model-related code
│       ├── model_trainer.py # Model training module
│       └── model_predictor.py # Model prediction module
├── tests/                   # Unit and integration tests
├── .dockerignore            # Docker ignore file
├── .gitignore               # Git ignore file
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile               # Dockerfile for building containers
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
└── setup.sh                 # Setup script
```

## Usage Examples

### Training Models

To train the segmented models:

```python
from src.data.data_processor import DataProcessor
from src.models.model_trainer import ModelTrainer
import yaml

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize data processor
data_processor = DataProcessor(config)

# Load and preprocess data
df = data_processor.load_data()
df = data_processor.preprocess_data(df)
df = data_processor.perform_feature_engineering(df)

# Split data and create segments
X_train, X_test, y_train, y_test = data_processor.split_data(df)
numerical_features = [f for f in X_train.columns if X_train[f].dtype.kind in 'fc']
categorical_features = [f for f in X_train.columns if X_train[f].dtype.kind not in 'fc']
preprocessor = data_processor.create_preprocessing_pipeline(numerical_features, categorical_features)
premium_thresholds = data_processor.create_segments(y_train)
segment_data_train = data_processor.split_into_segments(X_train, y_train, premium_thresholds)
test_masks = data_processor.get_segment_masks_test(y_test, premium_thresholds)

# Train models
model_trainer = ModelTrainer(config)
segment_model_configs = model_trainer.define_model_configs(segment_data_train)
trained_models = model_trainer.train_segment_models(segment_data_train, preprocessor, segment_model_configs)

# Evaluate models
y_pred, combined_metrics, segment_metrics = model_trainer.evaluate_models(
    trained_models, X_test, y_test, test_masks
)

# Save models
model_trainer.save_models(trained_models, preprocessor, premium_thresholds)
```

### Making Predictions

Using the API:

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

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
        "instances": [
            {
                "Age": 35,
                "Vehicle_Age": 5,
                "Credit_Score": 720,
                "Annual_Income": 65000
            },
            {
                "Age": 25,
                "Vehicle_Age": 2,
                "Credit_Score": 650,
                "Annual_Income": 45000
            }
        ]
     }'
```

Using the model directly:

```python
from src.models.model_predictor import ModelPredictor
import pandas as pd
import yaml

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize model predictor
predictor = ModelPredictor(config)
predictor.load_models()

# Make predictions
data = pd.DataFrame([{
    'Age': 35,
    'Vehicle_Age': 5,
    'Credit_Score': 720,
    'Annual_Income': 65000,
    'Previous_Claims': 1,
    'Insurance_Duration': 3
}])

result = predictor.predict(data)
print(f"Prediction: {result['predictions'][0]}, Segment: {result['segments'][0]}")
```

### Running Locally with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f premium-api

# Access services
# - API: http://localhost:8000/docs
# - MLflow: http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

## Development Guidelines

### Code Structure

- Follow the modular structure of the project
- Keep data processing, model training, and API code separate
- Use config.yaml for all configurable parameters

### Testing

Run tests with pytest:

```bash
pytest tests/
```

Write tests for:
- Data processing logic
- Model training and evaluation
- API endpoints

### Adding Features

When adding new features:
1. Update the appropriate module in src/
2. Update config.yaml with any new parameters
3. Write tests for the new functionality
4. Update documentation if necessary

### Version Control

- Use git for code versioning
- Use DVC for data and model versioning
- Create feature branches for new functionality
- Submit pull requests for code review

## Deployment Instructions

### Kubernetes Deployment

1. Update the Kubernetes manifests in the kubernetes/ directory if needed.

2. Apply the Kubernetes manifests:
   ```bash
   # Create namespace
   kubectl apply -f kubernetes/namespace.yaml
   
   # Create ConfigMap and Secrets
   kubectl apply -f kubernetes/configmap.yaml
   # Create secrets manually or using a secure method
   
   # Deploy the application
   kubectl apply -f kubernetes/deployment.yaml
   kubectl apply -f kubernetes/service.yaml
   kubectl apply -f kubernetes/ingress.yaml
   
   # Deploy monitoring
   kubectl apply -f kubernetes/monitoring.yaml
   ```

3. Verify the deployment:
   ```bash
   kubectl get all -n mlops-premium
   ```

### Production Considerations

- Use proper secrets management (e.g., Kubernetes Secrets, HashiCorp Vault)
- Set up proper TLS certificates for HTTPS
- Implement authentication and authorization
- Configure resource limits appropriately
- Set up CI/CD pipelines for automated deployment
- Implement backup and recovery procedures

## Monitoring and Maintenance

### Monitoring Model Performance

1. Access Grafana (http://localhost:3000 in local development)
2. Use the pre-configured dashboards to monitor:
   - Prediction latency
   - Error rates
   - Data drift
   - System metrics

### Retraining Models

1. When data drift is detected or on a scheduled basis:
   ```bash
   # Pull latest data
   dvc pull
   
   # Run training
   python src/train.py
   
   # Register new model version
   # (This happens automatically in the training script)
   ```

2. Approve the new model version for deployment using MLflow or your model registry.

3. Update the deployed model version:
   ```bash
   kubectl set env deployment/premium-model-api MODEL_VERSION=<new-version> -n mlops-premium
   ```

## Troubleshooting

- Check logs: `kubectl logs -n mlops-premium deployment/premium-model-api`
- Verify configs: `kubectl describe configmap -n mlops-premium premium-model-config`
- Check health endpoint: `curl http://<api-url>/health`
- Check metrics endpoint: `curl http://<api-url>/metrics`

## License

[License information]

## Contributors

[Contributor information]

