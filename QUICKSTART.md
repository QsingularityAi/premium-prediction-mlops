# MLOps Premium Prediction: Quick Start Guide

This document provides a quick start guide for developers to get up and running with the MLOps Premium Prediction project. It covers setup, development workflow, testing, deployment, monitoring, and troubleshooting.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Development Workflow](#development-workflow)
4. [Testing](#testing)
5. [Deployment](#deployment)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

Before beginning, ensure you have the following installed:

- Python 3.9+
- Docker & Docker Compose
- Kubernetes (local cluster - Docker Desktop with Kubernetes enabled or Minikube)
- Git
- Make (optional, but recommended)

## Initial Setup

### Clone the Repository

```bash
git clone <repository-url>
cd End_to_End_MLOPs_regression
```

### Set Up Development Environment

Using Make (recommended):

```bash
make setup
source venv/bin/activate
```

Manual setup:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
mkdir -p data/{raw,processed} models logs artifacts
```

### Prepare Training Data

Place your training data CSV file in the `data/raw/` directory. If you don't have data yet, you can generate test data:

```bash
make test-data
```

or:

```bash
python -c "from src.data.data_processor import generate_test_data; generate_test_data()"
```

### Configure the Application

Review and update the configuration in `config/config.yaml` as needed.

## Development Workflow

### Code Organization

```
src/
├── api/                # API service
│   └── app.py          # FastAPI application
├── data/               # Data processing
│   └── data_processor.py
├── models/             # Model training and inference
│   ├── model_trainer.py
│   └── model_predictor.py
└── train.py            # Training script
```

### Code Style

We follow PEP 8 guidelines with some modifications. Format your code with:

```bash
make format
```

Check code style with:

```bash
make lint
```

### Development Process

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes to the code**

3. **Ensure code is formatted and passes linting**:
   ```bash
   make format
   make lint
   ```

4. **Write tests for your code**:
   Add tests to the `tests/` directory

5. **Run the tests**:
   ```bash
   make test
   ```

6. **Train the model** (if you've made changes to training logic):
   ```bash
   make train
   ```

7. **Run the API locally** to test changes:
   ```bash
   make serve
   ```

8. **Commit and push your changes**:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin feature/your-feature-name
   ```

9. **Create a pull request** to merge your changes

## Testing

### Running Tests

Run all tests:

```bash
make test
```

Run specific test files:

```bash
pytest tests/test_data_processor.py
```

Run tests with specific markers:

```bash
pytest -m "unit"  # Run unit tests only
pytest -m "model"  # Run model-related tests
pytest -m "e2e"    # Run end-to-end tests
```

### Test Coverage

Generate a test coverage report:

```bash
make coverage
```

### Testing Different Python Versions

Use tox to test with multiple Python versions:

```bash
tox
```

## Deployment

### Local Deployment with Docker

Build and run the Docker container locally:

```bash
make docker-build
make docker-run
```

### Kubernetes Deployment

Deploy to a Kubernetes cluster:

```bash
make deploy
```

This will apply the following Kubernetes manifests:
- `kubernetes/namespace.yaml`
- `kubernetes/configmap.yaml`
- `kubernetes/deployment.yaml`
- `kubernetes/service.yaml`
- `kubernetes/ingress.yaml`

Check deployment status:

```bash
kubectl get all -n mlops-premium
```

### Updating a Deployment

To update an existing deployment with a new model version:

```bash
kubectl set env deployment/premium-model-api -n mlops-premium MODEL_VERSION=<new-version>
```

## Monitoring

### Setting Up Monitoring

Deploy monitoring tools (Prometheus & Grafana):

```bash
make monitor
```

### Accessing Monitoring Dashboards

Port-forward the services to access them locally:

```bash
# Access Prometheus
kubectl port-forward -n mlops-premium svc/prometheus-service 9090:9090

# Access Grafana
kubectl port-forward -n mlops-premium svc/grafana-service 3000:3000
```

Then open:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default login: admin/admin)

### Key Metrics to Monitor

- **Model Performance**: RMSE, MAE, R² by segment
- **Prediction Latency**: Response time histogram
- **Prediction Volume**: Number of predictions by segment
- **Error Rate**: Number of prediction errors
- **Data Drift**: Distribution changes in key features

### Drift Detection

Run a data drift detection check:

```bash
make drift-check
```

## Troubleshooting

### Common Issues

#### Setup Issues

**Issue**: `make setup` fails with permission errors.  
**Solution**: Run `sudo make setup` or manually create directories and set permissions.

**Issue**: Package installation fails.  
**Solution**: Ensure you have the latest pip with `pip install --upgrade pip` and check if your Python version is compatible.

#### Training Issues

**Issue**: Training fails with memory errors.  
**Solution**: Reduce batch size or number of cross-validation folds in `config/config.yaml`.

**Issue**: Segmented model training skips segments.  
**Solution**: This is expected for segments with insufficient data. Check the logs for details.

#### Deployment Issues

**Issue**: Docker container fails to start.  
**Solution**: Check logs with `docker logs <container-id>` and ensure all volumes are properly mounted.

**Issue**: Kubernetes deployment fails.  
**Solution**: Check events with `kubectl get events -n mlops-premium` and verify ConfigMap and Secrets are correctly configured.

**Issue**: API returns 500 errors.  
**Solution**: Check the logs with `kubectl logs -n mlops-premium deployment/premium-model-api` to identify the issue.

#### Monitoring Issues

**Issue**: Cannot access Grafana dashboards.  
**Solution**: Verify port forwarding is working and check if Grafana pod is running with `kubectl get pods -n mlops-premium`.

**Issue**: Metrics not showing up in Prometheus.  
**Solution**: Check that the API service has the correct annotations for Prometheus scraping and that the metrics endpoint is accessible.

### Getting Help

If you encounter issues not covered here:

1. Check the logs for error messages
2. Review the documentation in the README.md
3. Search existing issues on the repository
4. Reach out to the team through the project's communication channels

## Next Steps

- Review the full [README.md](README.md) for detailed documentation
- Explore the codebase to understand the implementation details
- Try running the end-to-end pipeline with test data
- Read about MLOps best practices in the project's wiki

