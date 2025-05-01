.PHONY: setup clean lint test train serve deploy monitor drift-check test-data format help

# Python configuration
PYTHON = python3
VENV = venv
PIP = $(VENV)/bin/pip
PYTHON_VENV = $(VENV)/bin/python
PYTEST = $(VENV)/bin/pytest
BLACK = $(VENV)/bin/black
FLAKE8 = $(VENV)/bin/flake8
ISORT = $(VENV)/bin/isort

# Directories
SRC_DIR = src
TEST_DIR = tests
DATA_DIR = data
MODEL_DIR = models
CONFIG_FILE = config/config.yaml

# Docker and Kubernetes
DOCKER_IMAGE = premium-model-api
KUBE_NAMESPACE = mlops-premium

# Default help command
help:
	@echo "MLOps Premium Prediction - Development Commands"
	@echo "----------------------------------------------"
	@echo "setup             - Set up the development environment"
	@echo "clean             - Clean up generated files and directories"
	@echo "lint              - Run code linters (flake8, black, isort)"
	@echo "format            - Format code with black and isort"
	@echo "test              - Run tests"
	@echo "test-data         - Generate test data"
	@echo "train             - Train the model"
	@echo "serve             - Run the API server locally"
	@echo "docker-build      - Build Docker image"
	@echo "docker-run        - Run Docker container locally"
	@echo "deploy            - Deploy to Kubernetes"
	@echo "monitor           - Start monitoring tools"
	@echo "drift-check       - Run data drift detection"
	@echo "help              - Show this help message"

# Setup development environment
setup:
	@echo "Setting up development environment..."
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/processed $(MODEL_DIR) logs artifacts
	@echo "Setup complete. Activate the environment with: source venv/bin/activate"

# Clean up generated files
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +
	@echo "Clean complete."

# Run linters
lint:
	$(FLAKE8) $(SRC_DIR) $(TEST_DIR)
	$(BLACK) --check $(SRC_DIR) $(TEST_DIR)
	$(ISORT) --check --profile black $(SRC_DIR) $(TEST_DIR)

# Format code
format:
	$(BLACK) $(SRC_DIR) $(TEST_DIR)
	$(ISORT) --profile black $(SRC_DIR) $(TEST_DIR)

# Run tests
test:
	$(PYTEST) -xvs $(TEST_DIR)

# Test coverage
coverage:
	$(PYTEST) --cov=$(SRC_DIR) --cov-report=html $(TEST_DIR)
	@echo "Coverage report generated in htmlcov directory"

# Generate test data
test-data:
	@echo "Generating test data..."
	$(PYTHON_VENV) -c "from src.data.data_processor import generate_test_data; generate_test_data()"
	@echo "Test data generated"

# Train model
train:
	@echo "Training models..."
	$(PYTHON_VENV) -m src.train --config $(CONFIG_FILE)
	@echo "Training complete"

# Run API server locally
serve:
	@echo "Starting API server..."
	$(PYTHON_VENV) -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
	@echo "API server running at http://localhost:8000"

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	docker build -t $(DOCKER_IMAGE):latest .
	@echo "Docker image built"

# Run Docker container locally
docker-run:
	@echo "Running Docker container..."
	docker run -p 8000:8000 -v "$(PWD)/models":/app/models -v "$(PWD)/data":/app/data -v "$(PWD)/logs":/app/logs $(DOCKER_IMAGE):latest
	@echo "Docker container running"

# Deploy to Kubernetes
deploy:
	@echo "Deploying to Kubernetes..."
	kubectl apply -f kubernetes/namespace.yaml
	kubectl apply -f kubernetes/configmap.yaml
	kubectl apply -f kubernetes/deployment.yaml
	kubectl apply -f kubernetes/service.yaml
	kubectl apply -f kubernetes/ingress.yaml
	@echo "Deployment complete. Check status with: kubectl get all -n $(KUBE_NAMESPACE)"

# Start monitoring
monitor:
	@echo "Starting monitoring tools..."
	kubectl apply -f kubernetes/monitoring.yaml
	@echo "Monitoring deployed. Access at:"
	@echo "- Prometheus: http://localhost:9090 (after port-forwarding)"
	@echo "- Grafana: http://localhost:3000 (after port-forwarding)"
	@echo "To port-forward: kubectl port-forward -n $(KUBE_NAMESPACE) svc/prometheus-service 9090:9090"
	@echo "To port-forward: kubectl port-forward -n $(KUBE_NAMESPACE) svc/grafana-service 3000:3000"

# Check for data drift
drift-check:
	@echo "Checking for data drift..."
	$(PYTHON_VENV) -m src.drift_detection --config $(CONFIG_FILE)
	@echo "Drift check complete"
