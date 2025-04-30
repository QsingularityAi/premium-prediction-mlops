# MLOps Premium Prediction Setup Verification Checklist

Use this checklist to verify that your environment and all components of the MLOps Premium Prediction system are correctly set up and functioning properly.

## Environment Verification

- [ ] **Python Environment**
  - [ ] Verify Python version is 3.9 or higher
    ```bash
    python --version  # Should be 3.9.x or higher
    ```
  - [ ] Verify virtual environment is activated
    ```bash
    # Your prompt should show (venv) at the beginning
    # or run:
    which python  # Should point to the venv directory
    ```
  - [ ] Verify all dependencies are installed
    ```bash
    pip list | grep -E 'pandas|numpy|scikit-learn|lightgbm|xgboost|mlflow|fastapi'
    # All these packages should be listed
    ```

- [ ] **Docker Setup**
  - [ ] Verify Docker is installed and running
    ```bash
    docker --version
    docker ps  # Should not show an error
    ```
  - [ ] Verify Docker Compose is installed
    ```bash
    docker-compose --version
    ```

- [ ] **Kubernetes Setup**
  - [ ] Verify Kubernetes is enabled and running
    ```bash
    kubectl version
    kubectl get nodes  # Should show at least one node
    ```
  - [ ] Verify kubectl can access the cluster
    ```bash
    kubectl cluster-info
    ```

- [ ] **Directory Structure**
  - [ ] Verify all required directories exist
    ```bash
    ls -la data/raw data/processed models logs artifacts
    # All these directories should exist
    ```

## Data Pipeline Verification

- [ ] **Test Data Existence**
  - [ ] Verify test data exists or can be generated
    ```bash
    # Either check if the file exists:
    ls -la data/raw/*.csv
    
    # Or generate test data:
    make test-data
    # Then verify:
    ls -la data/raw/*.csv
    ```

- [ ] **Data Processing**
  - [ ] Verify data processing works
    ```bash
    # Run a simple Python script to test:
    python -c "
    from src.data.data_processor import DataProcessor
    import yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    processor = DataProcessor(config)
    df = processor.load_data()
    processed_df = processor.preprocess_data(df)
    print(f'Data loaded and processed: {processed_df.shape}')
    "
    # Should output data shape without errors
    ```

- [ ] **Data Version Control**
  - [ ] Verify DVC is initialized
    ```bash
    dvc status
    # Should not show an error
    ```
  - [ ] Try adding a data file to DVC
    ```bash
    # If you have test data:
    dvc add data/raw/*.csv
    git status  # Should show new .dvc files, not csv files
    # Cleanup after test:
    dvc remove data/raw/*.csv.dvc
    ```

## Model Training Verification

- [ ] **Basic Training Run**
  - [ ] Verify model training works with a minimal configuration
    ```bash
    # Adjust config for a quick test:
    # Edit config/config.yaml to set n_iter_search: 2, cv_folds_tuning: 2
    
    # Run training:
    python -m src.train --config config/config.yaml
    
    # Verify models were created:
    ls -la models/
    # Should see a timestamped directory
    ```

- [ ] **Model Artifacts**
  - [ ] Verify model artifact structure
    ```bash
    # Using the directory created in the previous step:
    MODEL_VERSION=$(ls -t models/ | head -1)
    ls -la models/$MODEL_VERSION
    # Should see files like preprocessor.joblib, premium_thresholds.json,
    # and segment model files like very_low_model.joblib, etc.
    ```

- [ ] **MLflow Tracking**
  - [ ] Verify MLflow tracking is working
    ```bash
    # Start MLflow UI:
    mlflow ui
    # Open http://localhost:5000 in a browser
    # Should see an experiment with runs
    ```

## API Deployment Verification

- [ ] **Local API Server**
  - [ ] Verify API server starts correctly
    ```bash
    # Start the API server:
    python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &
    
    # Check if it's running:
    curl http://localhost:8000/health
    # Should return a JSON response with "status": "healthy"
    
    # Stop the server:
    kill %1  # or find and kill the process
    ```

- [ ] **API Documentation**
  - [ ] Verify Swagger documentation is accessible
    ```bash
    # Start the API server again if needed
    python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &
    
    # Open in browser:
    echo "Open http://localhost:8000/docs in your browser"
    # Should see a Swagger UI with API endpoints
    
    # Stop the server when done
    kill %1
    ```

- [ ] **Docker Containerization**
  - [ ] Verify Docker image builds correctly
    ```bash
    docker build -t premium-model-api:test .
    # Should complete without errors
    
    # Check the image was created:
    docker images | grep premium-model-api
    ```
  - [ ] Verify container runs correctly
    ```bash
    # Run with minimal configuration:
    docker run -d -p 8000:8000 --name test-api premium-model-api:test
    
    # Check it's running:
    docker ps | grep test-api
    
    # Test health endpoint:
    curl http://localhost:8000/health
    
    # Clean up:
    docker stop test-api
    docker rm test-api
    ```

## Kubernetes Deployment Verification

- [ ] **Namespace Creation**
  - [ ] Verify namespace can be created
    ```bash
    kubectl apply -f kubernetes/namespace.yaml
    kubectl get ns | grep mlops-premium
    # Should show the namespace
    ```

- [ ] **ConfigMap and Secrets**
  - [ ] Verify ConfigMap can be applied
    ```bash
    kubectl apply -f kubernetes/configmap.yaml
    kubectl get configmap -n mlops-premium
    # Should show the configmap
    ```

- [ ] **Deployment Creation**
  - [ ] Verify deployment can be created
    ```bash
    # You can test this if you have a Docker image ready:
    kubectl apply -f kubernetes/deployment.yaml
    kubectl get pods -n mlops-premium
    # Should show pods being created
    
    # Clean up after verification:
    kubectl delete -f kubernetes/deployment.yaml
    ```

## Monitoring Setup Verification

- [ ] **Prometheus Setup**
  - [ ] Verify Prometheus configuration exists
    ```bash
    cat prometheus/prometheus.yml
    # Should show a valid Prometheus configuration
    ```
  - [ ] Verify Prometheus can be deployed (if using Kubernetes)
    ```bash
    kubectl apply -f kubernetes/monitoring.yaml
    kubectl get pods -n mlops-premium | grep prometheus
    # Should show Prometheus pods
    
    # Clean up after verification:
    kubectl delete -f kubernetes/monitoring.yaml
    ```

- [ ] **Grafana Setup**
  - [ ] Verify Grafana dashboards exist
    ```bash
    ls -la grafana/provisioning/dashboards/
    # Should show dashboard configuration files
    ```

- [ ] **Metrics Collection**
  - [ ] Verify metrics are being collected (with API running)
    ```bash
    # Start the API server:
    python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &
    
    # Send a test request:
    curl -X POST -H "Content-Type: application/json" -d '{"features": {"Age": 35, "Vehicle_Age": 5, "Credit_Score": 720, "Annual_Income": 65000}}' http://localhost:8000/predict
    
    # Get metrics:
    curl http://localhost:8000/metrics
    # Should show various metrics including prediction_counter
    
    # Stop the server:
    kill %1
    ```

## Security Checks

- [ ] **Code Security**
  - [ ] Verify no sensitive information in code
    ```bash
    grep -r "password\|secret\|key\|token" --include="*.py" .
    # Should not show actual credentials in code
    ```

- [ ] **Dependency Security**
  - [ ] Check for vulnerabilities in dependencies
    ```bash
    pip install safety
    safety check
    # Review any security issues found
    ```

- [ ] **Docker Security**
  - [ ] Verify Docker image runs as non-root
    ```bash
    # Build and run:
    docker build -t premium-model-api:test .
    docker run --rm premium-model-api:test id
    # Should show a non-root user ID (not 0)
    ```

- [ ] **Kubernetes Security**
  - [ ] Verify resource limits are set
    ```bash
    grep -A10 "resources:" kubernetes/deployment.yaml
    # Should show CPU and memory limits
    ```

- [ ] **API Security**
  - [ ] Verify API has authentication (if applicable)
    ```bash
    # Check if authentication is implemented:
    grep -r "authenticate\|auth\|security" src/api/
    # Review security middleware in the API code
    ```

## Final Verification

- [ ] **End-to-End Test**
  - [ ] Run an end-to-end test to verify all components work together
    ```bash
    pytest -xvs tests/test_e2e.py
    # Should complete without errors
    ```

## Clean Up

- [ ] **Remove Test Resources**
  - [ ] Clean up Kubernetes resources
    ```bash
    kubectl delete ns mlops-premium
    ```
  - [ ] Clean up Docker containers and images
    ```bash
    docker stop $(docker ps -a -q --filter "ancestor=premium-model-api:test") 2>/dev/null || true
    docker rm $(docker ps -a -q --filter "ancestor=premium-model-api:test") 2>/dev/null || true
    docker rmi premium-model-api:test 2>/dev/null || true
    ```

