#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Setup variables
LOG_FILE="setup_log.txt"
PYTHON_VERSION="3.9"
REQUIRED_PACKAGES=("docker" "python3" "git")
PYTHON_VENV="venv"

# Logging function
log() {
    echo -e "$1"
    echo "$(date): $1" | sed 's/\x1b\[[0-9;]*m//g' >> "$LOG_FILE"
}

# Error handling function
handle_error() {
    log "${RED}Error: $1${NC}"
    log "${YELLOW}Check $LOG_FILE for more details${NC}"
    exit 1
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Print header
echo -e "${YELLOW}==============================================${NC}"
echo -e "${YELLOW}      MLOps Premium Prediction Setup         ${NC}"
echo -e "${YELLOW}==============================================${NC}"

# Initialize log file
echo "MLOps Premium Prediction Setup Log - $(date)" > "$LOG_FILE"
log "Starting setup process"

# Step 1: Check prerequisites
log "\n${YELLOW}Step 1: Checking prerequisites...${NC}"

# Check Python version
if command_exists python3; then
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    log "Found Python version: ${GREEN}$python_version${NC}"
    
    # Check if Python version meets requirements
    if [[ $(echo "$python_version" | cut -d. -f1) -lt 3 || ($(echo "$python_version" | cut -d. -f1) -eq 3 && $(echo "$python_version" | cut -d. -f2) -lt 9) ]]; then
        handle_error "Python 3.9 or higher is required. Found $python_version"
    fi
else
    handle_error "Python 3 is not installed or not in PATH"
fi

# Check Docker
if command_exists docker; then
    docker_version=$(docker --version | awk '{print $3}' | tr -d ',')
    log "Found Docker version: ${GREEN}$docker_version${NC}"
else
    log "${YELLOW}Warning: Docker is not installed or not in PATH${NC}"
    log "${YELLOW}Docker is required for containerization. Please install Docker before deploying containers.${NC}"
fi

# Check kubectl
if command_exists kubectl; then
    kubectl_version=$(kubectl version --client --short 2>/dev/null | awk '{print $3}' || echo "unknown")
    log "Found kubectl version: ${GREEN}$kubectl_version${NC}"
else
    log "${YELLOW}Warning: kubectl is not installed or not in PATH${NC}"
    log "${YELLOW}kubectl is required for Kubernetes deployments. Install if you plan to deploy to Kubernetes.${NC}"
fi

# Check Git
if command_exists git; then
    git_version=$(git --version | awk '{print $3}')
    log "Found Git version: ${GREEN}$git_version${NC}"
else
    handle_error "Git is not installed or not in PATH"
fi

log "${GREEN}Prerequisites check completed${NC}"

# Step 2: Create directory structure
log "\n${YELLOW}Step 2: Creating directory structure...${NC}"

# List of directories to create
directories=(
    "data/raw"
    "data/processed"
    "models"
    "logs"
    "artifacts"
    "tests"
    "prometheus"
    "grafana/provisioning/dashboards"
    "grafana/provisioning/datasources"
    "mlruns"
)

# Create each directory
for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        log "Created directory: ${GREEN}$dir${NC}"
    else
        log "Directory already exists: $dir"
    fi
done

log "${GREEN}Directory structure created${NC}"

# Step 3: Create Python virtual environment
log "\n${YELLOW}Step 3: Setting up Python virtual environment...${NC}"

# Check if venv already exists
if [ -d "$PYTHON_VENV" ]; then
    log "Virtual environment already exists at $PYTHON_VENV"
    read -p "Do you want to recreate it? (y/n): " recreate_venv
    
    if [[ $recreate_venv == "y" || $recreate_venv == "Y" ]]; then
        log "Removing existing virtual environment..."
        rm -rf "$PYTHON_VENV"
    else
        log "Using existing virtual environment"
    fi
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$PYTHON_VENV" ]; then
    log "Creating new Python virtual environment..."
    python3 -m venv "$PYTHON_VENV" || handle_error "Failed to create virtual environment"
    log "${GREEN}Virtual environment created successfully${NC}"
fi

# Activate virtual environment and install dependencies
log "Activating virtual environment and installing dependencies..."

# Determine correct activation script based on OS
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
    # macOS or Linux
    source "$PYTHON_VENV/bin/activate" || handle_error "Failed to activate virtual environment"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source "$PYTHON_VENV/Scripts/activate" || handle_error "Failed to activate virtual environment"
else
    handle_error "Unsupported operating system: $OSTYPE"
fi

# Upgrade pip and install dependencies
python -m pip install --upgrade pip || handle_error "Failed to upgrade pip"

if [ -f "requirements.txt" ]; then
    python -m pip install -r requirements.txt || handle_error "Failed to install dependencies"
    log "${GREEN}Dependencies installed successfully${NC}"
else
    # Create basic requirements.txt if it doesn't exist
    log "${YELLOW}requirements.txt not found. Creating basic requirements...${NC}"
    cat > requirements.txt << EOF
# Core ML Libraries
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0
category_encoders>=2.3.0
matplotlib>=3.4.0
seaborn>=0.11.0

# MLOps Tools
mlflow>=2.0.0
dvc>=2.0.0
fastapi>=0.68.0
uvicorn>=0.15.0
prometheus-client>=0.12.0
pyyaml>=6.0

# Utilities
joblib>=1.1.0
python-dotenv>=0.19.0
EOF
    python -m pip install -r requirements.txt || handle_error "Failed to install dependencies"
    log "${GREEN}Basic dependencies installed successfully${NC}"
fi

# Step 4: Initialize Git and DVC
log "\n${YELLOW}Step 4: Initializing version control...${NC}"

# Initialize Git repository if not already initialized
if [ ! -d ".git" ]; then
    log "Initializing Git repository..."
    git init || handle_error "Failed to initialize Git repository"
    log "${GREEN}Git repository initialized${NC}"
    
    # Create initial .gitignore
    log "Creating .gitignore file..."
    cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*.so
.Python
venv/
ENV/
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebooks
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# Data (managed by DVC)
/data/raw/*
/data/processed/*
!/data/raw/.gitkeep
!/data/processed/.gitkeep

# Models and artifacts
/models/*
/artifacts/*
!/models/.gitkeep
!/artifacts/.gitkeep

# Logs
/logs/*
*.log
!/logs/.gitkeep

# MLflow
/mlruns/

# IDE
.idea/
.vscode/
*.swp
.DS_Store
EOF
    log "${GREEN}.gitignore file created${NC}"
else
    log "Git repository already initialized"
fi

# Initialize DVC
if ! command_exists dvc; then
    log "${YELLOW}DVC not found in PATH. It should have been installed with requirements.txt${NC}"
    log "Attempting to install DVC..."
    python -m pip install dvc || handle_error "Failed to install DVC"
fi

if [ ! -d ".dvc" ]; then
    log "Initializing DVC..."
    dvc init || handle_error "Failed to initialize DVC"
    
    # Add data directory to DVC if files exist
    if [ -n "$(ls -A data/raw 2>/dev/null)" ]; then
        log "Adding data files to DVC..."
        dvc add data/raw/* || log "${YELLOW}Warning: No data files to add or error adding files${NC}"
    else
        log "${YELLOW}No data files found to add to DVC${NC}"
        # Create placeholder file
        touch data/raw/.gitkeep
        touch data/processed/.gitkeep
    fi
    
    log "${GREEN}DVC initialized successfully${NC}"
else
    log "DVC already initialized"
fi

# Step 5: Set up monitoring infrastructure
log "\n${YELLOW}Step 5: Setting up monitoring infrastructure...${NC}"

# Create Prometheus configuration
log "Creating Prometheus configuration..."
cat > prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # Premium Prediction API
  - job_name: "premium-api"
    static_configs:
      - targets: ["premium-api:8000"]
    metrics_path: /metrics

  # Prometheus self-monitoring
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]
EOF

# Create Grafana datasource
log "Creating Grafana datasource configuration..."
mkdir -p grafana/provisioning/datasources
cat > grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
EOF

# Create Grafana dashboard configuration
log "Creating Grafana dashboard configuration..."
mkdir -p grafana/provisioning/dashboards
cat > grafana/provisioning/dashboards/dashboard.yml << EOF
apiVersion: 1

providers:
  - name: 'Default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
      foldersFromFilesStructure: true
EOF

# Create a sample dashboard
cat > grafana/provisioning/dashboards/premium_model_dashboard.json << EOF
{
  "annotations": {
    "list": [
      {
        "builtIn": 1,
        "datasource": "-- Grafana --",
        "enable": true,
        "hide": true,
        "iconColor": "rgba(0, 211, 255, 1)",
        "name": "Annotations & Alerts",
        "type": "dashboard"
      }
    ]
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": 1,
  "links": [],
  "panels": [
    {
      "aliasColors": {},
      "bars": false,
      "dashLength": 10,
      "dashes": false,
      "datasource": "Prometheus",
      "fieldConfig": {
        "defaults": {
          "custom": {}
        },
        "overrides": []
      },
      "fill": 1,
      "fillGradient": 0,
      "gridPos": {
        "h": 8,
        "w": 12,
        "x": 0,
        "y": 0
      },
      "hiddenSeries": false,
      "id": 2,
      "legend": {
        "avg": false,
        "current": false,
        "max": false,
        "min": false,
        "show": true,
        "total": false,
        "values": false
      },
      "lines": true,
      "linewidth": 1,
      "nullPointMode": "null",
      "options": {
        "alertThreshold": true
      },
      "percentage": false,
      "pluginVersion": "7.3.7",
      "pointradius": 2,
      "points": false,
      "renderer": "flot",
      "seriesOverrides": [],
      "spaceLength": 10,
      "stack": false,
      "steppedLine": false,
      "targets": [
        {
          "expr": "sum(rate(premium_predictions_total[5m])) by (status)",
          "interval": "",
          "legendFormat": "",
          "refId": "A"
        }
      ],
      "thresholds": [],
      "timeFrom": null,
      "timeRegions": [],
      "timeShift": null,
      "title": "Prediction Rate",
      "tooltip": {
        "shared": true,
        "sort": 0,
        "value_type": "individual"
      },
      "type": "graph",
      "xaxis": {
        "buckets": null,
        "mode": "time",
        "name": null,
        "show": true,
        "values": []
      },
      "yaxes": [
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        },
        {
          "format": "short",
          "label": null,
          "logBase": 1,
          "max": null,
          "min": null,
          "show": true
        }
      ],
      "yaxis": {
        "align": false,
        "alignLevel": null
      }
    }
  ],
  "schemaVersion": 26,
  "style": "dark",
  "tags": [],
  "templating": {
    "list": []
  },
  "time": {
    "from": "now-6h",
    "to": "now"
  },
  "timepicker": {},
  "timezone": "",
  "title": "Premium Model Dashboard",
  "uid": "premium-model",
  "version": 1
}
EOF

log "${GREEN}Monitoring infrastructure set up successfully${NC}"

# Step 6: Initial configuration
log "\n${YELLOW}Step 6: Setting up initial configuration...${NC}"

# Ensure config directory exists
mkdir -p config

# Create .gitkeep file in each empty directory to ensure Git tracks them
for dir in "${directories[@]}"; do
    if [ -d "$dir" ] && [ -z "$(ls -A "$dir")" ]; then
        touch "$dir/.gitkeep"
        log "Created .gitkeep in empty directory: ${GREEN}$dir${NC}"
    fi
done

log "${GREEN}Initial configuration complete${NC}"

# Final message
echo -e "\n${GREEN}==============================================${NC}"
echo -e "${GREEN}      Setup Completed Successfully!          ${NC}"
echo -e "${GREEN}==============================================${NC}"
echo -e "\n${YELLOW}Next Steps:${NC}"
echo -e "1. Activate the virtual environment:"
echo -e "   ${GREEN}source $PYTHON_VENV/bin/activate${NC}"
echo -e "2. Add your data to the ${GREEN}data/raw${NC} directory."
echo -e "3. Run the training pipeline using ${GREEN}make train${NC} or ${GREEN}python -m src.train${NC}"
echo -e "4. Explore other commands using ${GREEN}make help${NC}"
echo -e "\nCheck ${GREEN}$LOG_FILE${NC} for detailed setup logs."

# Deactivate virtual environment if script sourced it
if [[ "$BASH_SOURCE" == "$0" ]]; then
    # Script was executed directly, not sourced
    if type deactivate > /dev/null 2>&1; then
        deactivate
    fi
fi
