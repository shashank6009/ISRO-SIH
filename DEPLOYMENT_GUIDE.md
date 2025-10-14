# GENESIS-AI Deployment Guide

## Dataset Access Configuration

The dashboard is now configured for robust dataset access across different deployment environments:

### Local Development
- Datasets located in: `src/DATASETS/`
- Files: `DATA_GEO_Train.csv`, `DATA_MEO_Train.csv`, `DATA_MEO_Train2.csv`

### Deployment Options

#### Option 1: Docker Deployment
```dockerfile
# Add to Dockerfile
COPY src/DATASETS/ /app/DATASETS/
```

#### Option 2: Cloud Deployment (Streamlit Cloud, Heroku, etc.)
1. Move datasets to root: `datasets/`
2. Or keep in `src/DATASETS/` (auto-detected)

#### Option 3: Environment Variables
```bash
export DATASET_PATH="/path/to/datasets"
```

### Path Resolution
The app automatically searches for datasets in:
1. `src/DATASETS/` (current structure)
2. `src/datasets/` (alternative)
3. `datasets/` (deployed structure)
4. `DATASETS/` (root level)
5. `data/` (common data folder)
6. Current directory

### Files Required for Deployment
```
├── src/
│   ├── DATASETS/
│   │   ├── DATA_GEO_Train.csv
│   │   ├── DATA_MEO_Train.csv
│   │   └── DATA_MEO_Train2.csv
│   └── genesis_ai/
│       └── app/
│           └── competition_dashboard.py
├── .streamlit/
│   └── config.toml
├── requirements.txt
└── README.md
```

### Docker Deployment Example
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY .streamlit/ ./.streamlit/

EXPOSE 8502

CMD ["streamlit", "run", "src/genesis_ai/app/competition_dashboard.py", "--server.port=8502"]
```

### Verification
The dashboard will show specific error messages if datasets are not accessible, making troubleshooting easier during deployment.
