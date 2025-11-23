#!/bin/bash

# Get the absolute path of the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set Airflow Home to a local 'airflow' directory within the project
export AIRFLOW_HOME="${PROJECT_DIR}/airflow"

# Set the DAGs folder to the 'dags' directory within the project
export AIRFLOW__CORE__DAGS_FOLDER="${PROJECT_DIR}/dags"

# Force FAB Auth Manager for UI/User support
export AIRFLOW__CORE__AUTH_MANAGER=airflow.providers.fab.auth_manager.fab_auth_manager.FabAuthManager

# Add the virtual environment's bin directory to the system PATH
export PATH="${PROJECT_DIR}/.venv/bin:$PATH"

# Disable example DAGs
export AIRFLOW__CORE__LOAD_EXAMPLES=False

echo "----------------------------------------------------------------"
echo "Starting Airflow Standalone..."
echo "Project Directory: ${PROJECT_DIR}"
echo "Airflow Home:      ${AIRFLOW_HOME}"
echo "DAGs Folder:       ${AIRFLOW__CORE__DAGS_FOLDER}"
echo "----------------------------------------------------------------"
echo "Access the UI at: http://localhost:8080"
echo "----------------------------------------------------------------"

# Ensure airflow directory exists
mkdir -p "${AIRFLOW_HOME}"

# Run airflow standalone (this starts all components)
airflow standalone
