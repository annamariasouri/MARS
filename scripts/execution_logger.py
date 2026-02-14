"""
Execution logger for MARS system - creates persistent execution logs.
"""
import os
import sys
from datetime import datetime
import json

DATA_DIR = os.environ.get("MARS_DATA_DIR", "data")
LOG_DIR = os.path.join(DATA_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def log_execution(component, status, message="", metadata=None):
    """
    Log an execution event with timestamp and metadata.
    
    Args:
        component: str - 'download', 'forecast', 'merge', etc.
        status: str - 'START', 'SUCCESS', 'ERROR', 'WARNING'
        message: str - human-readable message
        metadata: dict - additional data to log
    """
    timestamp = datetime.utcnow().isoformat()
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    
    log_entry = {
        "timestamp": timestamp,
        "component": component,
        "status": status,
        "message": message,
        "metadata": metadata or {}
    }
    
    # Write to daily log file (one file per day)
    daily_log = os.path.join(LOG_DIR, f"execution_{date_str}.jsonl")
    with open(daily_log, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    # Also write to a persistent cumulative log
    cumulative_log = os.path.join(LOG_DIR, "execution_history.jsonl")
    with open(cumulative_log, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    # Print to console
    print(f"[{timestamp}] {component} - {status}: {message}")

def log_download_start(date_str, regions):
    log_execution("download", "START", f"Starting download for {date_str}", 
                  {"date": date_str, "regions": regions})

def log_download_success(date_str, region, expansion, file_count):
    log_execution("download", "SUCCESS", f"Downloaded {region}", 
                  {"date": date_str, "region": region, "expansion": expansion, "files": file_count})

def log_download_error(date_str, region, error_msg):
    log_execution("download", "ERROR", f"Download failed for {region}: {error_msg}", 
                  {"date": date_str, "region": region})

def log_forecast_start(date_str, regions):
    log_execution("forecast", "START", f"Starting forecasts for {date_str}", 
                  {"date": date_str, "regions": regions})

def log_forecast_success(date_str, region, forecast_count):
    log_execution("forecast", "SUCCESS", f"Generated {forecast_count} forecasts for {region}", 
                  {"date": date_str, "region": region, "count": forecast_count})

def log_forecast_error(date_str, region, error_msg):
    log_execution("forecast", "ERROR", f"Forecast failed for {region}: {error_msg}", 
                  {"date": date_str, "region": region})

if __name__ == "__main__":
    # Test logging
    log_execution("test", "SUCCESS", "Logging system initialized")
    print(f"\nLogs will be written to: {LOG_DIR}")
    print(f"Daily log: execution_{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl")
    print(f"Cumulative log: execution_history.jsonl")
