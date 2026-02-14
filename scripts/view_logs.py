"""
View execution logs - shows what happened each day.
"""
import os
import json
from datetime import datetime, timedelta
from collections import defaultdict
import sys

DATA_DIR = os.environ.get("MARS_DATA_DIR", "data")
LOG_DIR = os.path.join(DATA_DIR, "logs")

def load_logs(days=30):
    """Load logs from the last N days."""
    logs = []
    cumulative = os.path.join(LOG_DIR, "execution_history.jsonl")
    
    if os.path.exists(cumulative):
        with open(cumulative, "r") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except:
                    continue
    
    # Filter to last N days
    cutoff = datetime.utcnow() - timedelta(days=days)
    logs = [l for l in logs if datetime.fromisoformat(l["timestamp"]) >= cutoff]
    
    return logs

def summarize_by_date(logs):
    """Group logs by date and component."""
    by_date = defaultdict(lambda: defaultdict(list))
    
    for log in logs:
        date = log["timestamp"][:10]  # YYYY-MM-DD
        component = log["component"]
        by_date[date][component].append(log)
    
    return by_date

def print_summary(logs, days=30):
    """Print a summary of execution history."""
    if not logs:
        print(f"No logs found in {LOG_DIR}")
        print("\nLogs will be created when you run:")
        print("  - download_copernicus_day_test.py")
        print("  - run_daily_forecast.py")
        return
    
    by_date = summarize_by_date(logs)
    dates = sorted(by_date.keys(), reverse=True)[:days]
    
    print(f"\n{'='*80}")
    print(f"MARS EXECUTION HISTORY (last {days} days)")
    print(f"{'='*80}\n")
    
    for date in dates:
        components = by_date[date]
        print(f"📅 {date}")
        print(f"   {'─'*70}")
        
        for component in sorted(components.keys()):
            events = components[component]
            statuses = [e["status"] for e in events]
            
            success = statuses.count("SUCCESS")
            errors = statuses.count("ERROR")
            warnings = statuses.count("WARNING")
            
            status_str = f"✅ {success}" if success else ""
            status_str += f"  ⚠️ {warnings}" if warnings else ""
            status_str += f"  ❌ {errors}" if errors else ""
            
            print(f"   {component:15s} {status_str}")
            
            # Show metadata for key events
            for event in events:
                if event["status"] in ["ERROR", "SUCCESS"] and event.get("metadata"):
                    meta = event["metadata"]
                    if "region" in meta:
                        details = f"region={meta['region']}"
                        if "expansion" in meta:
                            details += f", expansion={meta['expansion']}"
                        if "count" in meta:
                            details += f", forecasts={meta['count']}"
                        print(f"      └─ {details}")
        
        print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="View MARS execution logs")
    parser.add_argument("--days", type=int, default=30, help="Show last N days")
    args = parser.parse_args()
    
    logs = load_logs(args.days)
    print_summary(logs, args.days)
    
    print(f"\n{'='*80}")
    print(f"Total events: {len(logs)}")
    print(f"Log directory: {LOG_DIR}")
    print(f"{'='*80}\n")
