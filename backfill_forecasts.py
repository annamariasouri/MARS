"""
Backfill forecasts for historical dates where model_ready_input files exist.
"""
import os
import subprocess
from datetime import datetime, timedelta
from glob import glob

# Find all model_ready_input files in root directory
root = os.path.dirname(os.path.abspath(__file__))
pattern = os.path.join(root, "model_ready_input_*_*.csv")
files = glob(pattern)

# Extract unique dates
dates = set()
for f in files:
    basename = os.path.basename(f)
    # Extract date from filename: model_ready_input_{region}_{date}.csv
    parts = basename.replace('.csv', '').split('_')
    if len(parts) >= 5:
        date_str = parts[-1]  # Last part should be the date YYYY-MM-DD
        try:
            date = datetime.strptime(date_str, '%Y-%m-%d')
            dates.add(date_str)
        except ValueError:
            continue

dates = sorted(dates)
print(f"Found {len(dates)} unique dates with model_ready_input files:")
for d in dates:
    print(f"  {d}")

print("\n" + "="*60)
response = input(f"\nRun forecasts for all {len(dates)} dates? (y/n): ")
if response.lower() != 'y':
    print("Cancelled.")
    exit(0)

# Move model_ready files to data/model_ready/ if needed
data_dir = os.path.join(root, 'data', 'model_ready')
os.makedirs(data_dir, exist_ok=True)

print("\nMoving model_ready files to data/model_ready/...")
for f in files:
    basename = os.path.basename(f)
    dest = os.path.join(data_dir, basename)
    if not os.path.exists(dest):
        os.rename(f, dest)
        print(f"  Moved: {basename}")

# Run forecast for each date
forecast_script = os.path.join(root, 'run_daily_forecast.py')

for date_str in dates:
    print(f"\n{'='*60}")
    print(f"Running forecast for {date_str}...")
    
    try:
        # Run forecast script with --date argument
        result = subprocess.run(['python', forecast_script, '--date', date_str], 
                              capture_output=True, 
                              text=True,
                              timeout=120)
        
        if result.returncode == 0:
            print(f"✅ Success for {date_str}")
            if result.stdout:
                # Show summary line if present
                for line in result.stdout.split('\n'):
                    if '✅' in line or 'complete' in line.lower():
                        print(f"  {line.strip()}")
        else:
            print(f"⚠️ Warning for {date_str}:")
            print(result.stderr[:500])
            
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout for {date_str}")
    except Exception as e:
        print(f"❌ Error for {date_str}: {e}")

print(f"\n{'='*60}")
print("Backfill complete! Check data/forecasts/forecast_log_*.csv for results.")
