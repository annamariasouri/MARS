import os
from datetime import datetime, timedelta
import subprocess

# Settings
regions = ["thermaikos", "peiraeus", "limassol"]
start_date = datetime(2026, 2, 14)
end_date = datetime(2026, 2, 19)

for region in regions:
    for i in range((end_date - start_date).days + 1):
        date = start_date + timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        print(f"Downloading env_history for {region} {date_str}")
        # Call the Copernicus download script for this date
        subprocess.run([
            "python", ".github/scripts/download_copernicus_day_test.py", "--date", date_str
        ], check=False)
print("Done.")
