import os
import glob
import pandas as pd
from datetime import datetime

DATA_DIR = os.environ.get("MARS_DATA_DIR", "data")
DATA_DIR = os.path.abspath(DATA_DIR)
ENV_DIR = os.path.join(DATA_DIR, "copernicus", "env_history")
pattern = os.path.join(ENV_DIR, 'env_history_*.csv')
files = sorted(glob.glob(pattern))

print('Found', len(files), 'env_history files')
for p in files:
    try:
        size = os.path.getsize(p)
        mtime = datetime.fromtimestamp(os.path.getmtime(p)).isoformat()
        # try reading and parsing time
        df = pd.read_csv(p)
        rows = len(df)
        last_date = None
        if rows>0:
            for c in df.columns:
                try:
                    parsed = pd.to_datetime(df[c], errors='coerce')
                    if parsed.notna().sum()>0:
                        last_date = parsed.dropna().max()
                        break
                except Exception:
                    continue
        print(f"{p}\n  size={size} bytes, mtime={mtime}, rows={rows}, last_date={last_date}")
    except Exception as e:
        print(p, 'ERROR', e)
