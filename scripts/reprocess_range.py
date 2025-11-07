"""
Simple wrapper to re-run the downloader for an inclusive date range.
Usage:
  python scripts/reprocess_range.py --start 2025-11-01 --end 2025-11-07
Options:
  --dry-run : print commands but don't execute

The script invokes the existing downloader script `download_copernicus_day_test.py --date YYYY-MM-DD` for each date.
It collects return codes and writes a small summary to stdout.
"""
import argparse
from datetime import datetime, timedelta
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
DOWNLOADER = os.path.join(ROOT, 'download_copernicus_day_test.py')

parser = argparse.ArgumentParser()
parser.add_argument('--start', required=True, help='Start date YYYY-MM-DD')
parser.add_argument('--end', required=True, help='End date YYYY-MM-DD')
parser.add_argument('--dry-run', action='store_true', help='Print commands but do not run')
parser.add_argument('--python', default=sys.executable, help='Python executable to run (default: current)')
args = parser.parse_args()

try:
    start = datetime.strptime(args.start, '%Y-%m-%d')
    end = datetime.strptime(args.end, '%Y-%m-%d')
except Exception:
    print('Invalid date format. Use YYYY-MM-DD')
    sys.exit(2)

if end < start:
    print('End date must be >= start date')
    sys.exit(2)

days = (end - start).days + 1
results = []
for i in range(days):
    d = start + timedelta(days=i)
    datestr = d.strftime('%Y-%m-%d')
    cmd = [args.python, DOWNLOADER, '--date', datestr]
    print('\n=== Reprocess date:', datestr, '===')
    print('CMD:', ' '.join(cmd))
    if args.dry_run:
        results.append((datestr, 'dry-run'))
        continue
    try:
        proc = subprocess.run(cmd, cwd=ROOT)
        rc = proc.returncode
        results.append((datestr, rc))
        print(f'Finished {datestr} rc={rc}')
    except Exception as e:
        results.append((datestr, f'error:{e}'))
        print(f'Exception running downloader for {datestr}: {e}')

print('\nSummary:')
for d, r in results:
    print(d, r)

# exit non-zero if any run failed (non-zero rc or exception)
failed = any((isinstance(r, int) and r != 0) or (isinstance(r, str) and r.startswith('error:')) for _, r in results)
if failed:
    sys.exit(4)
else:
    sys.exit(0)
