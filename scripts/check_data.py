import glob, os, pandas as pd

print('Scanning data/model_ready/*.csv')
mr = sorted(glob.glob('data/model_ready/*.csv'))
print(len(mr), 'files')
for f in mr:
    try:
        s = os.path.getsize(f)
    except Exception:
        s = None
    try:
        df = pd.read_csv(f)
        r = len(df)
    except Exception as e:
        r = f'ERR:{e}'
    print(os.path.basename(f), 'size=', s, 'rows=', r)

print('\nScanning data/download_reports/*')
rep = sorted(glob.glob('data/download_reports/*'))
print(len(rep), 'files')
for f in rep:
    print('\n---', os.path.basename(f))
    try:
        print(open(f, 'r', encoding='utf-8', errors='replace').read())
    except Exception as e:
        print('ERR reading', e)
