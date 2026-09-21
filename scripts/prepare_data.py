from argparse import ArgumentParser
from pathlib import Path
from ser.data import scan_dataset, make_splits
p=ArgumentParser(); p.add_argument('--data',default=None); p.add_argument('--output',default='outputs'); a=p.parse_args()
import os
root=a.data or os.environ.get('DATA_DIR','/data/crema-d'); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
df,bad=scan_dataset(root); splits=make_splits(df); splits.to_csv(out/'split_manifest.csv',index=False); bad.to_csv(out/'invalid_files.csv',index=False)
print(f'Valid files: {len(df)}; invalid/unparsed: {len(bad)}; manifest: {out / "split_manifest.csv"}')
print(splits.groupby(['split','emotion']).size().unstack(fill_value=0).to_string())
