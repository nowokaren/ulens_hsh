import os
import pandas as pd

main_path = "/share/storage3/rubin/microlensing/CASLEO/HSH/"

df = pd.read_csv("image_collection_clean.csv")

missing = []

for _, r in df.iterrows():
    path = f"{main_path}{r['NGT_OLD']}/{r['FIL_OLD']}"
    if not os.path.exists(path):
        missing.append(path)

print("Missing files:", len(missing))

if len(missing) > 0:
    print("\nExamples:")
    for p in missing[:10]:
        print(p)