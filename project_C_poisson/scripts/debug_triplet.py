#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from scripts.find_triplet_ensemble import get_lambda, process_triplet

# Pick a point from the CSV
in_csv = os.path.join(ROOT, 'results/critical_ensemble_params.csv')
df = pd.read_csv(in_csv)
row = df.iloc[0].to_dict()

print(f"DEBUG: Processing row 0: {row}")
# We need to reach into get_lambda to see phi
import scripts.find_triplet_ensemble as ste
lam, fr = ste.get_lambda(row, 0)

print(f"DEBUG: lambda={lam}, fr={fr}")
