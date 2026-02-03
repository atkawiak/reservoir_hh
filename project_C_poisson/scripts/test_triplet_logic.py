#!/usr/bin/env python3
import sys
import os
import numpy as np

# sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Add project root to path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, 'src'))

from scripts.find_triplet_ensemble import get_lambda, process_triplet

trial_0 = {
    'gA': 20.2520, 'gL': 0.1444, 'rho': 0.3069, 'bias': 0.5284, 'lambda': 0.0, 'fr': 32.74
}

print("Testing Triplet Logic for Trial 0...")
res = process_triplet(trial_0, 0)

print(f"Results found: {len(res)}")
for r in res:
    print(f"  {r['regime']}: lambda={r['lambda']:.4f}, gL={r['gL']:.4f}")
