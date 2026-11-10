"""
Tests for LebwohlLasher.py

Run from the project root with:
    pytest tests/LLtests.py
"""
import os
import sys
import glob

import matplotlib
matplotlib.use("Agg")  # stops plot windows popping up
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import LebwohlLasher as LL


def test_initdat():
    arr = LL.initdat(5)
    assert arr.shape == (5, 5)
    assert arr.min() >= 0
    assert arr.max() < 2 * np.pi


def test_plotdat():
    arr = np.zeros((5, 5))
    for pflag in [0, 1, 2, 3]:
        LL.plotdat(arr, pflag, 5)


def test_savedat():
    arr = np.zeros((5, 5))
    data = np.ones(4)
    files_before = glob.glob("LL-Output-*.txt")
    LL.savedat(arr, 3, 0.5, 1.0, data, data, data, 5)
    files_after = glob.glob("LL-Output-*.txt")
    new_files = [f for f in files_after if f not in files_before]
    assert len(new_files) == 1
    os.remove(new_files[0])


def test_one_energy():
    arr = np.zeros((5, 5))
    # 4 neighbours, each one gives -1
    assert round(LL.one_energy(arr, 2, 2, 5), 6) == -4.0


def test_all_energy():
    arr = np.zeros((5, 5))
    # 25 cells, each one gives -4
    assert round(LL.all_energy(arr, 5), 6) == -100.0


def test_get_order():
    arr = np.zeros((5, 5))
    assert round(LL.get_order(arr, 5), 6) == 1.0


def test_MC_step():
    np.random.seed(0)
    arr = LL.initdat(5)
    ratio = LL.MC_step(arr, 0.5, 5)
    assert ratio >= 0
    assert ratio <= 1


# Saved from the original LebwohlLasher.py: seed 42, 50x50 lattice, T* = 0.5.
# One row per MC step for the first 10 steps: [ratio, energy, order, sum of all angles]
SAVED_STATES = [
    [0.686400, -3794.807467, 0.256393, 7822.295703],
    [0.617200, -4468.355196, 0.261813, 7805.251128],
    [0.590000, -5037.293206, 0.269975, 7829.338315],
    [0.563200, -5493.344513, 0.274953, 7838.785925],
    [0.547200, -5847.513129, 0.277248, 7817.006662],
    [0.508000, -6137.313793, 0.277063, 7830.101159],
    [0.505600, -6372.814149, 0.274737, 7831.715494],
    [0.490800, -6565.672747, 0.275068, 7834.865324],
    [0.472000, -6665.576506, 0.283359, 7825.569548],
    [0.483200, -6789.370368, 0.292047, 7843.460340],
]


def test_first_10_states():
    np.random.seed(42)
    arr = LL.initdat(50)
    for state in SAVED_STATES:
        ratio = LL.MC_step(arr, 0.5, 50)
        assert round(ratio, 6) == state[0]
        assert round(LL.all_energy(arr, 50), 6) == state[1]
        assert round(LL.get_order(arr, 50), 6) == state[2]
        assert round(arr.sum(), 6) == state[3]


def within_5_percent(value, expected):
    return abs(value - expected) <= 0.05 * abs(expected)


# Saved from the original LebwohlLasher.py: seed 42, 50x50 lattice, T* = 0.5, 300 steps.
# Averages over the last 100 steps, once the lattice has settled.
# Seed-to-seed spread of the original (8 seeds): ratio and energy within ~1.3%, so 5% is safe.
# Order is not checked: after 300 steps a 50x50 lattice is still far from ordered (~0.42)
# and the original gives order values up to ~50% apart for different seeds.
SETTLED_RATIO = 0.41448
SETTLED_ENERGY = -8391.429352


# REMINDER: use seed_variation() for the report, to show the seed-to-seed spread
# of the original and justify the 5% tolerance (and why order is not tested).
# It is slow (about a minute per seed for the original), so it is commented out.
#
# def seed_variation(nseeds=8, nmax=50, nsteps=300, temp=0.5):
#     results = []
#     for seed in range(nseeds):
#         np.random.seed(seed)
#         arr = LL.initdat(nmax)
#         ratios = []
#         energies = []
#         orders = []
#         for step in range(nsteps):
#             ratios.append(LL.MC_step(arr, temp, nmax))
#             energies.append(LL.all_energy(arr, nmax))
#             orders.append(LL.get_order(arr, nmax))
#         settled = nsteps - 100
#         results.append([np.mean(ratios[settled:]), np.mean(energies[settled:]), np.mean(orders[settled:])])
#     results = np.array(results)
#     mean = results.mean(axis=0)
#     spread = results.std(axis=0) / np.abs(mean) * 100
#     print("mean:", mean)
#     print("std %:", spread)
#     print("max dev %:", (np.abs(results - mean) / np.abs(mean) * 100).max(axis=0))
#     return results


def test_settled_numpy():
    # The numpy version uses a checkerboard update for even sizes, so the random path is
    # different from the original. Once settled the averages should still agree within 5%.
    import LebwohlLasher_numpy as LLnp
    np.random.seed(42)
    arr = LLnp.initdat(50)
    ratios = []
    energies = []
    for step in range(300):
        ratios.append(LLnp.MC_step(arr, 0.5, 50))
        energies.append(LLnp.all_energy(arr))
    assert within_5_percent(np.mean(ratios[200:]), SETTLED_RATIO)
    assert within_5_percent(np.mean(energies[200:]), SETTLED_ENERGY)


def test_main():
    files_before = glob.glob("LL-Output-*.txt")
    LL.main("LebwohlLasher.py", 2, 5, 0.5, 0)
    files_after = glob.glob("LL-Output-*.txt")
    new_files = [f for f in files_after if f not in files_before]
    assert len(new_files) == 1
    os.remove(new_files[0])

