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


def test_main():
    files_before = glob.glob("LL-Output-*.txt")
    LL.main("LebwohlLasher.py", 2, 5, 0.5, 0)
    files_after = glob.glob("LL-Output-*.txt")
    new_files = [f for f in files_after if f not in files_before]
    assert len(new_files) == 1
    os.remove(new_files[0])

