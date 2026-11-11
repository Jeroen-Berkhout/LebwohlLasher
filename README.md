# LebwohlLasher

Speeding up a Monte Carlo simulation of the Lebwohl-Lasher liquid crystal model.
`LebwohlLasher.py` is the original code. The other files are optimised versions of it:
NumPy, Numba, Cython, Cython with OpenMP, and NumPy with MPI.

## Setup

You need Python 3 with numpy, matplotlib, numba, cython and mpi4py (and an MPI install for mpi4py).

```
pip install numpy matplotlib numba cython mpi4py pytest
```

The two Cython versions have to be compiled once before they can be run:

```
python setup_LebwohlLasher_cython.py build_ext --inplace
python setup_LebwohlLasher_cython_openMP.py build_ext --inplace
```

## Running

Every version takes the same arguments as the original:

```
python <file> <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> [SEEDED]
```

- ITERATIONS: number of Monte Carlo steps
- SIZE: lattice is SIZE x SIZE
- TEMPERATURE: reduced temperature T*
- PLOTFLAG: 0 for no plot, 1 for energy, 2 for angles, 3 for black and white
- SEEDED: optional, 1 seeds the random numbers with 42 so runs can be repeated. The original doesn't take this, and in Numba it has no real effect (see the comment at the top of that file)

For example, 50 steps on a 50x50 lattice at T* = 0.5 with no plot:

```
python LebwohlLasher.py 50 50 0.5 0
python LebwohlLasher_numpy.py 50 50 0.5 0
python LebwohlLasher_numpy_numba.py 50 50 0.5 0
python run_LebwohlLasher_cython.py 50 50 0.5 0
python run_LebwohlLasher_cython_openMP.py 50 50 0.5 0
mpiexec -n 4 python LebwohlLasher_numpy_MPI.py 50 50 0.5 0
```

To choose the number of threads for the parallel versions:

```
OMP_NUM_THREADS=8 python run_LebwohlLasher_cython_openMP.py 50 50 0.5 0
NUMBA_NUM_THREADS=8 python LebwohlLasher_numpy_numba.py 50 50 0.5 0
```

Each run prints its run time and writes an `LL-Output-<date>.txt` file with the
energy, acceptance ratio and order parameter at every step.

The optimised versions update the lattice in a checkerboard pattern, which needs an even SIZE.
With an odd SIZE they fall back to the original one-site-at-a-time update.

## Tests

```
pytest tests/LLtests.py
```

## Data

`data/runs/` holds the output files from the runs used in the report, mostly done on one
72-core node of an HPC cluster. Each folder name gives the settings of that run, for example
`numba_N1000_S5_T0.500_w8_r0` is the Numba version, a 1000x1000 lattice, 5 steps, T* = 0.5,
8 threads, repeat 0.

- `serial`: every version on one core at sizes 32 to 1000
- `strong`: the parallel versions on 1 to 72 workers at a fixed size
- `weak`: the parallel versions with the lattice growing with the number of workers
- `science`: longer runs across a range of temperatures
