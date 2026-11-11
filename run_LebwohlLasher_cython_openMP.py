# Run with: OMP_NUM_THREADS=<N> python run_LebwohlLasher_cython_openMP.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> [SEEDED]
import os
import sys

# numpy does its linear algebra (np.linalg.eig in get_order) through a maths
# library called OpenBLAS, which starts one thread per core as soon as numpy is
# imported. After each eig call those threads keep spinning for a while waiting
# for more work, so they compete for the cores with the OpenMP threads in
# all_energy and slow everything down. A 3x3 matrix gains nothing from extra
# threads, so OpenBLAS is limited to one. This must be set before numpy is imported.
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from LebwohlLasher_cython_openMP import main

if len(sys.argv) in (5, 6):
    SEEDED = int(sys.argv[5]) if len(sys.argv) == 6 else 0
    main(sys.argv[0], int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), SEEDED)
else:
    print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> [SEEDED]".format(sys.argv[0]))
