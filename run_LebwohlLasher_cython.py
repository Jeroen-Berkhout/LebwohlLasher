# Run with: python run_LebwohlLasher_cython.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> [SEEDED]
import sys
from LebwohlLasher_cython import main

if len(sys.argv) in (5, 6):
    SEEDED = int(sys.argv[5]) if len(sys.argv) == 6 else 0
    main(sys.argv[0], int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), SEEDED)
else:
    print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> [SEEDED]".format(sys.argv[0]))
