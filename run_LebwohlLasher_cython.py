# Run with: python run_LebwohlLasher_cython.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>
import sys
from LebwohlLasher_cython import main

if len(sys.argv) == 5:
    main(sys.argv[0], int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
else:
    print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
