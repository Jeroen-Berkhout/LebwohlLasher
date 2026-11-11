"""



Run at the command line by typing:

python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

SH 16-Oct-23
"""
#=======================================================================
# Seeding note:
#   This version cannot be seeded reproducibly in an easy way. Numba keeps
#   its own random number generator, separate from NumPy's, so
#   np.random.seed() called from normal Python only seeds the initial lattice,
#   not the random numbers drawn inside the jitted MC_step. On top of that, with
#   parallel=True numba turns the np.random array calls into parallel loops,
#   and every thread has its own random state, so the stream depends on how
#   the work is split between threads, cant do a fix like in MPI.
#   Attempts made:
#      Seeding inside an @njit helper: matched the numpy version exactly,
#        but only with parallel=False.
#      Drawing the random numbers in a serial @njit function and passing
#        them to a separate parallel update function: also matched exactly,
#        but split MC_step in two, so it was dropped to keep this version
#        simple and close to the numpy one.
#    just gonna make it non-seeded this has been a big task
# Note for self - test with statistical tests. 
#=======================================================================
from numba import njit, prange
import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

#=======================================================================
def initdat(nmax):
    """
    Arguments:
      nmax (int) = size o
        lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr
#=======================================================================
def plotdat(arr,pflag,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    if pflag==0:
        return
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
#=======================================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
#=======================================================================
@njit
def one_energy(arr,ix,iy,nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  ix (int) = x lattice coordinate of cell;
	  iy (int) = y lattice coordinate of cell;
      nmax (int) = side length of square lattice.
    Description:
      Function that computes the energy of a single cell of the
      lattice taking into account periodic boundaries.  Working with
      reduced energy (U/epsilon), equivalent to setting epsilon=1 in
      equation (1) in the project notes.
	Returns:
	  en (float) = reduced energy of cell.
    """
    en = 0.0
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #
#
# Add together the 4 neighbour contributions
# to the energy
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return en
#=======================================================================
@njit
def bond_energys(angle1, angle2):
    """
    Calcs energy for the bond differences (which is the input) accoring to the Lebwohl-Lasher model.
    """
    return 0.5*(1.0 - 3.0*np.cos(angle1 - angle2)**2)
#=======================================================================
def all_energy(arr):
    """
    Changes:
        Removed max, made it call .shape
        Call the lattice, work out differences between each cell and its neighbours (only 2 neccesary), 
        then sum the arrays of differences, this gives total energy of the lattice 
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
      nmax (int) = side length of square lattice.
    Description:
      Function to compute the energy of the entire lattice. Output
      is in reduced units (U/epsilon).
	Returns:
	  enall (float) = reduced energy of lattice.
    """
    enall = 0.0
    #Roll creates shifted copies for vectorisation, this speeds up and removes the 4 x calls for each cell
    horizontal_energy = bond_energys(arr, np.roll(arr, -1, axis=1))
    vertical_energy = bond_energys(arr, np.roll(arr, -1, axis=0))
    return (2*(horizontal_energy.sum() + vertical_energy.sum())) # Multiply by 2 to account for double counting of bonds
#=======================================================================
def get_order(arr,nmax):

   # Identity matrix (the delta_ab term)
    delta = np.eye(3)

    # Turn each angle into a unit vector (x, y, z) with z = 0
    x = np.cos(arr)                                                                                                                                                                                                                                                                                                                                                                                                                                           
    y = np.sin(arr)
    z = np.zeros_like(arr)
    lab = np.array([x, y, z])

    # Multiply every pair of components together and add them up over the
    # whole lattice. This does the same job as the 4 nested loops above.
    products = np.einsum('aij,bij->ab', lab, lab)

    # Build the Q tensor. The delta term gets added once per lattice site,so it is multiplied by the number of sites (nmax*nmax).
    Qab = (3 * products - nmax * nmax * delta) / (2 * nmax * nmax)
    
    # The order parameter is the biggest eigenvalue of Q
    eigenvalues = np.linalg.eigvalsh(Qab)
    return eigenvalues.max()
#=======================================================================
@njit(parallel=True)
def MC_step(arr,Ts,nmax):
    """
    Changes:
        Removed max, made it call .shape
        Split into even and odd lattice sizes, for even sizes we can use a bipartite update scheme, which is faster.
        Numba: np.roll(axis=), np.indices().sum(axis=), keyword arguments to
        np.random and 2D boolean mask indexing are not supported, so the even
        case loops over sites with % nmax wraparound instead. The mask rows
        are split between threads with prange.

    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
    Description:
      Function to perform one MC step, which consists of an average
      of 1 attempted change per lattice site.  Working with reduced
      temperature Ts = kT/epsilon.  Function returns the acceptance
      ratio for information.  This is the fraction of attempted changes
      that are successful.  Generally aim to keep this around 0.5 for
      efficient simulation.
	Returns:
	  accept/(nmax**2) (float) = acceptance ratio for current MCS.
    """
    #
    # Pre-compute some random numbers.  This is faster than
    # using lots of individual calls.  "scale" sets the width
    # of the distribution for the angle changes - increases
    # with temperature.
    scale=0.1+Ts
    accept = 0
    if nmax % 2 == 1:
        xran = np.random.randint(0, nmax, (nmax,nmax))
        yran = np.random.randint(0, nmax, (nmax,nmax))
        aran = np.random.normal(0.0, scale, (nmax,nmax))
        for i in range(nmax):
            for j in range(nmax):
                ix = xran[i,j]
                iy = yran[i,j]
                ang = aran[i,j]
                en0 = one_energy(arr,ix,iy,nmax)
                arr[ix,iy] += ang
                en1 = one_energy(arr,ix,iy,nmax)
                if en1<=en0:
                    accept += 1
                else:
                # Now apply the Monte Carlo test - compare
                # exp( -(E_new - E_old) / T* ) >= rand(0,1)
                    boltz = np.exp( -(en1 - en0) / Ts )

                    if boltz >= np.random.uniform(0.0,1.0):
                        accept += 1
                    else:
                        arr[ix,iy] -= ang
    else:
        # Pre-compute some random numbers for both masks at once.
        # Masks are since the lattice is bipartite, we can update one mask at a time.
        # first mask = sites where (i+j) % 2 == 1, second mask = (i+j) % 2 == 0 (same as numpy version).
        aran = np.random.normal(0.0, scale, (nmax,nmax))
        for mask in (1, 0):
            uran = np.random.uniform(0.0, 1.0, (nmax,nmax))
            # Sites in one mask only have neighbours in the other mask, so rows can be
            # updated in parallel without a thread reading a site another thread is changing.
            for i in prange(nmax):
                for j in range(nmax):
                    if (i+j) % 2 != mask:
                        continue
                    # Neighbours in the same order as numpy: up, down, left, right
                    up = arr[(i+1)%nmax, j]
                    down = arr[(i-1)%nmax, j]
                    left = arr[i, (j+1)%nmax]
                    right = arr[i, (j-1)%nmax]

                    # Energy of the site with its 4 neighbours (same as one_energy)
                    en0 = 0.0
                    for direction in (up, down, left, right):
                        en0 += bond_energys(arr[i,j], direction)

                    # Trial angle: only this site is rotated, its neighbours are all off-mask
                    trial = arr[i,j] + aran[i,j]
                    en1 = 0.0
                    for direction in (up, down, left, right):
                        en1 += bond_energys(trial, direction)

                    # Metropolis / montecarlo test: exp( -(E_new - E_old) / T* ) >= rand(0,1)
                    boltz = np.exp(-(en1 - en0) / Ts)
                    if boltz >= uran[i,j]:
                        arr[i,j] = trial
                        accept += 1
    return accept/(nmax*nmax)
#=======================================================================
def main(program, nsteps, nmax, temp, pflag, seeded=0):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting;
	  seeded (int) = if 1, seed the random numbers with 42 for reproducible runs.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """
    # Seed before initdat so the starting lattice is reproducible too
    if seeded:
        np.random.seed(42)
    # Create and initialise lattice
    lattice = initdat(nmax)
    # Plot initial frame of lattice
    plotdat(lattice,pflag,nmax)
    # Create arrays to store energy, acceptance ratio and order parameter
    energy = np.zeros(nsteps+1)
    ratio = np.zeros(nsteps+1)
    order = np.zeros(nsteps+1)
    # Set initial values in arrays
    energy[0] = all_energy(lattice)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(lattice,nmax)

    # Warm-up: the first MC_step call compiles it, so do that on a copy
    # before the timer starts. Compile time is reported separately.
    compile_start = time.time()
    MC_step(lattice.copy(),temp,nmax)
    print("Numba compile time: {:8.6f} s".format(time.time()-compile_start))

    # Begin doing and timing some MC steps.
    initial = time.time()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(lattice,temp,nmax)
        energy[it] = all_energy(lattice)
        order[it] = get_order(lattice,nmax)
    final = time.time()
    runtime = final-initial
    
    # Final outputs
    print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order[nsteps-1],runtime))
    # Plot final frame of lattice and generate output file
    savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
    plotdat(lattice,pflag,nmax)
#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) in (5, 6):
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        SEEDED = int(sys.argv[5]) if len(sys.argv) == 6 else 0
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, SEEDED)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> [SEEDED]".format(sys.argv[0]))
#=======================================================================
