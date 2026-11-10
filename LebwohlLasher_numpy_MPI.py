"""
Run at the command line by typing:

mpiexec -n <NPROCS> python LebwohlLasher_numpy_MPI.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.

The initial configuration is set at random. The boundaries
are periodic throughout the simulation.

How the parallel version works:
   The lattice is split by rows. Each rank owns a block of rows and also
    keeps a copy of the row just above and just below its block ("ghost rows"),
    which it gets from the neighbouring ranks. Similar to the forest fire project.
   All random numbers are made on rank 0, in the same order as the serial
    numpy code, at the start of each MC step. Rank 0 then sends each rank the
    part of the (flattened, 1D) random arrays that matches its rows.
    This means the lattice ends up exactly the same as the serial version
    for the same seed, no matter how many ranks are used.
   Energy and order are added up per rank and then summed on rank 0.
    These can differ from serial in the last few decimal places only,
    because the numbers are added in a different order. (rounding?)

SH 16-Oct-23
"""

import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpi4py import MPI

#=======================================================================
# MPI SET UP
comm = MPI.COMM_WORLD
rank = comm.Get_rank()   # 0 .. size-1
size = comm.Get_size()   # number of processes from mpiexec -n

# Neighbouring ranks (periodic, so the last rank's neighbour below is rank 0)
rank_above = (rank - 1) % size
rank_below = (rank + 1) % size
#=======================================================================
def split_rows(nmax):
    """
    Changes from numpy version:
      New function.
    Arguments:
      nmax (int) = side length of square lattice.
    Description:
      Works out which rows each rank owns. Rows are shared out as evenly
      as possible, the first `remainder` ranks get one extra row.
    Returns:
      rows_per_rank (int(size)) = number of rows each rank gets;
      starts (int(size)) = first global row of each rank;
      counts (int(size)) = number of numbers each rank gets (rows * nmax);
      displs (int(size)) = where each rank's numbers start in the 1D array.
    """
    base, remainder = divmod(nmax, size)
    rows_per_rank = np.full(size, base)
    rows_per_rank[:remainder] += 1           # first 'remainder' ranks get one extra row
    starts = np.cumsum(rows_per_rank) - rows_per_rank
    counts = rows_per_rank * nmax
    displs = starts * nmax
    return rows_per_rank, starts, counts, displs
#=======================================================================
def send_out(full, my_part, counts, displs):
    """
    Changes from numpy version:
      New function.
    Arguments:
      full (float(nmax,nmax)) = full array, only needed on rank 0 (None elsewhere);
      my_part (float(my_rows,nmax)) = where this rank's section is written;
      counts, displs = how the rows are split between ranks (from split_rows).
    Description:
      Rank 0 flattens 'full' to 1D and sends each rank the section that
      matches its rows (MPI Scatterv).
    Returns:
      NULL (result is written into my_part)
    """
    if rank == 0:
        comm.Scatterv([full.flatten(), counts, displs, MPI.DOUBLE], my_part, root=0)
    else:
        comm.Scatterv(None, my_part, root=0)
#=======================================================================
def collect(my_part, nmax, counts, displs):
    """
    Changes from numpy version:
      New function.
    Arguments:
      my_part (float(my_rows,nmax)) = this rank's rows;
      nmax (int) = side length of square lattice;
      counts, displs = how the rows are split between ranks (from split_rows).
    Description:
      Opposite of send_out: puts every rank's rows back together on
      rank 0 (MPI Gatherv).
    Returns:
      full (float(nmax,nmax)) = full lattice on rank 0, None on the others.
    """
    my_part = np.ascontiguousarray(my_part)
    if rank == 0:
        full = np.empty((nmax, nmax))
        comm.Gatherv(my_part, [full, counts, displs, MPI.DOUBLE], root=0)
        return full
    else:
        comm.Gatherv(my_part, None, root=0)
        return None
#=======================================================================
def update_ghost_rows(local):
    """
    Changes from numpy version:
      New function. Replaces the wraparound that np.roll(axis=0) gave in
      the numpy version, as the rows above/below may live on another rank.
    Arguments:
      local (float(my_rows+2,nmax)) = my rows plus the two ghost rows.
        local[0] is the ghost row above my block, local[-1] is the ghost row
        below, local[1:-1] are the rows I actually own.
    Description:
      Swaps edge rows with the neighbouring ranks (periodic) so the ghost
      rows are up to date. Works with 1 rank too (it sends to itself).
    Returns:
      NULL (ghost rows of local are updated in place)
    """
    # My top real row goes to the rank above; I receive my bottom ghost row from the rank below
    comm.Sendrecv(local[1], dest=rank_above, recvbuf=local[-1], source=rank_below)
    # My bottom real row goes to the rank below; I receive my top ghost row from the rank above
    comm.Sendrecv(local[-2], dest=rank_below, recvbuf=local[0], source=rank_above)
#=======================================================================
def initdat(nmax):
    """
    Changes from numpy version:
      None. Only called on rank 0, main() then sends each rank its rows.
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
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
    Changes from numpy version:
      None. Only rank 0 calls this, with the full (gathered) lattice.
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
    Changes from numpy version:
      Adds the number of MPI ranks to the header. Only rank 0 calls this.
    Arguments:
      arr (float(nmax,nmax)) = array that contains lattice data;
      nsteps (int) = number of Monte Carlo steps (MCS) performed;
      Ts (float) = reduced temperature (range 0 to 2);
      runtime (float) = time taken for the MC steps (s);
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
    print("# MPI ranks:           {:d}".format(size),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
#=======================================================================
def one_energy(arr,ix,iy,nmax):
    """
    Changes from numpy version:
      None. Only used by plotdat and MC_step_odd, both on rank 0 with the
      full lattice.
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
def bond_energys(angle1, angle2):
    """
    Changes from numpy version:
      None.
    Arguments:
      angle1, angle2 (float or float array) = angles at each end of the bond(s).
    Description:
      Calcs energy for the bond differences accoring to the Lebwohl-Lasher model.
    Returns:
      bond energy (float or float array, same shape as the inputs).
    """
    return 0.5*(1.0 - 3.0*np.cos(angle1 - angle2)**2)
#=======================================================================
def all_energy(local):
    """
    Changes from numpy version:
      Takes this rank's rows (with ghost rows) instead of the full lattice.
      The vertical neighbour comes from the row below (local[2:]) instead of
      np.roll(axis=0), and the per-rank sums are added together on rank 0.
    Arguments:
      local (float(my_rows+2,nmax)) = my rows plus the two ghost rows.
    Description:
      Each rank works out the energy of the bonds to the right of and below
      each of its own cells (the bond below the last row uses the ghost row).
      Every bond is then counted exactly once over all ranks. Output is in
      reduced units (U/epsilon).
    Returns:
      enall (float) = reduced energy of lattice (on rank 0, None on others).
    """
    update_ghost_rows(local)
    my_rows = local[1:-1]
    row_below = local[2:]
    horizontal_energy = bond_energys(my_rows, np.roll(my_rows, -1, axis=1))
    vertical_energy = bond_energys(my_rows, row_below)
    my_energy = horizontal_energy.sum() + vertical_energy.sum()

    total_energy = comm.reduce(my_energy, op=MPI.SUM, root=0)
    if rank == 0:
        return 2*total_energy # Multiply by 2 to account for double counting of bonds
    return None
#=======================================================================
def get_order(local,nmax):
    """
    Changes from numpy version:
      Takes this rank's rows (with ghost rows) instead of the full lattice.
      Each rank does the einsum over its own cells only, the 3x3 results are
      added together on rank 0 (MPI Reduce), which builds Q and finds the
      eigenvalues.
    Arguments:
      local (float(my_rows+2,nmax)) = my rows plus the two ghost rows;
      nmax (int) = side length of square lattice.
    Description:
      Function to calculate the order parameter of a lattice
      using the Q tensor approach, as in equation (3) of the
      project notes. S_lattice = max(eigenvalues(Q_ab)).
    Returns:
      max(eigenvalues(Qab)) (float) = order parameter (on rank 0, None on others).
    """
    my_rows = local[1:-1]
    x = np.cos(my_rows)
    y = np.sin(my_rows)
    z = np.zeros_like(my_rows)
    lab = np.array([x, y, z])
    my_products = np.einsum('aij,bij->ab', lab, lab)

    # Add the 3x3 arrays from every rank together on rank 0
    products = np.zeros((3,3))
    comm.Reduce(my_products, products, op=MPI.SUM, root=0)

    if rank == 0:
        delta = np.eye(3)
        Qab = (3 * products - nmax * nmax * delta) / (2 * nmax * nmax)
        eigenvalues = np.linalg.eigvalsh(Qab)
        return eigenvalues.max()
    return None
#=======================================================================
def MC_step_odd(arr,Ts,nmax):
    """
    Changes from numpy version:
      Split out of MC_step into its own function, code unchanged. Runs on
      rank 0 only, with the full lattice (main() gathers and re-sends it).
    Arguments:
      arr (float(nmax,nmax)) = array that contains lattice data;
      Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice (odd).
    Description:
      Random-site MC step for odd lattice sizes. With an odd size the
      checkerboard trick does not work (periodic boundaries break it),
      so this is not parallelised.
    Returns:
      accept/(nmax**2) (float) = acceptance ratio for current MCS.
    """
    scale=0.1+Ts
    accept = 0
    xran = np.random.randint(0,high=nmax, size=(nmax,nmax))
    yran = np.random.randint(0,high=nmax, size=(nmax,nmax))
    aran = np.random.normal(scale=scale, size=(nmax,nmax))
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
                boltz = np.exp( -(en1 - en0) / Ts )
                if boltz >= np.random.uniform(0.0,1.0):
                    accept += 1
                else:
                    arr[ix,iy] -= ang
    return accept/(nmax*nmax)
#=======================================================================
def MC_step(local,Ts,nmax,masks,counts,displs):
    """
    Changes from numpy version:
      Even nmax only (odd sizes use MC_step_odd).
      Works on this rank's rows (with ghost rows) instead of the full lattice.
      All random numbers (aran and both uniform arrays) are made on rank 0
      at the start of the step, in the same order as the numpy version, and
      each rank is sent its section of the flattened 1D arrays. This keeps
      the lattice identical to the numpy version for any number of ranks.
      Ghost rows (haolos) are refreshed before each mask, and up/down come from the
      ghost rows instead of np.roll(axis=0).
      Masks are passed in (built once in main() from global row numbers).
      Accepted counts are added together on rank 0.
    Arguments:
      local (float(my_rows+2,nmax)) = my rows plus the two ghost rows;
      Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice;
      masks (list of 2 arrays) = my part of the two checkerboard masks;
      counts, displs = how the rows are split between ranks (from split_rows).
    Description:
      Function to perform one MC step, which consists of an average
      of 1 attempted change per lattice site, using the checkerboard so
      that each mask can be updated all at once.  Working with reduced
      temperature Ts = kT/epsilon.
    Returns:
      accept/(nmax**2) (float) = acceptance ratio (on rank 0, None on others).
    """
    scale=0.1+Ts
    my_n_rows = local.shape[0] - 2

    # Step 1: make the random numbers on rank 0
    if rank == 0:
        aran_full = np.random.normal(scale=scale, size=(nmax,nmax))
        uran1_full = np.random.uniform(0.0, 1.0, size=(nmax,nmax))
        uran2_full = np.random.uniform(0.0, 1.0, size=(nmax,nmax))
    else:
        aran_full = None
        uran1_full = None
        uran2_full = None

    # Step 2: every rank receives its own section
    aran = np.empty((my_n_rows, nmax))
    uran1 = np.empty((my_n_rows, nmax))
    uran2 = np.empty((my_n_rows, nmax))
    send_out(aran_full, aran, counts, displs)
    send_out(uran1_full, uran1, counts, displs)
    send_out(uran2_full, uran2, counts, displs)

    # Step 3: update each mask in turn
    accept = 0
    for mask, uran in [(masks[0], uran1), (masks[1], uran2)]:
        update_ghost_rows(local)

        # my_rows is a view into local, so changing my_rows changes local too
        my_rows = local[1:-1]
        up = local[2:]                         # row below each cell (same as np.roll(arr,-1,axis=0) in serial)
        down = local[:-2]                      # row above each cell (same as np.roll(arr, 1,axis=0) in serial)
        left = np.roll(my_rows, -1, axis=1)
        right = np.roll(my_rows, 1, axis=1)
        directions = [up, down, left, right]

        # Energy before the change
        en0 = np.zeros((my_n_rows, nmax))
        for direction in directions:
            en0 += bond_energys(my_rows, direction)
        en0 = en0 * mask

        # Energy after the change (only cells in this mask are rotated)
        trial = my_rows + aran * mask
        en1 = np.zeros((my_n_rows, nmax))
        for direction in directions:
            en1 += bond_energys(trial, direction)
        en1 = en1 * mask

        # Metropolis test: exp( -(E_new - E_old) / T* ) >= rand(0,1)
        boltz = np.exp(-(en1 - en0) / Ts)
        accepted = (mask == 1) & (boltz >= uran)
        my_rows[accepted] = trial[accepted]
        accept += np.sum(accepted)

    # Add up accepted moves from every rank on rank 0
    total_accept = comm.reduce(int(accept), op=MPI.SUM, root=0)
    if rank == 0:
        return total_accept/(nmax*nmax)
    return None
#=======================================================================
def main(program, nsteps, nmax, temp, pflag, seeded=0, save=True):
    """
    Changes from numpy version:
      Run by every rank. Rank 0 makes the lattice and sends each rank its
      rows; each rank keeps its rows plus two ghost rows in 'local'.
      Builds the checkerboard masks once, from global row numbers.
      Odd sizes gather the lattice on rank 0 for MC_step_odd each step.
      Timing uses MPI.Wtime() between barriers instead of time.time().
      Only rank 0 stores results, prints, saves and plots.
      New optional seeded and save arguments, and returns the results
      (useful for tests).
    Arguments:
      program (string) = the name of the program;
      nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
      temp (float) = reduced temperature (range 0 to 2);
      pflag (int) = a flag to control plotting;
      seeded (int) = if 1, seed the random numbers with 42 (same as the numpy version);
      save (bool) = whether to write the output file.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      lattice, energy, ratio, order on rank 0 (None on the other ranks).
    """
    if size > nmax:
        if rank == 0:
            print("Too many ranks: need at least one row per rank ({} ranks, {} rows)".format(size, nmax))
        sys.exit(1)

    # Only rank 0 makes random numbers, so only rank 0 needs the seed
    if rank == 0 and seeded:
        np.random.seed(42)

    # Work out which rows this rank owns
    rows_per_rank, starts, counts, displs = split_rows(nmax)
    my_n_rows = rows_per_rank[rank]
    my_first_row = starts[rank]

    # local holds my rows plus a ghost row above (local[0]) and below (local[-1])
    local = np.zeros((my_n_rows + 2, nmax))

    # Create and initialise lattice on rank 0, then send each rank its rows
    if rank == 0:
        lattice = initdat(nmax)
        plotdat(lattice,pflag,nmax)
    else:
        lattice = None
    send_out(lattice, local[1:-1], counts, displs)

    # Checkerboard masks. Use the global row number so the pattern lines up between ranks.
    global_rows = np.arange(my_first_row, my_first_row + my_n_rows)
    global_cols = np.arange(nmax)
    first_mask = (global_rows[:, None] + global_cols[None, :]) % 2
    second_mask = 1 - first_mask
    masks = [first_mask, second_mask]

    # Create arrays to store energy, acceptance ratio and order parameter (only used on rank 0)
    energy = np.zeros(nsteps+1)
    ratio = np.zeros(nsteps+1)
    order = np.zeros(nsteps+1)
    # Set initial values. Every rank must call these as they use MPI,
    # but only rank 0 gets the answer back.
    step_energy = all_energy(local)
    step_order = get_order(local,nmax)
    if rank == 0:
        energy[0] = step_energy
        ratio[0] = 0.5 # ideal value
        order[0] = step_order

    # Begin doing and timing some MC steps.
    comm.Barrier()
    initial = MPI.Wtime()
    for it in range(1,nsteps+1):
        if nmax % 2 == 1:
            # Odd size: collect the lattice on rank 0, do a serial step, send it back out
            lattice = collect(local[1:-1], nmax, counts, displs)
            if rank == 0:
                step_ratio = MC_step_odd(lattice,temp,nmax)
            send_out(lattice, local[1:-1], counts, displs)
        else:
            step_ratio = MC_step(local,temp,nmax,masks,counts,displs)
        step_energy = all_energy(local)
        step_order = get_order(local,nmax)
        if rank == 0:
            ratio[it] = step_ratio
            energy[it] = step_energy
            order[it] = step_order
    comm.Barrier()
    final = MPI.Wtime()
    runtime = final-initial

    # Put the final lattice back together on rank 0
    lattice = collect(local[1:-1], nmax, counts, displs)

    # Final outputs (rank 0 only)
    if rank == 0:
        print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s, Ranks: {:d}".format(program, nmax,nsteps,temp,order[nsteps-1],runtime,size))
        if save:
            savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
        plotdat(lattice,pflag,nmax)
        return lattice, energy, ratio, order
    return None
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
        if rank == 0:
            print("Usage: mpiexec -n <NPROCS> python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> [SEEDED]".format(sys.argv[0]))
#=======================================================================
