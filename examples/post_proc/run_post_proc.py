from mpi4py import MPI
from omp_uq import HistData, all_quantities
import sys

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    data = HistData((int(sys.argv[1]), int(sys.argv[2])), all_quantities)
    data.post_process(mpi_comm=comm)
    data.gather(comm, rank, size)
    if rank == 0:
        print('Writing output to "{}/*.npy"'.format(str(data.res_dir)))
        sys.stdout.flush()
        data.write()
        data.write_bins()

if __name__ == "__main__":
    main()
