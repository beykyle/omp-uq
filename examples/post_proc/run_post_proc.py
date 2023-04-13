from mpi4py import MPI
import sys

from omp_uq import HistData


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    data = HistData((int(sys.argv[1]), int(sys.argv[2])), ["nubar"])
    rank_slice = data.post_process(mpi_comm=comm)
    data.gather(comm, rank, size, rank_slice)
    if rank == 0:
        print('Writing output to "{}/*.npy"'.format(str(data.res_dir)))
        sys.stdout.flush()
        data.write()
        data.write_bins()


if __name__ == "__main__":
    main()
