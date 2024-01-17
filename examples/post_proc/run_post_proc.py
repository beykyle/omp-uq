import sys
from mpi4py import MPI
from pathlib import Path

from cgmf_analysis import HistData


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    histories_path = Path(sys.argv[3])
    results_path = Path(sys.argv[4])

    if rank == 0:
        assert(histories_path.is_dir())

        if not results_path.is_dir():
            results_path.mkdir(parents=True)

    qs = [
        "nugbar",
        "nubar",
        "enbar",
        "egbar",
        "pfns",
        "encomTKE",
        "encomA",
        "encomATKE",
        "pfnscomA",
        "nuATKE",
        "nutATKE",
        "pnu",
        "pnug",
        "nubarA",
        "nubarTKE",
        "nubarA",
    ]

    data = HistData(
        (int(sys.argv[1]), int(sys.argv[2])),
        quantities=qs,
        hist_dir=histories_path,
        res_dir=results_path,
        convert_cgmf_to_npy=False,
    )

    # gamma threshold and cutoff for Oberstedt and Cyzsh nugbar
    data.Ethg = 0.1
    data.Emaxg = 6.5
    #data.max_time = 5.0E-9 #s

    rank_slice = data.post_process(mpi_comm=comm)
    data.gather(comm, rank, size, rank_slice)
    if rank == 0:
        print('Writing output to "{}/*.npy"'.format(str(data.res_dir)))
        sys.stdout.flush()
        data.write()
        data.write_bins()


if __name__ == "__main__":
    main()
