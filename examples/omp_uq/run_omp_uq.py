# instructions to run:
#   mpirun -n {nproc} python run_comp_uq.py

from pyCGMF import CGMF_Input, run
from CGMFtk.histories import Histories

import sys
import numpy as np
from mpi4py import MPI
from pathlib import Path


def run_cgmf_mpi(inp: CGMF_Input, comm, name: str):
    # run worker on each rank
    hists = run(inp)

    # gather histories from all MPI ranks
    result = comm.gather(hists.histories, root=0)

    # concatenate them and print the result
    if inp.MPI_rank == 0:
        # create new Histories object with history info from all MPI ranks,
        # write the output to disk as binary np array
        all_histories = Histories(from_arr=np.concatenate(result, axis=0))
        out_fname = Path("./histories_{}.npy".format(name))
        all_histories.save(out_fname)
        print("Writing sample output to {}".format(out_fname))
        return all_histories

    return None


def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    histories_per_sample = 2400
    histories_per_MPI_rank = histories_per_sample // size

    sample_omp_path = Path("/home/beykyle/omplib/data/KDUQSamples/samples/")

    (start, end) = (int(sys.argv[1]), int(sys.argv[2]))
    end += 1

    for s in range(start, end):
        name = str(s)
        omp_sample_fname = sample_omp_path / "kd_{}.json".format(s)
        assert omp_sample_fname.is_file()
        if rank == 0:
            print(
                "Running CGMF w/ sample file {}, with {} histories on {} ranks."
                "\nHistories per MPI rank: {}".format(
                    omp_sample_fname,
                    histories_per_sample,
                    rank,
                    histories_per_MPI_rank,
                )
            )

        inp = CGMF_Input(
            nevents=histories_per_MPI_rank,
            zaid=98252,
            einc=0.0,
            omp_fpath=str(omp_sample_fname),
            MPI_rank=rank,
            seed=(s + 177713957),
        )

        # run on all workers
        run_cgmf_mpi(inp, comm, name)


if __name__ == "__main__":
    main()
