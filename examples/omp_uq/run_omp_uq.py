# instructions to run:
#   mpirun -n {nproc} python run_comp_uq.py

from pyCGMF import Input, run
from CGMFtk.histories import Histories

import sys
import numpy as np
from mpi4py import MPI
from pathlib import Path


def run_cgmf_mpi(inp: Input, comm, name: str, output_dir: Path):
    print("Running {} histories on rank {}".format(inp.nevents, inp.MPI_rank))
    sys.stdout.flush()

    # run worker on each rank
    hists = run(inp)

    # gather histories from all MPI ranks
    result = comm.gather(hists.histories, root=0)

    # concatenate them and print the result
    if inp.MPI_rank == 0:
        # create new Histories object with history info from all MPI ranks,
        # write the output to disk as binary np array
        all_histories = Histories(from_arr=np.concatenate(result, axis=0))
        out_fname = Path(output_dir / "histories_{}.npy".format(name))
        all_histories.save(out_fname)
        print("Writing output to {}".format(out_fname))
        sys.stdout.flush()
        return all_histories

    return None


def run_samples(
    start,
    end,
    sample_omp_dir,
    output_dir,
    omp_prefix,
    zaid=98252,
    einc=0.0,
    histories_per_sample=1e6,
):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    histories_per_sample = int(histories_per_sample)
    histories_per_MPI_rank = histories_per_sample // size

    if rank == 0:
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)

    for s in range(start, end+1):
        name = str(s)
        omp_sample_fname = sample_omp_dir / f"{omp_prefix}_{s}.json"
        assert omp_sample_fname.is_file()
        sys.stdout.flush()
        if rank == 0:
            print(
                "Running CGMF w/ sample file {}, "
                "with {} histories, on {} ranks.\n"
                "Histories per MPI rank: {}".format(
                    omp_sample_fname,
                    histories_per_sample,
                    size,
                    histories_per_MPI_rank,
                )
            )
        sys.stdout.flush()

        inp = Input(
            nevents=int(histories_per_MPI_rank),
            zaid=zaid,
            einc=einc,
            MPI_rank=rank,
            seed=3957 * s + s + 13,
            omp_fpath=str(omp_sample_fname),
        )

        # run on all workers
        run_cgmf_mpi(inp, comm, name, output_dir)


def main():

    run_samples(
        start = int(sys.argv[1]),
        end = int(sys.argv[2]),
        sample_omp_dir = Path("/home/beykyle/omplib/data/WLHSamples/samples/"),
        output_dir = Path("/home/beykyle/turbo/omp_uq/run3_all/cf252/wlh"),
        omp_prefix = "wlh",
        zaid = 98252,
        einc = 0.0,
    )


if __name__ == "__main__":
    main()
