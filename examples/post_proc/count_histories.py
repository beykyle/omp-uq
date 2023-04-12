import sys
from CGMFtk import histories as fh

if __name__ == "__main__":
    hist = fh.Histories(sys.argv[1], ang_mom_printed=True)
    print(int(len(hist.getFissionHistories()) / 2))
