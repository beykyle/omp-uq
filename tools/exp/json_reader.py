import pandas as pd
import numpy as np
import sys
import argparse
import pickle

from exfor_plot import Spec

class Quantity:
    def __init__(self, quantity : str,  fmt : str, data : list, meta : list, units : list):
        self.quantity = quantity
        self.data = data
        self.fmt = fmt
        self.meta = meta
        self.units = units

    def get_specs(self):
        assert(self.fmt == "x,dx,y,dy")
        return [ Spec(d[:,2], d[:,3], d[:,0], dx=d[:,1]) for d in self.data ]


def select(df, key):
    dfq = df[ df["quantity"] == key ]

    if dfq.empty:
        print("No data found for quantity " + key)

    meta = dfq[["quantity", "authors", "label", "reference", "web",
                "title", "exfor", "year", "comments"]].to_dict('records')

    print("Found {} results for quantity {}".format(len(meta), key))

    return dfq, meta

def extract_units( row, fmt : list ):
    units = []
    for f in fmt:
        if f is None:
            units.append(None)
        else:
            units.append( row[f].to_string(index=None).strip() )

    return units

def read_scalar(df, quantity):

    dfq, meta = select(df, quantity)

    datay  = []
    datady = []
    units  = []

    for i, (d, fmt) in enumerate(zip(dfq["data"], dfq["fmt"])):
        entry = dfq.iloc[[i]]
        if fmt == "xy":
            y  = d[0][0][1]
            dy = 0
            units.append(extract_units( entry, ["units-y", None] ) )
        elif fmt == "xydy":
            y  = d[0][0][1]
            dy = d[0][0][2]
            units.append(extract_units( entry, ["units-y", "units-dy"] ) )
            print(entry["units-dy"])
        elif fmt == "xdxydy":
            y  = d[0][0][2]
            dy = d[0][0][3]
            units.append(extract_units( entry, ["units-y", "units-dy"] ) )
        else:
            print("Invalid format specifier for quantity " + quantity + ": " + fmt)
            exit(1)
        datay.append(y)
        datady.append(dy)

    data = np.array([datay, datady])
    return Quantity(quantity, "y,dy", data, meta, units)

def read_specs(df, quantity):

    dfq, meta = select(df, quantity)

    specs = []
    units  = []

    for i, (d, fmt) in enumerate(zip(dfq["data"], dfq["fmt"])):
        entry = dfq.iloc[[i]]
        data        = np.array( list(d)[0] )
        data_xdxydy = np.zeros( (4, data.shape[0]) )
        if fmt == "xy":
            data_xdxydy[0,:] = data[:,0]
            data_xdxydy[2,:] = data[:,1]
            units.append(extract_units( entry, ["units-x", None, "units-y", None] ) )
        elif fmt == "xydy":
            data_xdxydy[0,:] = data[:,0]
            data_xdxydy[2,:] = data[:,1]
            data_xdxydy[3,:] = data[:,2]
            units.append(extract_units( entry, ["units-x", None, "units-y", "units-dy"] ) )
        elif fmt == "xdxldxuydy":
            data_xdxydy[0,:] = data[:,0]
            data_xdxydy[1,:] = data[:,1]
            data_xdxydy[2,:] = data[:,3]
            data_xdxydy[3,:] = data[:,4]
            units.append(extract_units( entry, ["units-x", "units-dxl", "units-y", "units-dy"] ) )
        elif fmt == "xdxydy":
            data_xdxydy = data
            units.append(extract_units( entry, ["units-x", "units-dx", "units-y", "units-dy"] ) )
        else:
            print("Invalid format specifier for quantity " + quantity + ": " + fmt)
            exit(1)
        specs.append(data_xdxydy)

    return Quantity(quantity, "x,dx,y,dy", specs, meta, units)

def read_3D(df, quantity):
    dfq, meta = select(df, quantity)

    specs = []
    units  = []

    for i, (d, fmt) in enumerate(zip(dfq["data"], dfq["fmt"])):
        entry = dfq.iloc[[i]]
        data    = np.array( list(d)[0][0] )
        data_fm = np.zeros( (6, data.shape) )
        if fmt == "xdxyz":
            data_fm[:3,:] = data[:,:3]
            data_fm[4,:]  = data[:,3]
            units.append(extract_units( entry,
                                       ["units-x", "units-dx",
                                        "units-y", None,
                                        "units-z", "units-z"] ) )
        elif fmt == "xdxydyz":
            data_fm[:5] = data
            data_fm[5]  = data[:,4]
            units.append(extract_units( entry,
                                       ["units-x", "units-dx",
                                        "units-y", "units-dy",
                                        "units-z", "units-z"] ) )
        elif fmt == "xydyz":
            data_fm[0,:]    = data[:,0]
            data_fm[2:5,:]  = data[:,1:]
            data_fm[5]      = data[:,3]

            units.append(extract_units( entry,
                                       ["units-x", None,
                                        "units-y", "units-dy",
                                        "units-z", "units-z"] ) )
        elif fmt == "xydyzminzmax":
            data_fm[0,:]    = data[:,0]
            data_fm[2:5,:]  = data[:,1:]
            data_fm[5]      = data[:,3]

            units.append(extract_units( entry,
                                       ["units-x"   , None,
                                        "units-y"   , "units-dy",
                                        "units-zmin", "units-zmax"] ) )
        elif fmt == "xdxydyzminzmax":
            data_fm = data
            units.append(extract_units( entry,
                                       ["units-x "  , "units-dx",
                                        "units-y"   , "units-dy",
                                        "units-zmin", "units-zmax"] ) )
        else:
            print("Invalid format specifier for quantity " + quantity + ": " + fmt)
            exit(1)
        specs.append(data_xdxydy)

    return Quantity(quantity, "x,dx,y,dy,zmin,zmax", specs, meta, units)

def read_pfns(df):
    return [ read_specs(df, "PFNS"), read_specs(df, "PFNS_sqrtE"), read_specs(df, "PFNS_max") ]

def read(fname : str, quantity : str):
    df = pd.DataFrame.from_records(pd.read_json(fname)['entries'])
    return read_json(df, quantity)

def read_json(df : pd.DataFrame, quantity : str):
    q = quantity.replace("HF", "").replace("LF", "")
    if q == "nubar":
        return read_scalar(df, "nubar")
    elif q == "nubarA":
        return read_specs(df, "nubarA")
    elif q == "nubarTKE":
        return read_specs(df, "nubarTTKE")
    elif q == "nugbar":
        return read_scalar(df, "nugbar")
    elif q == "nugbarA":
        return read_specs(df, "nugbarA")
    elif q == "nugbarTKE":
        return read_specs(df, "nugbarTTKE")
    elif q == "pfns":
        return read_pfns(df)
    elif q == "pfns_cm":
        return read_specs(df, "PFNS_cm")
    elif q == "pfgs":
        return read_specs(df, "PFGS")
    elif q == "pfgsA":
        return read_specs(df, "PFGSA")
    elif q == "pnu":
        return read_specs(df, "Pnu")
    elif q == "pnug":
        return read_specs(df, "Pnug")
    elif q == "nug / nu":
        return read_specs(df, "multiplicityRatioA")
    elif q == "nubarTKEA":
        return read_3D(df, "nubarTKEA")
    elif q == "pfnsA":
        return read_3D(df, "PFNSA")
    else:
        print("Unkown quantity: " + quantity)
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Read fission quantities from JSON experimental compendium.')
    parser.add_argument("-p", "--path", type=str, dest='fpath', required=True,
                        help='path to JSON experimental compendium')
    parser.add_argument("-q", "--quantity", dest='quantity', type=str, required=True,
                        choices=["nubar", "nugbar", "pfns", "pfgs", "pnu", "pnug", "nug / nu"],
                        help='quantity to read')
    parser.add_argument("-d", "--diff", dest='diff', default=None,
                        choices=["A", "TKE"], type=str,
                        help='variable upon which quantity should be differentiated, (default=None)')
    parser.add_argument('--HF', action=argparse.BooleanOptionalAction,
                        help='find quantity associated only with heavy fragment')
    parser.add_argument('--LF', action=argparse.BooleanOptionalAction,
                        help='find quantity associated only with light fragment')
    parser.add_argument("-o", "--out-path", type=str, dest='opath', required=False,
                        help='path to dump pickled output', default="default_out.pickle")

    args = parser.parse_args()

    if args.HF and args.LF:
        exit(1)

    if args.HF:
        args.quantity += "HF"
    elif args.LF:
        args.quantity += "LF"

    if args.diff is not None:
        args.quantity += args.diff

    quantity = read(args.fpath, args.quantity)

    if args.opath == "default_out.pickle":
        q_str = args.quantity
        if args.HF:
            q_str += "HF"
        if args.LF:
            q_str += "LF"
        diff_str = ""
        if args.diff is not None:
            diff_str = "_given_" + args.diff
        args.opath = args.quantity + diff_str + ".pickle"

    print("Writing " + args.quantity + " to " + args.opath)
    ofile = open(args.opath, 'wb')
    pickle.dump(quantity, ofile)
    ofile.close()
