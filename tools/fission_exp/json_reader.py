import pandas as pd
import numpy as np
import sys
import argparse
import pickle

from .spec_analysis import Spec


class Quantity:
    def __init__(self, quantity: str, fmt: str, data: list, meta: list, units: list):
        self.quantity = quantity
        self.data = data
        self.fmt = fmt
        self.meta = meta
        self.units = units

    def get_specs(self, norm="none"):
        assert self.fmt == "x,dx,y,dy"

        def get_normed_spec(data):
            x = data[0, :]
            dx = data[1, :]
            y = data[2, :]
            dy = data[3, :]

            # x must be sorted
            assert np.all(x[:-1] <= x[1:])

            return Spec(y, dy, x, xerr=dx).normalize()

        return [get_normed_spec(d) for d in self.data]


def select(df, key):
    dfq = df[df["quantity"] == key]

    if dfq.empty:
        print("No data found for quantity " + key)

    meta = dfq[
        [
            "quantity",
            "authors",
            "label",
            "reference",
            "web",
            "title",
            "exfor",
            "year",
            "comments",
        ]
    ].to_dict("records")

    print("Found {} results for quantity {}".format(len(meta), key))

    return dfq, meta


def extract_units(row, fmt: list):
    units = []
    for f in fmt:
        if f is None:
            units.append(None)
        else:
            units.append(row[f].to_string(index=None).strip())

    return units


def read_scalar(df, quantity):
    dfq, meta = select(df, quantity)

    data = []
    units = []

    for i, (d, fmt) in enumerate(zip(dfq["data"], dfq["fmt"])):
        entry = dfq.iloc[[i]]
        if fmt == "xy":
            y = d[0][1]
            dy = 0
            units.append(extract_units(entry, ["units-y", None]))
        elif fmt == "xydy":
            y = d[0][1]
            dy = d[0][2]
            units.append(extract_units(entry, ["units-y", "units-dy"]))
        elif fmt == "xdxydy":
            y = d[0][2]
            dy = d[0][3]
            units.append(extract_units(entry, ["units-y", "units-dy"]))
        else:
            print("Invalid format specifier for quantity " + quantity + ": " + fmt)
            exit(1)
        data.append(np.array([y, dy]))

    return Quantity(quantity, "y,dy", data, meta, units)


def read_specs(df, quantity):
    dfq, meta = select(df, quantity)

    specs = []
    units = []

    for i, (d, fmt) in enumerate(zip(dfq["data"], dfq["fmt"])):
        entry = dfq.iloc[[i]]
        data = np.array(list(d))
        data_xdxydy = np.zeros((4, data.shape[0]))
        if fmt == "xy":
            data_xdxydy[0, :] = data[:, 0]
            data_xdxydy[2, :] = data[:, 1]
            units.append(extract_units(entry, ["units-x", None, "units-y", None]))
        elif fmt == "xydy":
            data_xdxydy[0, :] = data[:, 0]
            data_xdxydy[2, :] = data[:, 1]
            data_xdxydy[3, :] = data[:, 2]
            units.append(extract_units(entry, ["units-x", None, "units-y", "units-dy"]))
        elif fmt == "xydydz":
            data_xdxydy[0, :] = data[:, 0]
            data_xdxydy[2, :] = data[:, 1]
            data_xdxydy[3, :] = data[:, 2]
            units.append(extract_units(entry, ["units-x", None, "units-y", "units-dy"]))
        elif fmt == "xdxldxuydy":
            data_xdxydy[0, :] = data[:, 0]
            data_xdxydy[1, :] = data[:, 1]
            data_xdxydy[2, :] = data[:, 3]
            data_xdxydy[3, :] = data[:, 4]
            units.append(
                extract_units(entry, ["units-x", "units-dxl", "units-y", "units-dy"])
            )
        elif fmt == "xdxydy":
            data_xdxydy = data.T
            units.append(
                extract_units(entry, ["units-x", "units-dx", "units-y", "units-dy"])
            )
        else:
            print("Invalid format specifier for quantity " + quantity + ": " + fmt)
            exit(1)
        specs.append(data_xdxydy)

    return Quantity(quantity, "x,dx,y,dy", specs, meta, units)


def read_3D(df, quantity):
    dfq, meta = select(df, quantity)

    specs = []
    units = []

    for i, (d, fmt) in enumerate(zip(dfq["data"], dfq["fmt"])):
        entry = dfq.iloc[[i]]
        data = np.array(list(d))
        data_fm = np.zeros((6, data.shape[0]))
        if fmt == "xdxyz":
            data_fm[:3, :] = data[:, :3]
            data_fm[4, :] = data[:, 3]
            data_fm[5, :] = data_fm[4,:]
            units.append(
                extract_units(
                    entry,
                    ["units-x", "units-dx", "units-y", None, "units-z", "units-z"],
                )
            )
        if fmt == "xminxmaxyz":
            data_fm[0, :] = (data[:, 0] + data[:, 1]) * 0.5
            data_fm[1, :] = data[:, 1] - data[:, 0]
            data_fm[2, :] = data[:, 2]
            data_fm[4, :] = data[:, 3]
            data_fm[5, :] = data_fm[4,:]
            units.append(
                extract_units(
                    entry,
                    ["units-xmin", "units-xmin", "units-y", None, "units-z"]
                )
            )
        elif fmt == "xdxydyz":
            data_fm[:5,...] = data
            data_fm[5, :] = data_fm[4,:]
            units.append(
                extract_units(
                    entry,
                    [
                        "units-x",
                        "units-dx",
                        "units-y",
                        "units-dy",
                        "units-z",
                        None
                    ],
                )
            )
        elif fmt == "xydyz":
            data_fm[0, :] = data[:, 0]
            data_fm[2, :] = data[:, 1]
            data_fm[3, :] = data[:, 2]
            data_fm[4, :] = data[:, 3]
            data_fm[5, :] = data_fm[4,:]

            units.append(
                extract_units(
                    entry,
                    ["units-x", None, "units-y", "units-dy", "units-z", "units-z"],
                )
            )
        elif fmt == "xydyzminzmax":
            data_fm[0, :] = data[:, 0]
            data_fm[2:5, :] = data[:, 1:].T

            units.append(
                extract_units(
                    entry,
                    [
                        "units-x",
                        None,
                        "units-y",
                        "units-dy",
                        "units-zmin",
                        "units-zmax",
                    ],
                )
            )
        elif fmt == "xdxydyzminzmax":
            data_fm = data.T
            units.append(
                extract_units(
                    entry,
                    [
                        "units-x ",
                        "units-dx",
                        "units-y",
                        "units-dy",
                        "units-zmin",
                        "units-zmax",
                    ],
                )
            )
        else:
            print("Invalid format specifier for quantity " + quantity + ": " + fmt)
            exit(1)
        specs.append(data_fm)

    return Quantity(quantity, "x,dx,y,dy,zmin,zmax", specs, meta, units)


def read_pfns(df):
    return [
        read_specs(df, "PFNS"),
        read_specs(df, "PFNS_sqrtE"),
        read_specs(df, "PFNS_max"),
    ]


def read(fname: str, quantity: str, energy_range=None):
    print("parsing {}".format(fname))
    df = pd.DataFrame.from_records(pd.read_json(fname)["entries"])
    if 'Einc' in df:
        df1 = df[ df['Einc'].str.len() >= 1].explode(['Einc', 'data'])
        df1['Einc'] = pd.to_numeric(df1['Einc'])
        if energy_range is not None:
            (emin, emax) = energy_range
            df1 = df1[ (df1['Einc'] >= emin) & ~ (df1['Einc'] < emax)  ]
            return read_json(df1, quantity)

        #TODO sanitize for unlisted Einc
        # some of these entries are just thermal -> set Einc == [2.5E-08]
        # others have incident energy as a dimension -> set Einc == [x]

        df2 = df[ df['Einc'].str.len() == 0]
        df2['data'] = df2['data'].map( lambda x : x[0] )
        return read_json(pd.concat([df, df2]), quantity)
    else:
        df['data'] = df['data'].map( lambda x : x[0] )
        return read_json(df, quantity)


def read_nubarTKEA(df):
    """
    convert all <nu_fragment | A, TKE> to u,du,nu,dnu,TKE_min,TKE_max
    """
    nubarTKEA = read_3D(df, "nubarTKEA")

    return nubarTKEA

def read_nubartTKEA(df):
    """
    convert all <nu_total | A, TKE> to u,du,nu,dnu,TKE_min,TKE_max
    """
    nubartTKEA = read_3D(df, "nubartTKEA")

    return nubartTKEA


def read_json(df: pd.DataFrame, quantity: str):
    q = quantity.replace("HF", "").replace("LF", "")
    if q == "nubar":
        return read_scalar(df, "nubar")
    elif q == "nubarA":
        return read_specs(df, "nubarA")
    elif q == "nubarZ":
        return read_specs(df, "nubarZ")
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
    elif q == "multiplicityRatioA":
        return read_specs(df, "multiplicityRatioA")
    elif q == "encomA":
        return read_specs(df, "EncomA")
    elif q == "egtbarA":
        return read_specs(df, "EgTbarA")
    elif q == "encomTKE":
        return read_specs(df, "EncomTKE")
    elif q == "egtbarTKE":
        return read_specs(df, "EgTbarTKE")
    elif q == "egtbarnu":
        return read_specs(df, "EgTbarnubar")
    elif q == "nubarTKEA":
        return read_nubarTKEA(df)
    elif q == "nubartTKEA":
        return read_nubartTKEA(df)
    elif q == "pfnsA":
        return read_3D(df, "PFNSA")
    elif q == "encomATKE":
        return read_3D(df, "encomATKE")
    else:
        raise ValueError("Unknown quantity: " + quantity)
