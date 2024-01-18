import pandas as pd
import numpy as np
import sys
import argparse
import pickle
from pathlib import Path
from x4i3 import exfor_manager
import json

from .spec_analysis import Spec


def exfor_to_json(entry: int,  subentry: int, quantity: str, label_mapping=None):
    db = exfor_manager.X4DBManagerDefault()

    # grab entry
    query = next(iter(db.retrieve(ENTRY=entry).values())).getSimplifiedDataSets()
    if query == {}:
        return

    # will throw key error if entry and subentry are not in query
    data_set = query[(str(entry), str(entry*1000 + subentry), ' ')]

    # convert subentry meta fields string to dict
    meta = [
        [val.strip() for val in line.split(":")]
        for line in data_set.strHeader().split("#")[1:]
    ]
    meta = dict([[line[0].lower(), line[1]] for line in meta])

    # set expected meta fields
    idx_end_surname = meta["authors"].find(" ")
    idx_beg_surname = meta["authors"][0:idx_end_surname].rfind(".") + 1
    first_auth_surname = meta["authors"][idx_beg_surname:idx_end_surname]
    year = meta['year']
    meta['label'] = f"{first_auth_surname} et al., {year}"
    meta['exfor'] = meta.pop("subent")
    meta["quantity"] = quantity

    # guess at label mapping if not provided
    labels = data_set.labels
    dim_symbols = ["x", "y", "z"]
    if label_mapping is None:
        label_mapping = dict()
        non_err_dims = 0
        for i, label in enumerate(labels):
            if not label.find("ERR") > 0:
                symbol = dim_symbols[non_err_dims]
                label_mapping[label] = symbol
                non_err_dims += 1
            else:
                symbol = f"d{symbol}"
                label_mapping[label] = symbol

    # invert label mapping
    symbol_map = {v: k for k, v in label_mapping.items()}
    order = ["x", "dx", "y", "dy", "z", "dz"]
    order = [sym for sym in order if sym in symbol_map.keys()]

    # grab exfor indices in desired order
    def exfor_index_of(label):
        return next(i for i in range(len(labels)) if labels[i] == label)

    exfor_indices = [exfor_index_of(symbol_map[sym]) for sym in order]

    # handle fmt and units
    meta["fmt"] = ""
    mask = [label in label_mapping for label in labels]
    for symbol, i in zip(order, exfor_indices):
            meta["fmt"] += f"{symbol}"
            meta[f"units-{symbol}"] = data_set.units[i].lower()

    # santize data and put in desired order
    def reorder_santize(line):
        ordered_line = []
        for i in exfor_indices:
            ordered_line.append(float(line[i]) if line[i] is not None else 0.0)
        return ordered_line

    meta["data"] = [[reorder_santize(line) for line in data_set.data]]

    return meta

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


def select(df, key, allowed_labels):
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

    if allowed_labels is not None:
        meta = [m for m in meta if m["label"] in allowed_labels]
        dfq = dfq.apply(lambda row: row[df["label"].isin(allowed_labels)])

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


def read_scalar(df, quantity, allowed_labels):
    dfq, meta = select(df, quantity, allowed_labels)

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


def read_specs(df, quantity, allowed_labels):
    dfq, meta = select(df, quantity, allowed_labels)

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
        # ensure data is sorted by x value
        data_xdxydy = data_xdxydy[:, data_xdxydy[0, :].argsort()]
        specs.append(data_xdxydy)

    return Quantity(quantity, "x,dx,y,dy", specs, meta, units)


def read_3D(df, quantity, allowed_labels):
    dfq, meta = select(df, quantity, allowed_labels)

    specs = []
    units = []

    for i, (d, fmt) in enumerate(zip(dfq["data"], dfq["fmt"])):
        entry = dfq.iloc[[i]]
        data = np.array(list(d))
        data_fm = np.zeros((6, data.shape[0]))
        if fmt == "xyz":
            data_fm[0, :] = data[:,0]
            data_fm[2, :] = data[:, 2]
            data_fm[4, :] = data_fm[4, :]
            units.append(
                extract_units(
                    entry,
                    ["units-x", None, "units-y", None, "units-z", None],
                )
            )
        elif fmt == "xyzdz":
            data_fm[0, :] = data[:,0]
            data_fm[2, :] = data[:, 2]
            data_fm[4, :] = data_fm[4, :]
            data_fm[5, :] = data_fm[5, :]
            units.append(
                extract_units(
                    entry,
                    ["units-x", None, "units-y", None, "units-z", "units-z"],
                )
            )
        elif fmt == "xdxyz":
            data_fm[:3, :] = data[:, :3]
            data_fm[4, :] = data[:, 3]
            data_fm[5, :] = data_fm[4, :]
            units.append(
                extract_units(
                    entry,
                    ["units-x", "units-dx", "units-y", None, "units-z", "units-z"],
                )
            )
        elif fmt == "xminxmaxyz":
            data_fm[0, :] = (data[:, 0] + data[:, 1]) * 0.5
            data_fm[1, :] = data[:, 1] - data[:, 0]
            data_fm[2, :] = data[:, 2]
            data_fm[4, :] = data[:, 3]
            data_fm[5, :] = data_fm[4, :]
            units.append(
                extract_units(
                    entry, ["units-xmin", "units-xmin", "units-y", None, "units-z"]
                )
            )
        elif fmt == "xdxydyz":
            data_fm[:5, ...] = data
            data_fm[5, :] = data_fm[4, :]
            units.append(
                extract_units(
                    entry,
                    ["units-x", "units-dx", "units-y", "units-dy", "units-z", None],
                )
            )
        elif fmt == "xydyz":
            data_fm[0, :] = data[:, 0]
            data_fm[2, :] = data[:, 1]
            data_fm[3, :] = data[:, 2]
            data_fm[4, :] = data[:, 3]
            data_fm[5, :] = data_fm[4, :]

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


def read_pfns(df, allowed_labels):
    return [
        read_specs(df, "PFNS", allowed_labels),
        read_specs(df, "PFNS_sqrtE", allowed_labels),
        read_specs(df, "PFNS_max", allowed_labels),
    ]


def print_entries(entries: list, out_fname: str):
    entry = pd.Series(
        index=[i for i in range(len(entries))],
        data=dict(enumerate(entries))
    )
    r = json.loads(entry.to_json(orient="records"))
    with open(Path(out_fname), 'w') as f:
        return json.dump(r, f, indent=1)


def add_entries(fname: str, out_fname: str, entries: list):
    df = pd.read_json(Path(fname))
    index = [i for i in range(len(entries))]
    data = dict(enumerate(entries))
    entry = pd.Series(
        index=index,
        data=data,
    )
    series = pd.concat([df["entries"], entry], ignore_index=True)
    df = pd.DataFrame(data={"entries": series})
    r = json.loads(df.to_json(orient="records"))
    with open(out_fname, 'w') as f:
        json.dump(r, f, indent=1)


def read(fname: str, quantity: str, energy_range=None, allowed_labels=None):
    print("parsing {}".format(fname))
    df = pd.DataFrame.from_records(pd.read_json(fname)["entries"])
    if "Einc" in df:

        # set of entries where Einc is a separate column
        df1 = df[df["Einc"].str.len() >= 1].explode(["Einc", "data"])
        df1["Einc"] = pd.to_numeric(df1["Einc"])

        # set of entries where Einc is included as x variable in data
        df2 = df[df["Einc"].str.len() == 0]

        if energy_range is not None:
            (emin, emax) = energy_range

            # filter first set
            df1 = df1[(df1["Einc"] >= emin) & (df1["Einc"] < emax)]
            #df1["data"] = df1["data"].map(lambda x: x[0])

            # filter second set
            df2["data"] = df2["data"].map(lambda x: x[0])
            df2["data"] = df2["data"][
                df2["data"].apply(
                    lambda x: x == [
                        point for point in x
                        if point[0] >= emin and point[0] < emax
                    ]
                )
            ]
            df2 = df2[df2['data'].notna()]

            return read_json(pd.concat([df1, df2]), quantity, allowed_labels)

        # if no filter required just concat and go
        df2["data"] = df2["data"].map(lambda x: x[0])
        df = pd.concat([df1, df2])
        return read_json(df, quantity, allowed_labels)
    else:
        df["data"] = df["data"].map(lambda x: x[0])
        return read_json(df, quantity, allowed_labels)


def read_nubarTKEA(df, allowed_labels):
    """
    convert all <nu_fragment | A, TKE> to u,du,nu,dnu,TKE_min,TKE_max
    """
    nubarTKEA = read_3D(df, "nubarTKEA", allowed_labels)

    return nubarTKEA


def read_nubartTKEA(df, allowed_labels):
    """
    convert all <nu_total | A, TKE> to u,du,nu,dnu,TKE_min,TKE_max
    """
    nubartTKEA = read_3D(df, "nubartTKEA", allowed_labels)

    return nubartTKEA

def read_nubarATKE(df, allowed_labels):
    """
    convert all <nu_fragment | A, TKE> to u,du,nu,dnu,TKE_min,TKE_max
    """
    nubarATKE = read_3D(df, "nubarATKE", allowed_labels)

    return nubarATKE


def read_nubartATKE(df, allowed_labels):
    """
    convert all <nu_total | A, TKE> to u,du,nu,dnu,TKE_min,TKE_max
    """
    nubartATKE = read_3D(df, "nubartATKE", allowed_labels)

    return nubartATKE


def read_PFNSALAH(df, allowed_labels):
    pfns = read_specs(df, "PFNSALAH", allowed_labels)

    def get_mass_div(comment: str):
        key = "A_L/A_H = "
        idx = comment.find(key) + len(key)
        end_idx = idx + comment[idx:].find(",")
        substrs = [sub.strip("") for sub in  comment[idx:end_idx].split("/") ]
        return int(substrs[0]), int(substrs[1])

    A = [get_mass_div(entry["comments"]) for entry in pfns.meta ]

    return pfns, A


def set_bibtex(quantity, meta):
    path = Path("./" + quantity + "_meta.bib")
    bibtex = []

    for i, row in enumerate(meta):
        search_string = (
            str(row["authors"]) + " " + str(row["title"]) + " " + str(row["year"])
        )
        print("search string:\n" + search_string)
        print("enter bibtex below: ")
        bibtex.append(sys.stdin.readlines())

    with open(path, "w") as f:
        for entry in bibtex:
            for line in entry:
                f.write(line)
            f.write("\n")



def read_json(df: pd.DataFrame, quantity: str, allowed_labels=None, xrange=None):
    q = quantity.replace("HF", "").replace("LF", "")
    if q == "Enbar":
        return read_scalar(df, "Enbar", allowed_labels)
    if q == "nubar":
        return read_scalar(df, "nubar", allowed_labels)
    elif q == "nubarA":
        return read_specs(df, "nubarA", allowed_labels)
    elif q == "nubarZ":
        return read_specs(df, "nubarZ", allowed_labels)
    elif q == "nubarTKE":
        return read_specs(df, "nubarTTKE", allowed_labels)
    elif q == "nugbar":
        return read_scalar(df, "nugbar", allowed_labels)
    elif q == "nugbarA":
        return read_specs(df, "nugbarA", allowed_labels)
    elif q == "nugbarTKE":
        return read_specs(df, "nugbarTTKE", allowed_labels)
    elif q == "nubartotAHF":
        return read_specs(df, "nubartotAHF", allowed_labels)
    elif q == "nubartotALF":
        return read_specs(df, "nubartotALF", allowed_labels)
    elif q == "pfns":
        return read_pfns(df, allowed_labels)
    elif q == "pfns_cm":
        return read_specs(df, "PFNS_cm", allowed_labels)
    elif q == "pfgs":
        return read_specs(df, "PFGS", allowed_labels)
    elif q == "pfgsA":
        return read_specs(df, "PFGSA", allowed_labels)
    elif q == "pnu":
        return read_specs(df, "Pnu", allowed_labels)
    elif q == "pnug":
        return read_specs(df, "Pnug", allowed_labels)
    elif q == "multiplicityRatioA":
        return read_specs(df, "multiplicityRatioA", allowed_labels)
    elif q == "encomA":
        return read_specs(df, "EncomA", allowed_labels)
    elif q == "egtbarA":
        return read_specs(df, "EgTbarA", allowed_labels)
    elif q == "encomTKE":
        return read_specs(df, "EncomTKE", allowed_labels)
    elif q == "egtbarTKE":
        return read_specs(df, "EgTbarTKE", allowed_labels)
    elif q == "egtbarnu":
        return read_specs(df, "EgTbarnubar", allowed_labels)
    elif q == "nubarATKE":
        return read_nubarATKE(df, allowed_labels)
    elif q == "nubartATKE":
        return read_nubartATKE(df, allowed_labels)
    elif q == "nubarTKEA":
        return read_nubarTKEA(df, allowed_labels)
    elif q == "nubartTKEA":
        return read_nubartTKEA(df, allowed_labels)
    elif q == "pfnsA":
        return read_3D(df, "PFNSA", allowed_labels)
    elif q == "encomATKE":
        return read_3D(df, "encomATKE", allowed_labels)
    elif q == "nugnuA":
        return read_3D(df, "nugnu", allowed_labels)
    elif q == "PFNSALAH":
        return read_PFNSALAH(df, allowed_labels)
    else:
        raise ValueError("Unknown quantity: " + quantity)
