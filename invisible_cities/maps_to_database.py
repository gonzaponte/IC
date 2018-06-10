import sys
import argparse

from functools import partial

import pandas as pd
import tables as tb

from invisible_cities.io.dst_io import load_dst


def row_builder(separator, run_number, min_time, mean_time, max_time, x, y, value, uncertainty):
    values = map(str, (run_number, min_time, max_time, mean_time, x, y, value, uncertainty))
    return separator.join(values)


def write_map(variable, filename, separator, dst, run_number, t_min, t_max, t_mean):
    build_row = partial(row_builder, separator, run_number, t_min, t_mean, t_max)
    rows = map(build_row, dst.x.values, dst.y.values, dst.factor.values, dst.uncertainty.values)
    with open(filename, "w") as file:
        columns = "RunNumber MinTime MaxTime MeanTime X Y Lifetime Uncertainty".split()
        file.write(separator.join(columns) + "\n")
        file.write("\n".join(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(                "input_file", type=str, action="store")
    parser.add_argument("-olt" ,  "--lt_output_file", type=str, action="store", default= "lt_map.txt")
    parser.add_argument("-ogeo", "--geo_output_file", type=str, action="store", default="geo_map.txt")
    parser.add_argument("-sep" ,       "--separator", type=str, action="store", default=", "         )

    args = parser.parse_args()
    with tb.open_file(args.input_file) as file:
        run_number = file.root.RunInfo[0][0]
        t_min      = file.root.RunInfo[0][1]
        t_max      = file.root.RunInfo[0][2]
        t_mean     = int(t_min + t_max) // 2

        pitch = file.root.LTMapInfo[0][0]
        x_min = file.root.LTMapInfo[0][1]
        x_max = file.root.LTMapInfo[0][2]
        y_min = file.root.LTMapInfo[0][3]
        y_max = file.root.LTMapInfo[0][4]

        table = file.root.XYcorrections.Elifetime.read()
        dst   = pd.DataFrame.from_records(table)
        write_map("Lifetime", args.lt_output_file, args.separator, dst, run_number, t_min, t_max, t_mean)

        pitch = file.root.GEOMapInfo[0][0]
        x_min = file.root.GEOMapInfo[0][1]
        x_max = file.root.GEOMapInfo[0][2]
        y_min = file.root.GEOMapInfo[0][3]
        y_max = file.root.GEOMapInfo[0][4]

        table = file.root.XYcorrections.Egeometry.read()
        dst   = pd.DataFrame.from_records(table)
        write_map("Energy", args.geo_output_file, args.separator, dst, run_number, t_min, t_max, t_mean)
