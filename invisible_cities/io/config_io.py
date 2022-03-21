import pandas as pd
import tables as tb

from .. core.core_functions import str_df_from_dict
from .. io  .dst_io         import df_writer


def write_config(filename, variables, table):
    config_df   = str_df_from_dict(variables)
    longest_str = config_df.value.str.len().max() + 1

    with tb.open_file(filename, "a") as file:
        df_writer(file, config_df, "Config", table, str_col_length=longest_str)


def copy_config(filename_in, filename_out):
    with tb.open_file(filename_in) as file_in:
        if "Config" not in file_in.root: return

        with tb.open_file(filename_out, "a") as file_out:
            for table in file_in.walk_nodes("/Config/", "Table"):
                if not "Config" in file_out.root:
                    file_out.create_group(file_out.root, "Config")

                file_in.copy_node(f"/Config/{table.name}", file_out.root.Config)
