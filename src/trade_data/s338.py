'''
s338 final list and indicators 
'''
import pandas as pd
from pathlib import Path

#
prgtype = {"PART_A_ADDED_ALCOHOL": "alcohol", "PART_A_ADDED": "motor_vehicles"}


def updated_list(jay, upd):
    '''
    s338 yale (jay) list updated with the Sep . 2026 proclamations
    PART_B codes also re-added in PART_A were 10-digit changes
    '''
    add = upd[(upd["action"] == "added")]
    add = add.drop_duplicates("hts8")
    # Update removed list by checking added one: 
    # Some codes in the removed list at 10 digits were removed but other 10 digits variants valid at
    # HS8 were added, so we have to keep them
    rmv = set(upd.loc[upd["action"].eq("removed"), "hts8"]) - set(add["hts8"])

    base = jay[~jay["hts8"].isin(rmv)].copy()
    base["status"] = "July list" # Tracker 

    new = add[~add["hts8"].isin(base["hts8"])].copy()
    #new["program"] = new["list"].map(lambda k: prgtype[k][0])
    new["program"] = new["list"].map(prgtype)
    #new["ch99_heading"] = new["list"].map(lambda k: prgtype[k][1])
    new["status"] = "sep. 15"

    # Add the base list to the new one from sept 2026
    out = pd.concat([base, new[["hts8", "program", "status"]]],
                     ignore_index=True)

    banned_filter = upd["action"].eq("import ban")
    out["import_ban"] = out["hts8"].isin(upd.loc[banned_filter, "hts8"])

    return out


