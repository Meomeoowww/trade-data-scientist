'''
This file contains functions for data collection and ingesion from:
CIMT
CEPII
WB
BUGETLAB_YALE
'''

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import io

# Namibia code is missing in CEPII, and no TAIWAN
CEPII_ISO2_FIXES = {516: "NA", 490: "TW"}
CEPII_links = {
    "baci":("https://www.cepii.fr/DATA_DOWNLOAD/baci/data/BACI_HS22_V202601.zip",["BACI_HS22_Y2024_V202601.csv", "product_codes_HS22_V202601.csv", "country_codes_V202601.csv"]),
    "gravity":("https://www.cepii.fr/DATA_DOWNLOAD/gravity/data/Gravity_csv_V202211.zip",["Gravity_V202211.csv"]),
}

###
''' Set working directory for datafiles'''
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Dictionaries
ctry = "MEX;USA;CAN;KOR;GBR;CHN;JPN"

indictionary_wdi = {"NY.GDP.MKTP.CD":"gdp_usd", 
        "NY.GDP.PCAP.CD":"gdp_per_cap_usd",
        "SP.POP.TOTL":"pop",
        "NE.TRD.GNFS.ZS":"trade_op_pct_gdp",
#        "CC.PER.RNK": "corruption_percent",
#        "RL.PER.RNK": "rule_of_law_percent",
    }
indictionary_wgi = {
                    #"GOV_WGI_CC.EST": "corruption_est", 
                    "GOV_WGI_RL.EST": "rule_of_law_est",
                    "GOV_WGI_RQ.EST": "regulatory_quality_est"
    }

ind = [
    "distw_harmonic",
    "distw_arithmetic",
    "dist",
    "distcap",    
    "contig",
    "comlang_off",
    "comlang_ethno",
    "comcol",
    "col45",
    "fta_wto",
]

#################### CIMT
def load_cimt_data():
    """
    Loads CIMT from data/cimt folder, and return a dataframe
    """
    folderpath = DATA_DIR/ "cimt"
    
    files = Path(folderpath).glob("*12N.csv")
    data = pd.concat((pd.read_csv(file, 
                                    dtype={'HS8': str},
                                    keep_default_na=False # When working with broader datasets, we have NA as country code for Namibia
                                    ) for file in files), 
                        ignore_index=True,
                        
                        )
    return data

def load_HS_dictionary():
    """
    Loads the HS code dictionary from the data folder and returns a DataFrame.
    """
    filepath = DATA_DIR/ "worldbank" / "hs2022_comtrade.csv"

    df = pd.read_csv(filepath, dtype={"ProductCode": str})
    return df


def load_country_cimt_dictionary():
    """
    Loads the country CIMT dictionary from the data folder and returns a DataFrame.
    """
    filepath = DATA_DIR/ "cimt" / "ODPF_6_CtyDesc.TXT"

    df = pd.read_fwf(filepath,
                        header=None,
                        names=[
                            "iso2", "isonum3", "start", "end",
                            "country_en", "country_fr", "source", "updated",
                        ],
                        encoding="cp1252",
                        keep_default_na=False,
                        )
    df = df.sort_values("end").drop_duplicates("iso2", keep="last") # Keep only last entries to avoid duplicates
    return df

######################################################################## CEPII


def download_cepii(name):
    '''Downloads a CEPII archive ("baci" or "gravity") and extracts the files used by the project into data/cepii.'''
    url, filz = CEPII_links[name]
    folder = DATA_DIR / "cepii"
    
    missing = [] # Init list

    # add filename if it wasnt in the list
    for f in filz:
        if not (folder / f).exists():
            missing.append(f)
            
    if missing:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(folder, members=missing)

def load_baci_cepii_data():
    """
    Loads the BACI CEPII data from the data folder and returns a DataFrame.
    """

    download_cepii("baci")

    pathfolder = DATA_DIR/"cepii"
    files = list(pathfolder.glob("BACI_*.csv"))

    if not files:
        print("No BACI CEPII data files found in the specified folder.")
        return None
    else:
        filepath = files[0]

    df = pd.read_csv(filepath, dtype={'k':str})
    return df

def load_gravity_cepii_data(ctry=None, ind=None, year=None):
    """
    Loads the CEPII Gravity data, with optional filters on countries, indicators and year.
    """

    download_cepii("gravity")

    pathfolder = DATA_DIR/ "cepii"
    files = list(pathfolder.glob("Gravity_*.csv"))

    if not files:
        print("No Gravity CEPII data files found in the specified folder.")
        return None

    filepath = files[0]

    columns = None
    if ind is not None:
        columns = ["year", "iso3_o", "iso3_d"] + ind
    df = pd.read_csv(
        filepath,
        usecols=columns,
        dtype={"iso3_o": str, "iso3_d": str},
    )

    if ctry is not None:
        df = df[
            df["iso3_o"].isin(ctry) # Trick to avoid loading the whole db
            & df["iso3_d"].isin(ctry) # Trick to avoid loading the whole db
        ]

    if year is not None:
        df = df[df["year"] == year]
        df = df.drop("year", axis=1)    
    return df.reset_index(drop=True)

    
def load_cepii_country_dictionary():
    """
    Loads the CEPII country dictionary from the data folder and returns a DataFrame.
    Fills the ISO2 codes left empty in the source for Namibia and Taiwan (Other Asia, nes).
    """
    pathfolder = DATA_DIR / "cepii"
    files = list(pathfolder.glob("country*"))
    if not files:
        print("No CEPII country dictionary files found in the specified folder.")
        return None
    filepath = files[0]

    df = pd.read_csv(filepath, keep_default_na=False)
    for code, iso2 in CEPII_ISO2_FIXES.items():
        df.loc[df["country_code"].eq(code), "country_iso2"] = iso2
    return df

def load_cepii_product_dictionary():
    """
    Loads the CEPII product dictionary from the data folder and returns a DataFrame.
    """
    pathfolder = DATA_DIR/ "cepii"
    files = list(pathfolder.glob("product*"))
    if not files:
        print("No CEPII product dictionary files found in the specified folder.")
        return None
    else:
        filepath = files[0]

    df = pd.read_csv(filepath, dtype={'code':str})
    return df

###################################################### WorldBank Indicators Ingestion
def wb_indicator(indicator, name, countries=ctry, period="2024:2025", source=None):
    '''
    One function to define standard structure of wb indicators
    '''
    params = {"format": "json", "date": str(period), "per_page": 20000}
    if source:
        params["source"] = source
    r = requests.get(f"https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}",
                     params=params, timeout=120)
    r.raise_for_status()
    payload = r.json()
    rows = payload[1] if len(payload) > 1 and payload[1] else []
    res = pd.DataFrame([{"year": int(row["date"]), "country": row["countryiso3code"], name: row["value"]}
                        for row in rows if row["countryiso3code"]],
                    columns=["year", "country", name])
    return res

def wdi_one_ind(ind, ctry, period):
    res = wb_indicator(ind, indictionary_wdi[ind], countries=ctry, period=period)
    return res

def wgi_one_ind(ind, ctry, period):
    res = wb_indicator(ind, indictionary_wgi[ind], countries=ctry, period=period, source=3)
    return res

#### Import all WB indicators
def load_wdi_full_ind(period, countries=ctry):
    ''' Full list of selected indicators'''

    codes = list(indictionary_wdi)
    data = wdi_one_ind(codes[0], countries, period)

    for code in codes[1:]:
        data = data.merge(
            wdi_one_ind(code, countries, period),
            on=["year", "country"],
            how="outer",
        )
    return data

def load_wgi_full_ind(period, countries=ctry):
    ''' Full list of selected indicators'''

    codes = list(indictionary_wgi)

    data = wgi_one_ind(codes[0], countries, period)

    for code in codes[1:]:
        data = data.merge(
            wgi_one_ind(code, countries, period),
            on=["year", "country"],
            how="outer",
        )
    return data

############ ITC MARKET MAP
def load_itc_data():
    """
    Loads Tariff from data/ITC folder, and return a dataframe
    """
    folderpath = DATA_DIR/ "ITC"
    
    files = Path(folderpath).glob("*.xlsx")
    data = pd.concat((pd.read_excel(file,
                                    sheet_name="Data",
                                    dtype={'ProductCode': str},
                                    #keep_default_na=False # When working with broader datasets, we have NA as country code for Namibia
                                    ) for file in files), 
                        ignore_index=True,
                        
                        )
    return data

############### StatCan


def load_statcan_tables():
    '''
    Load statcan csv tables. Return a dict
    '''
    folderpath = DATA_DIR/ "statcan"
    
    pop_prov_f = Path(folderpath/"1710000901_databaseLoadingData.csv")
    gdp_f = Path(folderpath/"3610022201_databaseLoadingData.csv") #"Gross domestic product at market prices"

    pop_prov = pd.read_csv(pop_prov_f)
    gdp_prov = pd.read_csv(gdp_f)

    return {"province_pop": pop_prov, "province_gdp":gdp_prov}

############### BdB Rate

###################### SECTION 338

def load_usitc_hts8():
    '''
    Download and prepare usitc
    '''

    pathfolder = DATA_DIR / "section338"
    filepath = pathfolder /"usitc_hts8.json"

    link = 'https://hts.usitc.gov/reststop/exportList?from=0101&to=9999&format=JSON&styles=false'

    r = requests.get(link, timeout=30)

    r.raise_for_status()

    # Save as json file in cas of ...
    #with filepath.open("w", encoding="utf-8") as f:
    #    json.dump(r.json(), f, indent=4, ensure_ascii=False)

    data = pd.DataFrame(r.json()) # formats as dataframe
    htsno = data["htsno"].astype(str).str.replace(".", "", regex=False)
    hts8 = htsno[htsno.str.fullmatch(r"\d{8}(\d{2})?")].str[:8].drop_duplicates()
    dataf =  pd.DataFrame({"hts8": hts8.to_numpy(), "hs6": hts8.str[:6].to_numpy()})
    return dataf

def load_yale_s338():
    '''Jul. 2026 S338 list (BudgetLab)'''
    filepath = DATA_DIR / "section338" / "s338_products.csv"
    
    if not filepath.exists():
        r = requests.get("https://raw.githubusercontent.com/Budget-Lab-Yale/tariff-rate-tracker/master/resources/s338_products.csv", timeout=60)
        r.raise_for_status()
        filepath.write_bytes(r.content)
    
    de = pd.read_csv(filepath, dtype=str)
    
    return de


def load_updates_proc_s338():
    '''
    load updates of Section 338 made in september 2026 from an xls file
    '''

    lists = pd.read_excel(DATA_DIR / "section338" / "sept_15_2026_proclamations.xls", sheet_name=None, dtype=str)
    d =  pd.concat(lists, names=["list", None])
    d = d.reset_index(level="list").reset_index(drop=True)

    return d
