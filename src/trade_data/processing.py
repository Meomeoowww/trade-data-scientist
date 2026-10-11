'''
Processing of trade data for analysis and modeling. This module includes functions to transform and clean the CIMT dataframe, specifically extracting Year and Month from the 'YearMonth/AnnéeMois' column.
Sources used for data ingestion and processing include:
CIMT
CEPII
WB
BUGETLAB_YALE

'''

import pandas as pd
import numpy as np
from pathlib import Path

helper_province = {
    "CA": "Canada",
    "NL": "Newfoundland and Labrador",
    "PE": "Prince Edward Island",
    "NS": "Nova Scotia",
    "NB": "New Brunswick",
    "QC": "Quebec",
    "ON": "Ontario",
    "MB": "Manitoba",
    "SK": "Saskatchewan",
    "AB": "Alberta",
    "BC": "British Columbia",
    "YT": "Yukon",
    "NT": "Northwest Territories",
    "NU": "Nunavut",
}

PANEL_ISO2 = ['US', 'CA', 'MX', 'CN', 'KR', 'JP', 'GB']
PANEL_ISO3 = ["USA", "CAN", "MEX", "CHN", "KOR", "JPN", "GBR"]

# Issues with ISO 3 Namibia codes 

MY_MACRO_SELECTION = ["gdp_usd", "gdp_per_cap_usd", "pop", "trade_op_pct_gdp", "rule_of_law_est", "regulatory_quality_est"]
AB_GDP_GROWTH = {2025: 0.018}   # From 2026 fiscal plan growth as 2025 gdp is not yet available at statcan

# CEPII fta_wto stops in 2021 and misses CPTPP (in force 2018) and RCEP (in force 2022)
CPTPP = {"AUS", "BRN", "CAN", "CHL", "JPN", "MYS", "MEX", "NZL", "PER", "SGP", "VNM"}
RCEP = {"AUS", "BRN", "KHM", "CHN", "IDN", "JPN", "KOR", "LAO", "MYS", "MMR", "NZL", "PHL", "SGP", "THA", "VNM"}

def cimt_processing(data):
    """
    Transforms CIMT dataframe by extracting Year and Month from 'YearMonth/AnnéeMois' column
    Returns transformed df
    """
 
    data['Year'] = data['YearMonth/AnnéeMois'].astype(str).str[0:4]
    data['Month'] = data['YearMonth/AnnéeMois'].astype(str).str[4:6]
    data.drop(columns=['YearMonth/AnnéeMois'], inplace=True)
    data['HS4'] = data['HS8'].astype(str).str[0:4]
    return data
 

def table_baci_processing(data, cepii_ctr, countries=PANEL_ISO2):
    '''
    process baci and returns processed data
    '''

    # Mapping for all countries ISO3
    country_map = cepii_ctr.drop_duplicates('country_code').set_index("country_code")["country_iso3"]

    # optional filter on exporters and importers
    if countries is not None:
        codes = cepii_ctr.loc[cepii_ctr['country_iso2'].isin(countries), 'country_code']
        data = data[data['i'].isin(codes) & data['j'].isin(codes)]

    filtered_data = data.copy()
    filtered_data["c_o"] = filtered_data["i"].map(country_map)
    filtered_data["c_d"] = filtered_data["j"].map(country_map)

    filtered_data['HS4'] = filtered_data['k'].str[0:4]
    filtered_data = filtered_data[['t', 'c_o', 'c_d', 'HS4', 'v']].groupby(['t', 'c_o', 'c_d', 'HS4']).sum().reset_index()
    filtered_data.columns = ['year', 'c_o', 'c_d', 'hs4', 'value']
    return filtered_data


def tariff_processing(data):
    '''
    Proc tariff by adding iso3
    '''
    mapping = {'Canada':'CAN',
            'China':"CHN",
            'Japan': "JPN",
            'Mexico':"MEX",
            'Korea, Republic of':"KOR",
            'United Kingdom':"GBR",
            'United States of America':'USA',
    } 

    data['ReportingCountry'] = data['ReportingCountry'].map(mapping)
    data['PartnerCountry'] = data['PartnerCountry'].map(mapping)
    return data

def statcan_province_pop_processing(datadict, y1, y2, province):
    '''
    As input the dictionary of statcan datasets and returns a dataset filtered and processed
    '''
    province_pop = datadict['province_pop'] 

    # Filter with only july estimates of the population
    province_pop = province_pop[
        ((province_pop["REF_DATE"].str.startswith(str(y1))
        & province_pop["REF_DATE"].str.endswith("07-01")) |
        (province_pop["REF_DATE"].str.startswith(str(y2))
        & province_pop["REF_DATE"].str.endswith("07-01"))
        ) &
        (province_pop["GEO"]==helper_province[province]) 

        ][["REF_DATE","GEO","VALUE"]]
    province_pop["year"] = province_pop["REF_DATE"].str[0:4]
    province_pop.drop("REF_DATE", axis=1, inplace=True)
    return province_pop

def statcan_province_gdp_processing(datadict, y1, y2, province):

    '''
    Process gdp from ststcan database
    '''
    province_gdp = datadict['province_gdp']

    province_gdp
    # Filter with only july estimates of the population
    province_gdp = province_gdp[
        ((province_gdp["REF_DATE"]==y1) 
        | (province_gdp["REF_DATE"]==y2)
        ) &
        (province_gdp["Estimates"]=="Gross domestic product at market prices") &
        (province_gdp["GEO"]==helper_province[province]) 

        ][["REF_DATE","GEO","Estimates","VALUE"]]
    province_gdp = province_gdp.rename(columns={"REF_DATE":"year"})
    return province_gdp

### Map between iso2 and ISO3 countries

def map_iso2_iso3(cepii_ctr):
    '''
    Map iso2 to iso3 from cepii country file
    '''
    #remove all duplicates and filter '' lines
    country_map = cepii_ctr[cepii_ctr["country_iso2"].ne("")].drop_duplicates("country_iso2")
    # define iso2 as index, keep iso3 as correspondinc elt and convert to dict 
    country_map = country_map.set_index("country_iso2")["country_iso3"].to_dict()

    return country_map

def table_province_hs4_processing(cimt, province, year, rate, isomap):
    '''
    Provincial exports in thousands usd, by destination (ISO3 as cepii has iso3)
    ''' 


    c = cimt[(cimt["Province"] == province) & (cimt["Year"] == str(year))]
    c = c.assign(d=c["Country/Pays"].map(isomap), v=c["Value/Valeur"] / 1000 / rate[year])
    c = c.groupby(["d", "HS4"], as_index=False)["v"].sum().rename(columns={"HS4": "hs4"})
    return c



def table_macro_processing(wdi, wgi, ab_gdp, ab_pop, fx, years=(2024, 2025)):
    '''
    Define a complete table with all macro variables
    '''
    #1- Merge WDI with WGI
    w_merge = wdi.merge(wgi, on=["year", "country"], how="outer")

    # Define new index combining countries, year. It gives us a complete grid country X year
    idx = pd.MultiIndex.from_product([w_merge["country"].unique(), 
                                      list(years)], 
                                      names=["country", "year"])

    # Defines country and year as index, pass the complete index table to our merged WDI WGI 
    # This introduces NaN as some combinations are probably missing
    # By country, it fills missing elements by latest value
    w_merge = w_merge.set_index(["country", "year"]).reindex(idx).groupby(level="country").ffill().reset_index()

    # Pick specific CANADA CASE as it will be used as proxy at Province level
    can = w_merge[w_merge["country"].eq("CAN")].set_index("year")
    gdp_cad = {2024: ab_gdp * 1e6} # province (in our case Alberta's )gdp 2024 converted in millions cad
    gdp_cad[2025] = gdp_cad[2024] * (1 + AB_GDP_GROWTH[2025]) # estimated gdp 2025 using published finance's growth rate
    
    # Now, we're going to build a second dataset only for Alberta
    rows = []
    for y in years:
        gdp_usd = gdp_cad[y] / fx[y] # gdp in usd as wdi wgi macro data is in usd

        # Then create the listdic pf all of the variables for AB, Alberta
        rows.append({"country": "AB", "year": y, "gdp_usd": gdp_usd, "pop": ab_pop[y],
                     "gdp_per_cap_usd": gdp_usd / ab_pop[y],
                     # Proxies as defined in the assignment here 
                     # for alberta are at the country level
                     "trade_op_pct_gdp": can.loc[y, "trade_op_pct_gdp"],
                     "rule_of_law_est": can.loc[y, "rule_of_law_est"],
                     "regulatory_quality_est": can.loc[y, "regulatory_quality_est"]})

    # As w_merge has all of the countries macro, and now row has alberta's macro, we can merge them
    final = pd.concat([w_merge, pd.DataFrame(rows)], ignore_index=True)
        
    return final


def gravity_processing(gravity, exporters=PANEL_ISO3):
    '''
    process all gravity cepii variables from cepii_gravity by filtering based on exporters list
    removing any duplicates as for distance, only one row pair is enough
    renaming the colums with our conventions
    '''
    grav = gravity[gravity["iso3_o"].isin(exporters)]
    # only
    grav = grav.drop_duplicates(["iso3_o", "iso3_d"])
    grav = grav.rename(columns={"iso3_o": "o_ctry", "iso3_d": "d"})
    grav= grav.reset_index(drop=True)
    return grav