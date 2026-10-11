'''
This file contains function  for analysis

'''

import pandas as pd
import numpy as np
from trade_data.processing import helper_province

US = ["US","PR"] # US and PR as PR 'Porto Rico is part of the U.S. Customs Union
MAJOR_PROVINCES = ["ON","QC","AB","BC","SK","MB"]
MESURES = ["exposure","exposure_full_hs6","exposure_weighted","exposure_import_ban"]


## Quick filter province_year

def data_province_year(data, province, year):
    '''
    Filters dataframs (CIMT) by proivnce and year
    '''
    d = data[(data['Year'] == str(year)) & (data['Province'] == province)]
    return d

def total_exports_by_province(data, province, year):
    ''' Returns the total exports for a province in a given year based on the 'Value/Valeur' column.'''
 
    data_year = data_province_year(data, province, year)
    total = data_year['Value/Valeur'].sum()
    return total
 

def top10_prod_ab(data, year, province='AB'):
    """
    Returns the top 10 HS4 products for a given year in Alberta (AB) based on the 'Value/Valeur' column.
    """
 
    #data_year = data[(data['Year'] == str(year)) & (data['Province'] == 'AB')]
    data_year = data_province_year(data, province, year)
    top10 = data_year[['HS4', 'Value/Valeur']].groupby('HS4')['Value/Valeur'].sum().nlargest(10).reset_index()
    top10['Value/Valeur'] = (top10['Value/Valeur'] / 10**9)#.round(2)  # Convert to billions
    top10.rename(columns={'Value/Valeur': 'Value (Billion CAD)'}, inplace=True)
    top10['Share (%)'] = (100 * top10['Value (Billion CAD)'] / (total_exports_by_province(data, province, year) / 10**9)).round(2)
    top10['Value (Billion CAD)'] = top10['Value (Billion CAD)'].round(2)
    return top10
 

def top10_country_ab(data, year,province='AB'):
    """
    Returns the top 10 countries for a given year in Alberta (AB) based on the 'Value/Valeur' column.
    """

    data_year = data_province_year(data, province, year)
    top10 = data_year[['Country/Pays', 'Value/Valeur']].groupby('Country/Pays')['Value/Valeur'].sum().nlargest(10).reset_index()
    top10['Value/Valeur'] = (top10['Value/Valeur'] / 10**9)#.round(2)  # Convert to billions
    top10.rename(columns={'Value/Valeur': 'Value (Billion CAD)'}, inplace=True)
    top10['Share (%)'] = (100 *top10['Value (Billion CAD)'] / (total_exports_by_province(data, province, year) / 10**9)).round(2)
    top10['Value (Billion CAD)'] = top10['Value (Billion CAD)'].round(2)
    return top10
 
def top5_prod_ctry(data, year, country):
    """
    Returns the top 5 HS4 products for a given year and country based on the 'Value/Valeur' column.
    """
  
    data_year_country = data[(data['Year'] == str(year)) & (data['Country/Pays'] == country)]
    top5 = data_year_country[['HS4', 'Value/Valeur']].groupby('HS4')['Value/Valeur'].sum().nlargest(5).reset_index()
    top5['Value/Valeur'] = (top5['Value/Valeur'] / 10**9)#.round(2)  # Convert to billions
    top5.rename(columns={'Value/Valeur': 'Value (Billion CAD)'}, inplace=True)
    return top5


def top5_prod_by_top10_ctry(
        cimt,
        country_dict,
        hs_dict,
        year=2025,
        province="AB",
        n_countries=10,
        n_products=5,
):
    """
    Returns the top 5 HS4 products for each of the top 10 countries for a given year and province based on the 'Value/Valeur' column.
    """

    #filter the data for the specified province and year
    c2025_ab = cimt[(cimt['Province']==province) & (cimt['Year']==str(year))]
    c2025_ab = c2025_ab[['Country/Pays','HS4','Value/Valeur']]
    c2025_ab = c2025_ab.groupby(['Country/Pays','HS4']).sum().reset_index()

    B = c2025_ab[['Country/Pays','Value/Valeur']].groupby(['Country/Pays']).sum().reset_index().sort_values(by='Value/Valeur', ascending=False).head(n_countries)
    B = B.merge(c2025_ab, on='Country/Pays', how='left', suffixes=('', '_per_HS4'))
    B['Share (%) by partner'] = ((B['Value/Valeur_per_HS4'] / B['Value/Valeur']) * 100).round(2)

    B['Rank'] = B.groupby(['Country/Pays'])[('Value/Valeur_per_HS4')].rank(method ='first', ascending=False) #dense method could give the same rank multiple times then excel the 5

    B_final = B[B['Rank'] <= n_products].sort_values(['Value/Valeur','Value/Valeur_per_HS4'], ascending=[False,False])
    B_final = B_final.join(country_dict.set_index('iso2'), on='Country/Pays', how='left').join(hs_dict.set_index('ProductCode'), on='HS4', how='left')[['country_en','HS4','Product Description','Value/Valeur_per_HS4','Share (%) by partner']]
    B_final = B_final.rename(columns={'country_en':'Country','HS4':'HS4 Code','Product Description':'Product Description','Value/Valeur_per_HS4':'Value (Billion CAD)','Share (%) by partner':'Share (%) by partner'})
    B_final['Value (Billion CAD)'] = B_final['Value (Billion CAD)'].apply(lambda x: round(x/1e9, 2))

    return B_final



def trade_indices(cimt, province='AB',year1=2024, year2=2025):
    '''
    Returns a dictionary of trade indices components: UVI, Value Index, Volume Index, and Coverage Rate.
    '''

    cols = ['Value/Valeur', 'Quantity/Quantité']
    cimty1_ab = data_province_year(cimt, province, year1).groupby('HS8')[cols].sum().reset_index()
    cimty2_ab = data_province_year(cimt, province, year2).groupby('HS8')[cols].sum().reset_index()
    merged_cimt = pd.merge(cimty1_ab, cimty2_ab, on=['HS8'], suffixes=('_' + str(year1), '_' + str(year2)), how='inner')
    merged_cimt['Value Change (%)'] = (100 * (merged_cimt['Value/Valeur_' + str(year2)] - merged_cimt['Value/Valeur_' + str(year1)]) / merged_cimt['Value/Valeur_' + str(year1)])#.round(2)
    merged_cimt['Quantity Change (%)'] = (100 * (merged_cimt['Quantity/Quantité_' + str(year2)] - merged_cimt['Quantity/Quantité_' + str(year1)]) / merged_cimt['Quantity/Quantité_' + str(year1)])#.round(2)
    merged_cimt['Elementary UV(%)'] = ((merged_cimt['Value/Valeur_' + str(year2)] / merged_cimt['Quantity/Quantité_' + str(year2)]) / (merged_cimt['Value/Valeur_' + str(year1)] / merged_cimt['Quantity/Quantité_' + str(year1)]) * 100)#.round(2)

    # We have to remove the rows where the Elementary UV(%) is NaN or infinite, as they are not valid for analysis. This can happen if the quantity in 2024 or 2025 is zero, leading to division by zero.
    merged_cimt = merged_cimt.loc[np.isfinite(merged_cimt['Elementary UV(%)']), ["HS8", "Value/Valeur_" + str(year1), "Quantity/Quantité_" + str(year1),"Value/Valeur_" + str(year2), "Quantity/Quantité_" + str(year2), 'Elementary UV(%)']]

    UV_Indice_y2_y1 = (merged_cimt['Value/Valeur_' + str(year1)] * merged_cimt['Elementary UV(%)']).sum()/sum(merged_cimt['Value/Valeur_' + str(year1)])

    Value_Indice_y2_y1 = 100 *(merged_cimt['Value/Valeur_' + str(year2)].sum()/merged_cimt['Value/Valeur_' + str(year1)].sum())

    Volume_Indice_y2_y1 = Value_Indice_y2_y1/UV_Indice_y2_y1 * 100

    Coverage_rate_2024 = merged_cimt['Value/Valeur_' + str(year1)].sum()/cimty1_ab['Value/Valeur'].sum() * 100
    Coverage_rate_2025 = merged_cimt['Value/Valeur_' + str(year2)].sum()/cimty2_ab['Value/Valeur'].sum() * 100

    string = f"Global UVI for Alberta in 2025 compared to 2024 is {UV_Indice_y2_y1:.2f}, Value Index is {Value_Indice_y2_y1:.2f}, Volume Index is {Volume_Indice_y2_y1:.2f}, sample coverage rate is {Coverage_rate_2024:.2f}% for 2024 and {Coverage_rate_2025:.2f}% for 2025."

    return {
        "UV_Indice_y2_y1": UV_Indice_y2_y1,
        "Value_Indice_y2_y1": Value_Indice_y2_y1,
        "Volume_Indice_y2_y1": Volume_Indice_y2_y1,
        "Coverage_rate_y1": Coverage_rate_2024,
        "Coverage_rate_y2": Coverage_rate_2025,
    }
 
################################# Complementary tables for Q2 and indicators

def hs6_coverage(s338, hts8_schedule):
    '''
    From usitc to hs6 touched by the proclamations: provides hs6 lines covered py s338
    # listed_lines: nb. hts8 lines included in a specific hs6 (in usitc)
    # us_lines: notal nb of hts8 lines in a specific hs6 (usitc)
    # line_share = listed_lines/ us_lines
    # import_ban: True if at least 1 hts8 line is ban
    # program : annex of lines toched in a hs6
    '''
    # exraire hs6 
    listed = s338.copy()
    listed["hs6"] = listed["hts8"].str[:6]

    cov = listed.groupby("hs6").agg(listed_lines=("hts8", "nunique"),
                                    program=("program", lambda s: "/".join(sorted(set(s)))),
                                    import_ban=("import_ban", "any"))

    #All lines
    alln = hts8_schedule.groupby("hs6")["hts8"].nunique().reindex(cov.index).fillna(0)
    
    cov["us_lines"] = alln.clip(lower=cov["listed_lines"])
    cov["line_share"] = cov["listed_lines"] / cov["us_lines"]
      
    return cov


def tot_can_exp_hs6(cimt, year):
    '''
    Exports by province, hs6, hs4, between the us and other dests
    '''
    c = cimt[cimt["Year"] == str(year)].copy()
    c["hs6"] = c["HS8"].astype(str).str[:6]
    c["to_us"] = c["Country/Pays"].isin(US)
    c= c.groupby(["Province", "hs6", "HS4", "to_us"], as_index=False)["Value/Valeur"].sum()
    c = c.rename(columns={"HS4": "hs4"})

    return c


def province_gdp(datadict, year):
    '''
    Nominal GDP (CAD) by province and territory code
    '''
    code = {name: c for c, name in helper_province.items() if c != "CA"}
    g = datadict["province_gdp"]
    g = g[(g["REF_DATE"] == year) & g["GEO"].isin(code)
          & (g["Prices"] == "Current prices") & (g["Estimates"] == "Gross domestic product at market prices")]
    g = g.set_index(g["GEO"].map(code))["VALUE"] * 1e6

    return g


###
############################### Compute exposure
def xposr_trade(exports, coverage):
    '''
    HS6 exports to the US by exposed lines
    ##exposure (whole HS6),
    ##exposure_full_hs6 (only HS6 whose U.S. lines are all listed),
    ##exposure_weighted (x share of the HS6's U.S. lines listed),
    ##exposure_import_ban (lines banned sep 26).
    '''
    us = exports[exports["to_us"]].merge(coverage[["line_share", "import_ban", "program"]],
                                         left_on="hs6", right_index=True, how="left")
    us["exposed"] = us["line_share"].notna()
    us["exposure"] = us["Value/Valeur"].where(us["exposed"], 0)
    us["exposure_full_hs6"] = us["exposure"].where((us["line_share"] == 1), 0)
    us["exposure_weighted"] = us["exposure"] * us["line_share"].fillna(0)
    us["exposure_import_ban"] = us["exposure"].where(us["import_ban"].fillna(False).astype(bool), 0)
    
    return us


def xposr_prov(exports, us, gdp):
    '''
    exposure by province and for canada
    in cad
    '''
    p = us.groupby("Province")[MESURES].sum()
    p["exports_to_us"] = us.groupby("Province")["Value/Valeur"].sum()
    p["total_exports"] = exports.groupby("Province")["Value/Valeur"].sum()
    p.loc["Canada"] = p.sum()
    p["gdp"] = gdp.reindex(p.index)
    p.loc["Canada", "gdp"] = gdp.sum()
    p["share_of_canada_exposure_%"] = 100 * p["exposure"] / p.loc["Canada", "exposure"]
    p["intensity_total_exports_%"] = 100 * p["exposure"] / p["total_exports"]
    p["intensity_us_exports_%"] = 100 * p["exposure"] / p["exports_to_us"]
    p["intensity_gdp_%"] = 100 * p["exposure"] / p["gdp"]
    k =  p.sort_values("exposure", ascending=False)

    return k


def xposr_prog(us):
    '''
    exposure by province and section program (alcohol, dairy, vehicles)
    '''
    p = us[us["exposed"]].pivot_table(index="Province", columns="program", values="exposure",
                                      aggfunc="sum", fill_value=0)
    p.loc["Canada"] = p.sum()
    return p


def xposr_hs4(exports, us, province):
    '''
    A prov exposure by HS4
    with exposure intensity = xpsr / province's total exports of this HS4
    all destinations is the share of the product's exports that is exposed or banned
    '''
    e = us[us["Province"].eq(province)].groupby("hs4")[MESURES].sum()
    e = e[e["exposure"] > 0]
    e["hs4_tot_exp"] = exports[exports["Province"].eq(province)].groupby("hs4")["Value/Valeur"].sum().reindex(e.index)
    e["intensity"] = 100 * e["exposure"] / e["hs4_tot_exp"]
    e["share_prov_exposure"] = 100 * e["exposure"] / e["exposure"].sum()
    e["min_line_share"] = us[us["Province"].eq(province) & us["exposed"]].groupby("hs4")["line_share"].min().reindex(e.index)
    xpshs4 =  e.sort_values("exposure", ascending=False)

    return xpshs4


################ results
def province_table(prov):
    '''
    exposure by province output in billion CAD, with readable column names.'''
    t = prov.copy()
    money = [*MESURES, "exports_to_us", "total_exports", "gdp"]
    t[money] = t[money] / 1e9
    t=  t.rename(columns={"exposure": "Exposure, HS6 (B CAD)", "exposure_full_hs6": "Lower bound: fully listed HS6 (B CAD)",
                             "exposure_line_weighted": "Line-weighted estimate (B CAD)",
                             "exposure_import_ban": "of which import ban (B CAD)", "exports_to_us": "Exports to U.S. (B CAD)",
                             "total_exports": "Total exports (B CAD)", "gdp": "GDP 2024 (B CAD)",
                             "share_of_canada_exposure_%": "Share of Canada (%)",
                             "intensity_total_exports_%": "% of total exports",
                             "intensity_us_exports_%": "% of exports to U.S.", "intensity_gdp_%": "% of GDP"})
    return t


def hs4_table(exports, us, province, hs_names, n=15):
    '''Top `n` exposed HS4 of a province, values in million CAD.'''
    e = xposr_hs4(exports, us, province).head(n)
    e.insert(0, "Product", e.index.map(hs_names))
    money = [*MESURES, "hs4_tot_exp"]
    e[money] = e[money] / 1e6
    hs4tab = e.rename(columns={"exposure": "Exposure, HS6 (Millions CAD)", 
                             "exposure_full_hs6": "Lower bound: fully listed HS6 (Millions CAD)",
                             "exposure_line_weighted": "Line-weighted estimate (Millions CAD)",
                             "exposure_import_ban": "Import ban (Millions CAD)",
                             "hs4_tot_exp": "Province exports of HS4, all destinations (Millions CAD)",
                             "intensity": "Exposure intensity (%)", "share_prov_exposure": "Share of province exposure (%)",
                             "min_line_share": "Min. U.S. line share"})

    return hs4tab


def tophs4_prov(exports, us, hs_names, provinces=MAJOR_PROVINCES, n=3):
    '''
    N largest exposed HS4 of each province,
    '''
    rows = []
    for prov in provinces:
        for hs4, r in xposr_hs4(exports, us, prov).head(n).iterrows():
            rows.append({"Province": prov, 
                         "HS4": hs4, 
                         "Product": hs_names.get(hs4, ""),
                         "Exposure (Millions CAD)": r["exposure"] / 1e6, 
                         "Intensity (%)": r["intensity"],
                         "Share of province exposure (%)": r["share_prov_exposure"]})
    tophs4prov =  pd.DataFrame(rows)
    
    return tophs4prov