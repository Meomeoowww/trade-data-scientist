import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from trade_data.processing import CPTPP, RCEP, gravity_processing, map_iso2_iso3, statcan_province_gdp_processing, statcan_province_pop_processing, table_baci_processing, table_macro_processing, table_province_hs4_processing, tariff_processing

ALBERTA = "AB"
PANEL = ["USA", "CAN", "MEX", "CHN", "KOR", "JPN", "GBR"]
PARTNERS = [c for c in PANEL if c != "CAN"]
SCALE = 1000                     # target in million USD (BACI is in thousand USD)

FEATURES = ["log_gdp_usd_o", "log_gdp_usd_d", "gdp_per_cap_usd_o", "gdp_per_cap_usd_d", "log_pop_o", "log_pop_d",
            "trade_op_pct_gdp_o", "trade_op_pct_gdp_d", "rule_of_law_est_o", "rule_of_law_est_d",
            "regulatory_quality_est_o", "regulatory_quality_est_d",
            "log_distw_harmonic", "contig", "comlang_off", "comlang_ethno", "comcol", "col45",
            "fta", "ave", "log_supply", "log_demand", "hs2"]


############## COMPLEMENTARY PROCESSING STEPS
########### 1 CREATE TABLES INPUT
def prepare_inputs(cimt, baci, cepii_country, wdi, wgi, statcan, gravity, itc, rate, province=ALBERTA):
    '''
    Function that builds all 5 reference tables used by the model form raw.
    Returns a dictionary with flows, ab_2024, ab_2025, macro, grav, tariffs, hs4 (sorted list), rate.
    '''
    # world HS4 flows in thousand USD
    flows = (table_baci_processing(baci, cepii_country, countries=None)
             .drop(columns="year").rename(columns={"c_o": "o", "c_d": "d", "value": "v"}))

    # AB's exports by partner (ISO3) and hs4 in thousand usd
    isomap = map_iso2_iso3(cepii_country)
    ab_2024 = table_province_hs4_processing(cimt, province, 2024, rate, isomap)
    ab_2025 = table_province_hs4_processing(cimt, province, 2025, rate, isomap)

    # macro table: every country + an Alberta row (StatCan GDP and population, Canadian proxies)
    pop = statcan_province_pop_processing(statcan, 2024, 2025, province)
    gdp = statcan_province_gdp_processing(statcan, 2024, 2025, province)
    ab_pop = dict(zip(pop["year"].astype(int), pop["VALUE"]))
    ab_gdp = gdp.loc[gdp["year"] == 2024, "VALUE"].iloc[0]          # million CAD
    macro = table_macro_processing(wdi, wgi, ab_gdp, ab_pop, rate)

    # gravity variables and applied tariffs (ofc, for ITC: the reporter is the importer)
    grav = gravity_processing(gravity)
    tariffs = (tariff_processing(itc)
               .rename(columns={"ReportingCountry": "d", "PartnerCountry": "o_ctry",
                                "Year": "year", "ProductCode": "hs4", "AVE": "ave"})
               [["year", "d", "o_ctry", "hs4", "ave"]]
               .drop_duplicates(["year", "d", "o_ctry", "hs4"]))

    #final dictionary dicomix
    dicomix =  {"flows": flows, "ab_2024": ab_2024, "ab_2025": ab_2025, "macro": macro, "grav": grav,
            "tariffs": tariffs, "hs4": sorted(flows["hs4"].unique()), "rate": rate}
    
    
    return dicomix

############ 2 GRIDS FEATURES
def make_grid(exporters, destinations, hs4_list):
    '''
    Grid all possible combinations, as we mentioned that some combinations might not record trade from our cepii database
    we do this so that the model will also learn that we might have 0 trade in some cases

    '''
    o = pd.DataFrame({"o": exporters})
    d = pd.DataFrame({"d": destinations})
    
    # cross merge to get all o d combi
    grido = o.merge(d, how="cross")
    # filter cases where country is trading with itself and eventually cases of interprovince trade
    grido = grido.query("o != d and not (o == 'AB' and d == 'CAN')")
    # add effectively used hs4 codes
    grido = grido.merge(pd.DataFrame({"hs4": hs4_list}), how="cross").reset_index(drop=True)
    
    return grido


def build_features(grid, year, inputs):
    '''
    Merges all the databases into 1 matrix: 
    inputs: dict returned by prepare_inputs:
    input: dict returned by prepare_input (flows, ab_2024, macro, grav, tariffs)
    supply/demand are always 2024, latest baci data available. Canada proxy of Alberta in gravity and tariff variables, except contig (Alberta borders with u.s.)
    returns grid with explanatory vars, their logs and hs2
    '''
    flows, macro, grav, tariffs = inputs["flows"], inputs["macro"], inputs["grav"], inputs["tariffs"]
    g = grid.copy()
    g["o_ctry"] = g["o"].replace({ALBERTA: "CAN"})


    # Create supply variable as total exports of country o by hs4
    # and demand as total imports of country d by hs4 in a year (2024)
    temp = inputs["ab_2024"].copy()
    temp['o'] = "AB" # ab24 doesnt natively have origin, so we add to make it similar to flows dataframe
    A = pd.concat([flows, temp])

    world_exp = A.groupby(["o", "hs4"])["v"].sum().rename("supply")
    world_imp = flows.groupby(["d", "hs4"])["v"].sum().rename("demand")
    # join on the grid supply and demand variables based exporter importer and hs4 code
    g = g.join(world_exp, on=["o", "hs4"]).join(world_imp, on=["d", "hs4"])
    g[["supply", "demand"]] = g[["supply", "demand"]].fillna(0) # fill 0 when no trade


    # Now we ad macro variables  for 
    # - exporter "_o"
    # - importer "_d" for a year
    mac = macro[macro["year"].eq(year)].drop(columns="year")
    g = g.merge(mac.add_suffix("_o"), 
                left_on="o", 
                right_on="country_o", 
                how="left", 
                validate="many_to_one")
    g = g.merge(mac.add_suffix("_d"), left_on="d", right_on="country_d", 
                how="left", validate="many_to_one")

    # Adding gravity cepii elements  for a pair exporter - destination
    # Alberta has borders with USA
    g = g.merge(grav, on=["o_ctry", "d"], how="left", validate="many_to_one")

    # Define boolean mask for alberta that will be used to filter df
    # is flase in train and true in pred df
    ab = g["o"].eq(ALBERTA)

    # We then correct borders variable specifically for alberta that only has borders with USA
    # Is "True" only when dest is USA and we convert to 0 1 with astype
    g.loc[ab, "contig"] = g.loc[ab, "d"].eq("USA").astype(float)

    # We correct FTA free trade agreement between pairs with CPTPP RCEP
    # Add Comprehensive and Progressive Agreement for Trans-Pacific Partnership (CPTPP) with/without UK
    # Regional Comprehensive Economic Partnership (RCEP) 
    cptpp = CPTPP | ({"GBR"} if year >= 2025 else set())
    g["fta"] = (g["fta_wto"].eq(1)
                | (g["o_ctry"].isin(cptpp) & g["d"].isin(cptpp))
                | (g["o_ctry"].isin(RCEP) & g["d"].isin(RCEP))).astype(int)

    # applied tariff set by the importer on the exporter's country
    g = g.merge(tariffs[tariffs["year"].eq(year)].drop(columns="year"),
                on=["d", "o_ctry", "hs4"], how="left", validate="many_to_one")
    
    # to mitigate scale issues, log macro vars
    for c in ["supply", "demand", "gdp_usd_o", "gdp_usd_d", "pop_o", "pop_d", "distw_harmonic"]:
        g[f"log_{c}"] = np.log1p(g[c].astype(float))
    g["hs2"] = pd.Categorical(g["hs4"].str[:2])
    
    return g


def training_set(inputs):
    '''2024 training table, w.(7) panel countries x 6 partners x HS4
        with the target y (0 when no trade).
    '''
    # Build complete train dataframe
    ## create all features for our panel (grid_train)
    train = build_features(make_grid(PANEL, PANEL, inputs["hs4"]), 2024, inputs) # no value variable here
    # add export values as y (objective) and fill 0 for empty flows
    train = train.merge(inputs["flows"].rename(columns={"v": "y"}), 
                        on=["o", "d", "hs4"],
                        how="left", validate="one_to_one"
                        ) #
    train["y"] = train["y"].fillna(0)
    return train


def alberta_destinations(inputs):
    '''
    # we prepare alberta's destinations that have :
    # * cepii gravity data 
    # ** have a 2025 gdp or at least imputed gdp from previous year 
    # *** have at least traded (then has a demand)
    '''
    # pick grav, macro, flows as input
    grav, macro, flows = inputs["grav"], inputs["macro"], inputs["flows"]
    
    albeedest =  sorted((set(grav.loc[grav["o_ctry"].eq("CAN"), "d"]) # have at leats cepii gravity data for canada as exporter
                   & set(macro.loc[macro["year"].eq(2025) & macro["gdp_usd"].notna(), "country"]) # have a 2025 gdp or at least imputed gdp from previous year 
                   & set(flows["d"])) - {"CAN"}) #have at least traded (then has a demand)
    return albeedest



def alberta_set(inputs):
    '''
    # Build complete predict dataframe (ALBERTA),
    # Prediction table: Alberta x every destination x HS4 (one row per destination-product pair)
    # Returns:
    # strictly 2025 macro vars for alberta + hs2 + log of supply & demand 2024 + observed_2024, observed_2025
    # reminder: observed_2024, observed_2025 are our Alberta's CIMT data (real data) and wont be used by func. predict. Used for validation 
    '''
    ab_world = build_features(make_grid([ALBERTA], alberta_destinations(inputs), inputs["hs4"]), 2025, inputs)
    for year in (2024, 2025):
        obs = inputs[f"ab_{year}"].rename(columns={"v": f"observed_{year}"})
        ab_world = ab_world.merge(obs, on=["d", "hs4"], how="left", validate="many_to_one")
    
    ab_world[["observed_2024", "observed_2025"]] = ab_world[["observed_2024", "observed_2025"]].fillna(0)
    
    return ab_world


# Function to  check tables quality on some criteria
def check_tables(train, ab_world):
    '''
    # Quality checks on the merged tables, 
    # Run before modelling.
    '''
    n_hs4 = train["hs4"].nunique()
    assert len(train) == 42 * n_hs4, f"train has {len(train)} rows instead of {42 * n_hs4}"

    # Check number of rows , columns , number of zero values for for 'y' as exports
    print(f"Training rows: {len(train):,} | share of zero flows: {train['y'].eq(0).mean():.1%}")
    print("Missing values ('ave' could be > 0):")
    print(train[["gdp_usd_o", "gdp_usd_d", "distw_harmonic", "contig", "ave"]].isna().mean().round(3).to_string())
    
    us = ab_world.loc[ab_world["d"].eq("USA"), ["gdp_usd_o", "contig", "fta"]].drop_duplicates()
    print(f"Alberta -> USA: GDP {us['gdp_usd_o'].iloc[0]:.3e} USD, contig {us['contig'].iloc[0]:.0f}, fta {us['fta'].iloc[0]}")
    print(f"Alberta destinations: {ab_world['d'].nunique()}")


######################## 3 model & validation

def make_xgb():
    '''
    As replacement to PPML Gravity model (Santos Silva & Tenreyro, 2006) 
    that uses Poisson's loss function (dev) to handle zero flows
    we pick XGBoost with a Poisson objective : 
    '''
    mod = xgb.XGBRegressor(
        objective="count:poisson", 
        n_estimators=600, 
        learning_rate=0.05, 
        max_depth=6,
        min_child_weight=5, subsample=0.8, 
        colsample_bytree=0.8, reg_lambda=1.0,
        tree_method="hist", enable_categorical=True, 
        random_state=42
        )
    return mod


def fit_xgb(df):
    ''' 
    train the xgb model and take training_ds as input. 
    We divide y by 1000 to get millions usd 
    '''
    fitfunc=  make_xgb().fit(df[FEATURES], df["y"] / SCALE)

    return fitfunc


def predict(model, df):
    '''
    Predicted flows in thousand USD.'''
    predo =  model.predict(df[FEATURES]) * SCALE

    return predo

def allocated_share(y, pred):
    '''
    Share of observed value placed in the right cell: 
    From 0 to 1.
    '''
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    alloco =  np.minimum(y / y.sum(), pred / pred.sum()).sum()
    
    return alloco

def evaluate(y, pred, dest):
    '''
    Eval functions.
    WAPE, rank correlation, spearman
    presence AUC and WAPE on destination totals
    plus the share of value correctly allocated across flows and across destinations
    '''
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    by_dest = pd.DataFrame({"d": np.asarray(dest), "y": y, "p": pred}).groupby("d")[["y", "p"]].sum()
    
    res_eval = {"WAPE_flow": np.abs(y - pred).sum() / y.sum(),
            "Spearman_flow": spearmanr(y, pred)[0],
            "AUC_presence": roc_auc_score(y > 0, pred) if 0 < (y > 0).mean() < 1 else np.nan,
            "WAPE_destination": np.abs(by_dest["y"] - by_dest["p"]).sum() / by_dest["y"].sum(),
            "Allocated_flow": allocated_share(y, pred),
            "Allocated_destination": allocated_share(by_dest["y"], by_dest["p"])}    
    
    return res_eval

def capture_top(y, score, share=0.10):
    '''
    Pairs ranked in the top `share` by `score`
    '''
    y, score = np.asarray(y, float), np.asarray(score, float)
    top = pd.Series(score).rank(method="first", ascending=False).to_numpy() <= round(share * len(y))
    opened = y > 0
    return {"AUC": roc_auc_score(opened, score),
            f"Openings in top {share:.0%}": opened[top].sum() / opened.sum(),
            f"Value in top {share:.0%}": y[top].sum() / y.sum(),
            f"Lift top {share:.0%}": opened[top].mean() / opened.mean()}

def leave_one_exporter_out(train):
    '''
    As we plan to use a model that knows Canada but not specifically Alberta during ints training
    We want to test if each panel exporter is predicted by a model that never saw it (Alberta's situation)
    So we remove one country and train the model to see how it performs when we use it to predict the new country then
    '''
    rows = []
    for exporter in PANEL:
        test, fit_df = train[train["o"].eq(exporter)], train[train["o"].ne(exporter)]
        rows.append({"held_out_exporter": exporter,
                     **evaluate(test["y"], predict(fit_xgb(fit_df), test), test["d"])})
     
    valid = pd.DataFrame(rows).set_index("held_out_exporter")
    return valid


def feature_importance(model):
    '''
    Importance of features by variables (%).
    '''
    gain = pd.Series(model.get_booster().get_score(importance_type="gain"))
    f_i = (100 * gain / gain.sum()).sort_values(ascending=False).round(1).rename("gain share (%)")
    return f_i


################## 4 ALBERTA POTENTIAL
def add_potential(ab_world, model, inputs):
    '''
    Potential is measuring here geographic diversification.
    Alberta's 2024 exports of each HS4 product reallocated across destinations in proportion to the model's predictions
    # How it is built:
    Adds the raw prediction, 
    the supply-constrained potential (Alberta's 2024 HS4 exports reallocated across
    destinations in proportion to the predictions) 
    and the result columns in million CAD.
    We work with ab_world as Alberta x 194 destinations x 1 228 HS4 and real flows observed in 2024 2025
    '''

    # We predict with ab_world dataframe 2025 AB's exports
    df = ab_world.copy()
    df["pred"] = predict(model, df)

    # Real supply (here AB exports in 2024 by hs4)
    supply = inputs["ab_2024"].groupby("hs4")["v"].sum()
    # Exports predicted by hs4
    total = df.groupby("hs4")["pred"].transform("sum")

    # contrained prediction is applying 2025 predicted structure for a product to its 2024 supply
    df["pred_constrained"] = df["pred"] * df["hs4"].map(supply).fillna(0) / total.where(total > 0)
    to_mcad = inputs["rate"][2025] / 1000 # 1000 usd to million CAD
    df["Observed 2025 (M CAD)"] = df["observed_2025"] * to_mcad
    df["Potential 2025 (M CAD)"] = df["pred_constrained"] * to_mcad
    return df


def validate_alberta(ab_world, inputs, share=0.10):
    '''
    ### Validation 2: Alberta 2025 vs observed on the panel partners, 
    against the naive "2025 = 2024" benchmark
    
    ### Validation 3: among pairs with no Alberta exports in 2024, 
    does the ranking point to those that opened in 2025?
    The model is compared with the rule where canada is already exporting a product to a destination in 2024 
    (the naive benchmark cannot rank these pairs: AUC 0.5)
    '''
    pan = ab_world[ab_world["d"].isin(PARTNERS)]
    scores = pd.DataFrame({
        "XGBoost (unconstrained)": evaluate(pan["observed_2025"], pan["pred"], pan["d"]),
        "XGBoost (supply-constrained)": evaluate(pan["observed_2025"], pan["pred_constrained"], pan["d"]),
        "Naive: 2025 is equal to 2024": evaluate(pan["observed_2025"], pan["observed_2024"], pan["d"]),
    }).T

    flows = inputs["flows"]
    canada = flows[flows["o"].eq("CAN")].set_index(["d", "hs4"])["v"]
    new = ab_world[ab_world["observed_2024"].eq(0)]
    rules = {"Prediction made by XGBoost": new["pred"],
             "Canada already exports the HS4 to d (2024)": canada.reindex(pd.MultiIndex.from_frame(new[["d", "hs4"]])).fillna(0),
    }
    new_markets = pd.DataFrame({k: capture_top(new["observed_2025"], v, share) for k, v in rules.items()}).T

    double_res =  scores, new_markets
    
    return double_res



########################## GENERATE RESULT TABLES

def top_tables(df, 
               col, 
               hs_names, 
               ctry_names, 
               n_dest=10, 
               n_prod=5, 
               other_cols=("Observed 2025 (M CAD)",)):
    '''
    Top destinations by provided `col` and the top HS4 products of each
    '''
    allcols = [col, *other_cols]
    dest = df.groupby("d")[allcols].sum().sort_values(col, ascending=False).head(n_dest)
    dest.insert(0, "Country", dest.index.map(ctry_names))
    dest["In training panel"] = dest.index.isin(PANEL)
    top = (df[df["d"].isin(dest.index)].sort_values(col, ascending=False)
           .groupby("d", sort=False).head(n_prod).copy())
    top["Country"] = top["d"].map(ctry_names)
    top["Product"] = top["hs4"].map(hs_names)
    top["rank"] = top["d"].map({d: i + 1 for i, d in enumerate(dest.index)})

    double_res =  dest, top.sort_values(["rank", col], ascending=[True, False])[["rank", "Country", "hs4", "Product", *allcols]]
    return double_res

def without_fuels(ab_world):
    '''
    Check same results without HS 27
    # With the unrealized potential max(potential - observed, 0).
    '''

    nofuel = ab_world[~ab_world["hs4"].str.startswith("27")].copy() #filter hs27
    nofuel["Unrealized 2025 (M CAD)"] = (nofuel["Potential 2025 (M CAD)"] - nofuel["Observed 2025 (M CAD)"]).clip(lower=0)
    
    totalixs = {"Observed 2025 (B CAD)": nofuel["Observed 2025 (M CAD)"].sum() / 1e3,
              "Unrealized potential (B CAD)": nofuel["Unrealized 2025 (M CAD)"].sum() / 1e3}
    
    return nofuel, totalixs