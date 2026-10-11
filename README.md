# Trade Data Scientist

This repo contains code and supporting documents to the written assessment:
- Question 1: Export growth and diversification;
- Question2: Exposure to U.S. Section 338 measures.

## Repository structure

```
notebooks/Assignment.ipynb  Results of the written assessment. Run it from first to last cell.
src/trade_data/
    ingestion.py    #loads data from multiple sources
    processing.py   #cleaning source data
    analysis.py     #indicators calculations
    modelling.py    #model and validations
    s338.py builds  #Section 338 updated list
data/   #source files
```
## Install
Python 3.14 was used.

Run the following to configure the virtual environment and install all dependencies:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Then, to replicate the results presented in the document, 
- run notebook 'notebooks/Assignment.ipynb'.

## Note
An internet connexion is required as some functions download files at runtime.

## S338 Data

|Filename|link|source_name|
|---|---|---|
|s338_products.csv|https://github.com/Budget-Lab-Yale/tariff-rate-tracker/blob/master/resources/s338_products.csv|load_s338_yale|
|cbp_section338_canada_hts_list_2026-08-21.pdf|https://content.govdelivery.com/attachments/USDHSCBP/2026/08/21/file_attachments/3754630/Section|338 Canada HTS LIST Final.pdf|	
fr_2026-18837.htm / .pdf|https://www.govinfo.gov/content/pkg/FR-2026-09-14/html/2026-18837.htm and https://www.govinfo.gov/content/pkg/FR-2026-09-14/pdf/2026-18837.pdf|IMPORT_BAN|
|fr_2026-18838.htm|https://www.govinfo.gov/content/pkg/FR-2026-09-14/html/2026-18838.htm|PART_A_ADDED_ALCOHOL|
|fr_2026-18839.pdf|Federal Register : https://public-inspection.federalregister.gov/2026-18839.pdf · document page : https://www.federalregister.gov/d/2026-18839|PART_A_ADDED and PART_B_REMOVED|
|usitc_hts.json|(~12 Mo)	https://hts.usitc.gov/reststop/exportList?from=0101&to=9999&format=JSON&styles=false|USITC|
