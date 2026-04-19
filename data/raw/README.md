# Raw Data

## extracted_2x2.csv

Included in this repository. Contains 2x2 diagnostic accuracy data extracted from 6 published studies (11 cohort-rule combinations). See `data_dictionary.md` for variable definitions.

## PECARN Biosignatures Dataset

**Not included** — governed by a Research Data Use Agreement.

To obtain the dataset:

1. Visit https://pecarn.org/datasets/
2. Navigate to the Biosignatures study dataset
3. Complete the click-through Data Use Agreement
4. Download the full dataset (Biosignatures_Full)
5. Extract the CSV files and place them at:

```
data/raw/pecarn_tig/Biosignatures_Full/CSV datasets/
  ├── demographics.csv
  ├── clinicaldata.csv
  ├── labresults.csv
  ├── pctdata.csv
  ├── labresults_otherblood.csv
  ├── culturereview_blood.csv
  └── culturereview_csf.csv
```

The Python scripts expect CSV format. If you download SAS datasets (.sas7bdat), convert them to CSV first.
