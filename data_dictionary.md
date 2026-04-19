# Data Dictionary

## extracted_2x2.csv

Source: Author extraction from published studies. Each row is one cohort-rule combination.

| Variable | Type | Description | Units |
|----------|------|-------------|-------|
| rule | string | Decision rule name (e.g., "PECARN", "Aronson") | — |
| cohort | string | Study cohort identifier (e.g., "Kuppermann2019_val") | — |
| publication_year | integer | Year of publication | — |
| threshold | string | Decision threshold applied | — |
| n | integer | Total cohort size | patients |
| age_range | string | Age range of included infants | — |
| ibi_prevalence | float | IBI prevalence in the cohort | proportion |
| TP | integer | True positives (IBI detected by rule) | count |
| FP | integer | False positives (no IBI, flagged by rule) | count |
| FN | integer | False negatives (IBI missed by rule) | count |
| TN | integer | True negatives (no IBI, not flagged) | count |
| age_group | string | Age stratum label | — |
| notes | string | Source details, PMID, verification status | — |

## PECARN Biosignatures Variables Used

Source: PECARN Biosignatures public-use dataset (not redistributable). Variables listed here are those used by the prediction model and analysis scripts.

### demographics.csv

| Variable | Type | Description | Units |
|----------|------|-------------|-------|
| PId | integer | Patient identifier | — |
| BirthDay | integer | Days before enrolment (negative; age_days = -BirthDay) | days |

### clinicaldata.csv

| Variable | Type | Description | Units |
|----------|------|-------------|-------|
| PId | integer | Patient identifier | — |
| Temperature | float | Rectal temperature (mixed F/C; converted to C if >60) | degrees |
| YOSCry | integer | Yale Observation Scale: quality of cry (1/3/5) | score |
| YOSReaction | integer | YOS: reaction to parent stimulation (1/3/5) | score |
| YOSState | integer | YOS: state variation (1/3/5) | score |
| YOSColor | integer | YOS: colour (1/3/5) | score |
| YOSHydration | integer | YOS: hydration (1/3/5) | score |
| YOSResponse | integer | YOS: response to social overtures (1/3/5) | score |

### labresults.csv

| Variable | Type | Description | Units |
|----------|------|-------------|-------|
| PId | integer | Patient identifier | — |
| WBC | float | White blood cell count | x10^3/uL |
| ANC | float | Absolute neutrophil count | x10^3/uL |
| UrineLEC | integer | Urine leukocyte esterase (format DV7033G: 1=small, 2=moderate, 3=large, 4=negative) | categorical |
| NitriteRes | integer | Urine nitrite result (1=positive) | binary |

### pctdata.csv

| Variable | Type | Description | Units |
|----------|------|-------------|-------|
| PId | integer | Patient identifier | — |
| PCTResult | string | Procalcitonin result (converted to numeric) | ng/mL |

### labresults_otherblood.csv

| Variable | Type | Description | Units |
|----------|------|-------------|-------|
| PId | integer | Patient identifier | — |
| BloodTest | integer | Test type (1 = CRP) | categorical |
| BloodResult | float | Test result value | mg/L (for CRP) |

### culturereview_blood.csv

| Variable | Type | Description | Units |
|----------|------|-------------|-------|
| PId | integer | Patient identifier | — |
| BloodDCCAssess | integer | DCC assessment: 1 = true pathogen (bacteraemia) | binary |

### culturereview_csf.csv

| Variable | Type | Description | Units |
|----------|------|-------------|-------|
| PId | integer | Patient identifier | — |
| CSFDCCAssess | integer | DCC assessment: 1 = true pathogen (meningitis) | binary |

## Derived Variables

| Variable | Derivation | Used in |
|----------|------------|---------|
| age_days | -BirthDay (from demographics.csv) | All models |
| temp_c | Temperature if <=60, else (Temperature - 32) * 5/9 | All models |
| ua_pos | 1 if UrineLEC in {1,2,3} OR NitriteRes == 1; else 0 | All models |
| yos_total | Sum of 6 YOS components (min_count=6) | v6 model |
| age_young | 1 if age_days <= 14; else 0 | v6 model |
| has_ibi | 1 if PId in bacteraemia OR meningitis set | Outcome |
| pct | Numeric conversion of PCTResult | Degradation analysis |
