# Data: BRFSS 2015 Diabetes Health Indicators

**Raw data is never committed.** `data/raw/*` is git-ignored. Only `data/raw/MANIFEST.sha256` is tracked, so that anyone can verify they hold byte-identical files. The scripts in `ml/` only read the CSVs and never modify them.

All figures below were produced by `python -m ml.data_validation` and `python -m ml.exploratory_analysis` on 2026-10-14. The machine-readable versions are in `results/`.

## 1. Files

| File | Role | Rows × cols | Target | SHA-256 (first 16 hex) |
|---|---|---|---|---|
| `diabetes_binary_health_indicators_BRFSS2015.csv` | **Primary** (all modelling) | 253,680 × 22 | `Diabetes_binary` | `19f367e3e3350768` |
| `diabetes_012_health_indicators_BRFSS2015.csv` | Alternative (3-class) | 253,680 × 22 | `Diabetes_012` | `a971809d0786d2d7` |
| `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` | Alternative (balanced) | 70,692 × 22 | `Diabetes_binary` | `873128ba0264d66a` |
| `online_retail.csv` | **Excluded.** Retail transactions, not health data. It is hashed only and never loaded. | n/a | n/a | `c820e928a9cb01d0` |

The full hashes are in `MANIFEST.sha256`.

**To obtain the files:**
1. Download the dataset from the Kaggle page [alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).
2. Place the three diabetes CSVs, with their original names, in `data/raw/`.
3. Run `python -m ml.data_validation`. The `sha256_matches_manifest` check must pass. If it fails, your copy differs from the one analysed here.

## 2. Source, provenance, citation, license

Each statement below is marked as **verified** (read from the source on 2026-10-14) or **not verified**.

| Item | Statement | Status |
|---|---|---|
| Original survey | CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015, a health-related telephone survey that combines landline and cell-phone samples. The 2015 file covers the 50 states, DC, Guam, and Puerto Rico, with 441,456 records and 330 variables. Source: [CDC 2015 BRFSS page](https://www.cdc.gov/brfss/annual_data/annual_2015.html). | Verified |
| Cleaned dataset | Created by Alex Teboul on Kaggle as "a cleaned and consolidated dataset created from the BRFSS 2015 dataset already on Kaggle". The author states: "I did not create this dataset". Source: [Kaggle page](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset). | Verified |
| Kaggle dataset license | **CC0: Public Domain**, as shown on the Kaggle dataset page. | Verified |
| Cleaning notebook license | Apache 2.0. Source: [Kaggle notebook](https://www.kaggle.com/code/alexteboul/diabetes-health-indicators-dataset-notebook). | Verified (license only). The notebook's code did not render without a login, so **the cleaning steps themselves are not verified.** |
| UCI mirror | "CDC Diabetes Health Indicators", UCI ML Repository dataset 891, linked 2023-09-25, funded by the CDC. Source: [UCI page](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators). | Verified |
| UCI license | Not shown on the fetched page, and the UCI API returned an empty `license` field. | **Not verified** |
| UCI source inconsistency | The UCI page links the **2014** BRFSS release, while the filenames and Kaggle description say **2015**. This project treats the data as 2015, following the filenames and the Kaggle author. | Verified (the inconsistency itself) |
| CDC reuse terms | BRFSS data is publicly released by CDC. The CDC reuse-policy page returned "Access Denied" to automated requests, so no formal license statement was confirmed. | **Not verified** |

**Citations to use in the report:**
- UCI, verified citation text: *CDC Diabetes Health Indicators [Dataset]. (2017). UCI Machine Learning Repository. https://doi.org/10.24432/C53919.* UCI adds: "please follow the acknowledgment policy on the original dataset website".
- Kaggle: *Teboul, A. Diabetes Health Indicators Dataset. Kaggle. https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset* (CC0).
- CDC (standard BRFSS wording, quoted from memory and **not re-verified**): *Centers for Disease Control and Prevention (CDC). Behavioral Risk Factor Surveillance System Survey Data. Atlanta, Georgia: U.S. Department of Health and Human Services, Centers for Disease Control and Prevention, 2015.*

## 3. Target definition: documentation vs. observed data

| File | Documented definition (Kaggle, UCI) | Observed in the files |
|---|---|---|
| `diabetes_binary_*` | 0 = no diabetes, 1 = **prediabetes or diabetes** | 1 = **diabetes only**. Prediabetes is coded 0. |
| `diabetes_012_*` | 0 = no diabetes or only during pregnancy, 1 = prediabetes, 2 = diabetes | Matches the documentation |

**Evidence** (`results/label_mapping_012_vs_binary.csv`):
- The 012 file and the primary file contain the same respondents in the same row order. All 21 features are identical row by row.
- Cross-tabulating the two targets gives this mapping:

| Diabetes_012 | Diabetes_binary = 0 | Diabetes_binary = 1 |
|---|---|---|
| 0 (no diabetes) | 213,703 | 0 |
| 1 (prediabetes) | **4,631** | 0 |
| 2 (diabetes) | 0 | 35,346 |

- In the balanced file, there are 35,346 positives, exactly the number of diabetes rows, and every positive matches a diabetes row. At least 684 of its class-0 rows have feature profiles that occur only among prediabetes respondents.

**Decision:**
- **Labels are not changed.** This project's target is *self-reported diagnosed diabetes (1) versus no diabetes, where class 0 includes prediabetes and diabetes only during pregnancy*.
- All reports will use this observed definition, not the documented one.
- A model trained on this target is **not** a prediabetes screener.

## 4. Features (21 inputs; the target is excluded)

Meanings come from the UCI variable table (verified). BRFSS source variable names are given only where UCI states them.

| Feature | Type | Encoding |
|---|---|---|
| HighBP | binary | 0 = no high blood pressure, 1 = high BP |
| HighChol | binary | 0 = no high cholesterol, 1 = high cholesterol |
| CholCheck | binary | 0 = no cholesterol check in 5 years, 1 = yes |
| BMI | continuous (integer) | Body Mass Index; observed 12–98 |
| Smoker | binary | Smoked at least 100 cigarettes in entire life: 0 = no, 1 = yes |
| Stroke | binary | Ever told you had a stroke: 0 = no, 1 = yes |
| HeartDiseaseorAttack | binary | Coronary heart disease or myocardial infarction: 0 = no, 1 = yes |
| PhysActivity | binary | Physical activity in past 30 days, not including job: 0 = no, 1 = yes |
| Fruits | binary | Fruit 1+ times per day: 0 = no, 1 = yes |
| Veggies | binary | Vegetables 1+ times per day: 0 = no, 1 = yes |
| HvyAlcoholConsump | binary | Men >14 or women >7 drinks per week: 0 = no, 1 = yes |
| AnyHealthcare | binary | Any health-care coverage: 0 = no, 1 = yes |
| NoDocbcCost | binary | Couldn't see a doctor because of cost in past 12 months: 0 = no, 1 = yes |
| GenHlth | ordinal 1–5 | Self-rated health: 1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor |
| MentHlth | count 0–30 | Days of poor mental health in past 30 days. UCI says "1-30", but **0 occurs** (most common value). 0 is read as zero days. |
| PhysHlth | count 0–30 | Days of poor physical health in past 30 days. Same note about 0 as MentHlth. |
| DiffWalk | binary | Serious difficulty walking or climbing stairs: 0 = no, 1 = yes |
| Sex | binary | 0 = female, 1 = male |
| Age | ordinal 1–13 | `_AGEG5YR`: 1 = 18–24, 9 = 60–64, 13 = 80+. These three anchors are from UCI. The intermediate codes are presumably 5-year bands, but the CDC codebook PDF returned HTTP 403, so they are **not verified**. |
| Education | ordinal 1–6 | `EDUCA`: 1 = never attended or kindergarten only, 2 = grades 1–8, 3 = grades 9–11, 4 = grade 12 or GED, 5 = college 1–3 years, 6 = college 4+ years |
| Income | ordinal 1–8 | `INCOME2`: 1 = < $10,000, 5 = < $35,000, 8 = $75,000+. These anchors are from UCI. The intermediate brackets are **not verified**, for the same codebook reason as Age. |

Every value in every column is within the documented range. Per-feature statistics, including class-conditional means, standardized mean differences, and correlation with the target, are in `results/feature_statistics.csv`.

## 5. Validation findings (primary file)

| Check | Result |
|---|---|
| Shape | 253,680 × 22, as expected |
| Columns | All 22 expected names, in the expected order |
| Data types | All `float64`, but all 5,580,960 cells are integer-valued |
| Missing values | 0 |
| Target values | {0, 1}. 0 = 218,334 (86.07%), 1 = 35,346 (13.93%) |
| Balance | **Imbalanced**, 6.18 : 1. Accuracy alone is misleading: predicting "no diabetes" for everyone scores 86.07%. |
| Exact duplicate rows | 24,206 (9.54%). **Retained, see below.** |
| Identical features, different label | 1,566 feature profiles covering 5,218 rows. The features cannot fully determine the label. |
| BMI extremes | 68 rows with BMI < 15. 805 rows with BMI > 60, of which 54 have BMI ≥ 90. Retained, because no valid range is documented. |

**Duplicates are kept.** Each row has only 21 coarse integer codes, so two different respondents can easily share an identical profile. Removing duplicates would remove real people and distort class frequencies. Two safeguards apply:
1. A sensitivity analysis with duplicates removed will be reported in the modelling phase.
2. Splits are made at the row level, so identical profiles can fall in both the training and test sets. That is legitimate here, because the rows are distinct respondents, but it is stated as a caveat.

**Alternative datasets:**
- The 3-class file is highly imbalanced: 84.24% no diabetes, 1.83% prediabetes, 13.93% diabetes, a 46.15 : 1 ratio between the largest and smallest class.
- The balanced file is exactly 50 / 50, and **all 70,692 rows match rows of the primary file**. It is therefore a subsample of the primary file.
- **Neither alternative file may be combined with the primary file.** Combining the 3-class file would duplicate every respondent. Combining the balanced file would leak test rows into training.

## 6. Limitations

- **Self-reported, cross-sectional survey.** The diagnoses, BMI (from self-reported height and weight), and behaviours are all answers to a telephone survey, not clinical measurements. The models predict association, not causation, and are **not diagnostic tools**.
- **No survey weights.** BRFSS relies on weighting (raking) to represent the population, but the cleaned files contain no weight column. Prevalence and model performance therefore describe this sample, not the U.S. adult population.
- **Undocumented row exclusion.** 441,456 records became 253,680 rows. The cleaning code was not readable without a Kaggle login. If rows with refused or missing answers were dropped, they were probably not missing at random, which could bias results.
- **Coarse target.**
  - Prediabetes and diabetes only during pregnancy are both coded 0.
  - Type 1 and type 2 diabetes are not distinguished.
  - Undiagnosed diabetes is necessarily labelled 0.
- **Coarse, heaped features.** Age and income are banded. MentHlth and PhysHlth are dominated by 0 (69% and 63% of rows) and show digit preference. The value 30 is the second or third most common, and 5, 10 and 15 are each more common than neighbouring values (see `results/figures/feature_distributions_by_class.png`).
- **Sensitive attributes.** UCI flags Sex, Income and Education as sensitive. Subgroup performance will be reported to check fairness.
- **No site or geography variable.** The file has no state or provider column, so **federated clients in this project are simulated partitions of a single national survey.** They are not hospitals and not real institutions.
- **Single year (2015).** Temporal generalization is not assessed.
- **Excluded data.** `online_retail.csv` is unrelated to health and is excluded from all healthcare modelling.
