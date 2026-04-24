# Adult Income Dataset Codebook

## Overview
The Adult (Census Income) dataset has 32,561 rows and 15 columns. The canonical task is to predict whether a person earns more than $50K per year based on demographic and employment attributes. The data come from the 1994 US Census.

## File
`data/adult.csv` — comma-separated, header row present.

## Columns

### age
Numeric. Age of the respondent in years.

### workclass
Categorical. Employment type. Values include Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. Contains some missing values encoded as `?`.

### fnlwgt
Numeric. Census "final weight" — an estimate of how many people this row represents in the US population. Usually not informative for prediction; often excluded.

### education
Categorical. Highest education level reached. Values include Bachelors, HS-grad, Some-college, Masters, Doctorate, and others.

### education_num
Numeric. Years-of-education encoding of `education`. Redundant with `education` — use one or the other.

### marital_status
Categorical. Marital status. Values include Never-married, Married-civ-spouse, Divorced, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.

### occupation
Categorical. Job category. Values include Tech-support, Craft-repair, Exec-managerial, Prof-specialty, Sales, and others. Contains `?` missing values.

### relationship
Categorical. Household role. Values include Wife, Husband, Own-child, Not-in-family, Unmarried, Other-relative.

### race
Categorical. Values: White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other.

### sex
Categorical. Values: Male, Female.

### capital_gain
Numeric. Reported capital gains in dollars. Highly skewed — most rows are 0.

### capital_loss
Numeric. Reported capital losses in dollars. Highly skewed — most rows are 0.

### hours_per_week
Numeric. Hours worked per week.

### native_country
Categorical. Country of origin. Dominated by United-States. Contains `?` missing values.

### income
Categorical target. Two values: `<=50K` and `>50K`. This is the prediction target.

## Cautions
- `?` is used to encode missing values in `workclass`, `occupation`, and `native_country`. Convert these to NaN before analysis.
- `education` and `education_num` encode the same information.
- `fnlwgt` is a sampling weight, not a predictor.
- The target is binary — use classification models. For regression tools, pick a numeric outcome like `hours_per_week` or `age` instead.
