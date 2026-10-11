# Titanic: Machine Learning from Disaster

Predicts which passengers survived the Titanic using EDA-driven feature
engineering and classification models, built for the
[Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).

**Status:** EDA and preprocessing complete. Modeling and ensembling in progress.

## Approach

1. **Data audit:** missing values, types, duplicates, summary statistics.
2. **Univariate analysis:** distributions of Age, Fare (heavily right-skewed),
   SibSp, Parch, and class balance of the target.
3. **Bivariate analysis:** survival rate by Sex, Pclass, Embarked, family size,
   Age and Fare.
4. **Multivariate analysis:** Sex x Pclass, Sex x Age group, and
   Embarked x Pclass x Sex, which showed that Embarked looks predictive mostly
   because of class mix.
5. **Preprocessing and feature engineering**, applied identically to train and
   test, with fill values learned from train rows only:
   - `Title` extracted from Name, rare titles grouped
   - `Age` filled by median of Title and Pclass, plus an `AgeMissing` flag
   - `CabinKnown` flag instead of imputing the 77% missing Cabin
   - `FamilySize` and grouped `FamilyGroup` (alone / small / large)
   - `TicketGroupSize` and grouped `TicketGroup` (passengers sharing a ticket)
   - `LogFare` and `FarePerPerson` to handle skew and shared fares
6. **Modeling (in progress):** repeated stratified cross-validation across
   logistic regression, random forest and gradient boosting, followed by an
   ensemble of diverse models.

## Key findings

- Sex and Pclass drive most of the signal: women in class 1 and 2 survived at
  roughly 92 to 97%, men in class 2 and 3 at roughly 14 to 16%.
- Family size is non-linear: groups of 2 to 4 did best, solo travelers and
  groups of 5 or more did worse.
- Missing Cabin is informative (about 67% survival when known vs 30% when
  missing).
- Large Fare values are real shared tickets, not errors, so no outliers were
  removed.

## Data leakage precautions

- Fill values (Age, Fare, Embarked) are computed from training rows only.
- `gender_submission.csv` is Kaggle's sample submission, not true labels, and
  is never used for training or validation.
- No target-derived features are built outside cross-validation.
- Model selection uses cross-validation, not repeated leaderboard feedback.

## Data

Download `train.csv` and `test.csv` from the
[Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/data)
and place them in a `data/` folder at the repo root.

## How to run

```bash
git clone https://github.com/YOUR_USERNAME/titanic-kaggle.git
cd titanic-kaggle
pip install -r requirements.txt
jupyter notebook notebooks/titanic.ipynb
```

Run all cells top to bottom. The notebook reads from `../data/` locally and
from the Kaggle input folder when run on Kaggle.

## Results

| Model | CV accuracy | Leaderboard |
|-------|-------------|-------------|
| TBD   | TBD         | TBD         |

## What I learned

- TBD (fill in after modeling)
