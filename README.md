# Titanic: Machine Learning from Disaster

Predicts which passengers survived the Titanic using EDA-driven feature
engineering and an ensemble of classification models, built for the
[Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).

**Leaderboard score: 0.78229** | **Cross-validation accuracy: 0.834**

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
   - `Age` filled by median of Title and Pclass
   - `CabinKnown` flag instead of imputing the 77% missing Cabin
   - `TicketGroupSize` and grouped `TicketGroup` (passengers sharing a ticket)
   - `LogFare` and `FarePerPerson` to handle skew and shared fares
6. **Feature selection by ablation:** removed Embarked, FamilySize,
   FamilyGroup and AgeMissing after repeated CV showed they added nothing.
   Going from 23 to 15 features scored the same or better on all five models.
7. **Modeling:** repeated stratified 5-fold CV (5 repeats) across logistic
   regression, SVC, random forest, extra trees and gradient boosting, followed
   by light grid-search tuning.
8. **Ensemble:** soft-voting classifier over tuned LogReg, SVC, RF and GBM.

## Key findings

- Sex and Pclass drive most of the signal: women in class 1 and 2 survived at
  roughly 92 to 97%, men in class 2 and 3 at roughly 14 to 16%.
- Family size is non-linear: groups of 2 to 4 did best, solo travelers and
  groups of 5 or more did worse. TicketGroupSize captured this better than
  SibSp/Parch because it also includes friends and servants traveling together.
- Missing Cabin is informative (about 67% survival when known vs 30% when
  missing).
- Large Fare values are real shared tickets, not errors, so no outliers were
  removed.
- A smaller feature set generalized as well as or better than the full one.

## Data leakage precautions

- Fill values (Age, Fare, Embarked) are computed from training rows only.
- `gender_submission.csv` is Kaggle's sample submission, not true labels, and
  is never used for training or validation.
- No target-derived features are built outside cross-validation.
- Model selection uses cross-validation, not repeated leaderboard feedback.

## Results

| Model | CV accuracy |
|-------|-------------|
| Logistic Regression (tuned) | 0.8294 |
| SVC (tuned) | 0.8334 |
| Random Forest (tuned) | 0.8386 |
| Gradient Boosting (tuned) | 0.8343 |
| **Ensemble (soft voting)** | **0.8339** |

Leaderboard score of the ensemble: **0.78229**. The gap between CV and
leaderboard is typical for Titanic: the test set has only 418 passengers, and
families split across folds make CV slightly optimistic.

## Repository structure

```
titanic-kaggle/
├── data/                 # not tracked, see Data section
├── notebooks/
│   └── titanic-1.ipynb     # EDA, preprocessing, modeling
├── README.md
├── requirements.txt
└── .gitignore
```

## Data

Download `train.csv` and `test.csv` from the
[Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic/data)
and place them in a `data/` folder at the repo root.

## How to run

```bash
git clone https://github.com/omaryan820-ux/titanic-Kaggle.git
cd titanic-Kaggle
pip install -r requirements.txt
jupyter notebook notebooks/titanic.ipynb
```

Run all cells top to bottom. The notebook reads from `../data/` locally and
from the Kaggle input folder when run on Kaggle.

## What I learned

- Judge changes by repeated cross-validation, not the leaderboard: with 418
  test rows, one passenger is about 0.24%.
- Fewer, well-chosen features beat many features. Ablation showed which ones
  earned their place.
- Interaction-heavy problems like this one are about features (Title, ticket
  groups) more than model tuning.
- Preprocessing must be built so train and test are transformed identically,
  with fill values learned from train only.
