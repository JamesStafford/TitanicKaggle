import pandas as pd
from pandas import DataFrame
from typing import NamedTuple

from src.constants import SURVIVED_HEADER, AGE_HEADER, AGE_GROUP_HEADER

def create_age_to_survival_df(titanic_training_df: DataFrame) -> DataFrame:
    # y will be survived. Potential useful X are Survived, Sex and Age
    age_survival_df: DataFrame = titanic_training_df[[SURVIVED_HEADER, AGE_HEADER]].copy()

    age_bins_and_labels = create_age_bins_and_labels()

    age_survival_df.loc[:, AGE_GROUP_HEADER] = pd.cut(age_survival_df[AGE_HEADER], bins=age_bins_and_labels.age_bins[:-1], labels=age_bins_and_labels.age_bin_labels[:-1])
    age_survival_df.loc[:, AGE_GROUP_HEADER] = pd.Categorical(age_survival_df[AGE_GROUP_HEADER], categories=age_bins_and_labels.age_bin_labels[:-1])

    return age_survival_df

class AgeBinsAndLabels(NamedTuple):
    age_bins: list[int]
    age_bin_labels: list[str]

def create_age_bins_and_labels() -> AgeBinsAndLabels:
    age_range: int = 100
    step: int = 10
    age_bins: list[int] = list(range(0, age_range + step, step))
    age_bin_labels: list[str] = [f'{bin_index}-{bin_index + step - 1}' for bin_index in range(0, age_range, step)]

    return AgeBinsAndLabels(age_bins, age_bin_labels)