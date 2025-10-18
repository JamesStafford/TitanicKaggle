import pytest
import pandas as pd
from pandas import DataFrame

from src.constants import NAME_HEADER, PASSENGER_ID_HEADER, PCLASS_HEADER, SURVIVED_HEADER, AGE_HEADER, AGE_GROUP_HEADER
from src.age_group_survival.age_group_utils import create_age_to_survival_df, create_age_bins_and_labels, AgeBinsAndLabels

@pytest.fixture
def titanic_training_df() -> pd.DataFrame:
    titanic_training_df_full: pd.DataFrame = pd.read_csv("data/train.csv")
    return titanic_training_df_full.head()

class TestCreateAgeToSurvival:
    def test_create_age_to_survival_df_adds_new_column(self, titanic_training_df):
        result: DataFrame = create_age_to_survival_df(titanic_training_df)
        assert AGE_GROUP_HEADER in result.columns

