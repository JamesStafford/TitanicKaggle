from typing import Any

from pandas import DataFrame, Series

from src.constants import SEX_ENCODED, AGE_HEADER, PCLASS_HEADER, SURVIVED_HEADER


def get_x(titanic_features_df: DataFrame) -> DataFrame:
	return titanic_features_df[[SEX_ENCODED, AGE_HEADER, PCLASS_HEADER]]

def get_y(titanic_features_df: DataFrame) -> Series[Any]:
	return titanic_features_df[SURVIVED_HEADER]
