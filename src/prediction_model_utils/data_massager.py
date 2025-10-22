from pandas import DataFrame

from src.constants import SURVIVED_HEADER, SEX_HEADER, AGE_HEADER, PCLASS_HEADER, SEX_ENCODED


def message_data(titanic_df: DataFrame) -> DataFrame:
	titanic_features_df: DataFrame = titanic_df[
		[SURVIVED_HEADER, SEX_HEADER, AGE_HEADER, PCLASS_HEADER]].copy()
	titanic_features_df[SEX_ENCODED] = titanic_features_df[SEX_HEADER].map({'male': 0, 'female': 1})
	titanic_features_df[AGE_HEADER] = titanic_features_df[AGE_HEADER].fillna(titanic_features_df[AGE_HEADER].median())
	return titanic_features_df
