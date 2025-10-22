from pandas import DataFrame

from src.constants import SURVIVED_HEADER, SEX_HEADER, AGE_HEADER, PCLASS_HEADER, SEX_ENCODED, PASSENGER_ID_HEADER


def message_training_data(titanic_df: DataFrame) -> DataFrame:
	titanic_features_df: DataFrame = titanic_df[
		[SURVIVED_HEADER, SEX_HEADER, AGE_HEADER, PCLASS_HEADER]].copy()
	return message_data(titanic_features_df)

def message_test_data(titanic_df: DataFrame) -> DataFrame:
	titanic_features_df: DataFrame = titanic_df[
		[PASSENGER_ID_HEADER, SEX_HEADER, AGE_HEADER, PCLASS_HEADER]].copy()
	return message_data(titanic_features_df)

def message_data(titanic_features_df: DataFrame) -> DataFrame:
	titanic_features_messaged_df = titanic_features_df.copy()
	titanic_features_messaged_df = encode_sex(titanic_features_messaged_df)
	titanic_features_messaged_df = fill_missing_age(titanic_features_messaged_df)
	return titanic_features_messaged_df

def encode_sex(titanic_features_df: DataFrame) -> DataFrame:
	titanic_features_with_encoded_sex_df = titanic_features_df.copy()
	titanic_features_with_encoded_sex_df[SEX_ENCODED] = titanic_features_df[SEX_HEADER].map({'male': 0, 'female': 1})
	return titanic_features_with_encoded_sex_df

def fill_missing_age(titanic_features_df: DataFrame) -> DataFrame:
	titanic_features_filled_ages_df = titanic_features_df.copy()
	titanic_features_filled_ages_df[AGE_HEADER] = titanic_features_df[AGE_HEADER].fillna(titanic_features_df[AGE_HEADER].median())
	return titanic_features_filled_ages_df
