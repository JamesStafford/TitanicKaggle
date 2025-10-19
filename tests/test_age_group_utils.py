from typing import Any

import pytest
import pandas as pd
from pandas import DataFrame, Series

from src.constants import AGE_GROUP_HEADER, AGE_HEADER, SURVIVED_HEADER
from src.age_group_survival.age_group_utils import create_age_to_survival_df


@pytest.fixture
def titanic_training_df() -> DataFrame:
	titanic_training_df_full: DataFrame = pd.read_csv("data/train.csv")  # type: ignore
	return titanic_training_df_full.head()


class TestCreateAgeToSurvival:
	def test_create_age_to_survival_df_adds_age_group_header_column(
		self, titanic_training_df: DataFrame
	) -> None:
		result: DataFrame = create_age_to_survival_df(titanic_training_df)
		assert AGE_GROUP_HEADER in result.columns

	def test_create_age_to_survival_df_contains_survived_column(
		self, titanic_training_df: DataFrame
	) -> None:
		result: DataFrame = create_age_to_survival_df(titanic_training_df)
		assert SURVIVED_HEADER in result.columns

	def test_create_age_to_survival_df_contains_age_column(
		self, titanic_training_df: DataFrame
	) -> None:
		result: DataFrame = create_age_to_survival_df(titanic_training_df)
		assert AGE_HEADER in result.columns

	def test_create_age_to_survival_df_contains_20_to_29_age_group(
		self, titanic_training_df: DataFrame
	) -> None:
		result: DataFrame = create_age_to_survival_df(titanic_training_df)

		age_group_20_to_29 = result[(result[AGE_GROUP_HEADER] == "20-29")]
		assert not age_group_20_to_29.empty, "No rows found with age group of '20-29'"

	def test_create_age_to_survival_df_contains_30_to_39_age_group(
		self, titanic_training_df: DataFrame
	) -> None:
		result: DataFrame = create_age_to_survival_df(titanic_training_df)

		age_group_30_to_39 = result[(result[AGE_GROUP_HEADER] == "30-39")]
		assert not age_group_30_to_39.empty, "No rows found with age group of '30-39'"

	def test_create_age_to_survival_df_contains_20_to_29_age_group_and_age_in_group(
		self, titanic_training_df: DataFrame
	) -> None:
		result: DataFrame = create_age_to_survival_df(titanic_training_df)

		age_group_20_to_29: DataFrame = result[(result[AGE_GROUP_HEADER] == "20-29")]
		assert not age_group_20_to_29.empty, "No rows found with age group of '20-29'"

		first_item: Series[Any] = age_group_20_to_29.iloc[0]
		age_value: float = first_item[AGE_HEADER] #type: ignore
		assert 20 <= age_value <= 29, f"Age {age_value} is not in range 20-29"

	def test_create_age_to_survival_df_contains_30_to_39_age_group_and_age_in_group(
		self, titanic_training_df: DataFrame
	) -> None:
		result: DataFrame = create_age_to_survival_df(titanic_training_df)

		age_group_30_to_39: DataFrame = result[(result[AGE_GROUP_HEADER] == "30-39")]
		assert not age_group_30_to_39.empty, "No rows found with age group of '30-39'"

		first_item: Series[Any] = age_group_30_to_39.iloc[0]
		age_value: float = first_item[AGE_HEADER] #type: ignore
		assert 30 <= age_value <= 39, f"Age {age_value} is not in range 30-39"
