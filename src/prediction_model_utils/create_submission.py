from typing import Any

from pandas import DataFrame

from src.constants import PASSENGER_ID_HEADER


def create_submission_csv(titanic_df: DataFrame, predictions: Any) -> DataFrame:
	random_forest_submission_df = DataFrame({
		'PassengerId': titanic_df[PASSENGER_ID_HEADER],
		'Survived': predictions
	})

	random_forest_submission_df.to_csv('../data/random_forest_submission.csv', index=False)

	return random_forest_submission_df
