import logging

import pandas as pd
from zenml import step


@step
def clean_df(df: pd.DataFrame) -> None:
    """
    This step cleans the data.
    :param df:
    :return:
    """
    pass
