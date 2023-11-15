import logging
import pandas as pd

from zenml import step



@step
def train_model(df: pd.DataFrame) -> None:
    """
    This step trains a model.
    :param df:
    :return:
    """
    pass