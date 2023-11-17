import logging
from abc import ABC, abstractmethod
import numpy as np
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Evaluation(ABC):
    """
    Abstract class for evaluation.
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass

class MSE(Evaluation):
    """
    Mean Squared Error.
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        This step calculates the MSE.
        :param y_true:
        :param y_pred:
        :return:
        """
        try:
            logging.info(f"Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error calculating MSE")
            raise e

class R2(Evaluation):
    """
    R2 Score.
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        This step calculates the R2.
        :param y_true:
        :param y_pred:
        :return:
        """
        try:
            logging.info(f"Calculating R2")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error calculating R2")
            raise e

# class MAE(Evaluation):
#     """
#     Mean Absolute Error.
#     """
#     def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
#         """
#         This step calculates the MAE.
#         :param y_true:
#         :param y_pred:
#         :return:
#         """
#         try:
#             logging.info(f"Calculating MAE")
#             mae = mean_absolute_error(y_true, y_pred)
#             logging.info(f"MAE: {mae}")
#             return mae
#         except Exception as e:
#             logging.error(f"Error calculating MAE")
#             raise e

class RMSE(Evaluation):
    """
    Root Mean Squared Error.
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        This step calculates the RMSE.
        :param y_true:
        :param y_pred:
        :return:
        """
        try:
            logging.info(f"Calculating RMSE")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error calculating RMSE")
            raise e
