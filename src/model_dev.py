import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for model.
    """
    @abstractmethod
    def train(self, X_train, y_train):
        pass

# class LinearRegressionModel(Model):
#     """
#     Linear Regression Model.
#     """
#     def __init__(self):
#         self.model = LinearRegressionModel()
#
#     def train(self, X_train, y_train, **kwargs):
#
#
#         try:
#             reg = LinearRegression(**kwargs)
#             reg.fit(X_train, y_train)
#             logging.info(f"Model Training Complete")
#
#             return reg
#         except Exception as e:
#             logging.error(f"Error Training Model")
#             raise e

class LinearRegressionModel(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        reg = LinearRegression(**kwargs)
        reg.fit(x_train, y_train)
        return reg

    # For linear regression, there might not be hyperparameters that we want to tune, so we can simply return the score
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        reg = self.train(x_train, y_train)
        return reg.score(x_test, y_test)