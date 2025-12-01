import logging
from abc import ABC, abstractmethod

from sklearn.metrics import accuracy_score, f1_score, log_loss

import numpy as np

class Evaluation(ABC):
    """Abstract class defining strategy evaluating our models
    """
    
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Calculates score for our model

        Args:
            y_true (np.ndarray): Real labels
            y_pred (np.ndarray): Predicted labels
        """
        pass
    
class AccuracyScore(Evaluation):
    """ Evaluation strategy that uses the accuracy score"""
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating the accuracy score")
            accuracy = accuracy_score(y_true, y_pred)
            logging.info(f"Accuracy score : {accuracy}")
            return accuracy
        except Exception as e:
            logging.error("Error occured while calculating the accuracy score: {}".format(e))
            raise e

class F1Score(Evaluation):  
    """ Evaluation strategy that uses the F1 score"""    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating the F1 score")
            f1_micro = f1_score(y_true, y_pred, average='micro')
            logging.info(f"F1 score : {f1_micro}")
            return f1_micro
        except Exception as e:
            logging.error("Error occured while calculating the f1 score: {}".format(e))
            raise e
            
class CrossEntropyLoss(Evaluation):  
    """ Evaluation strategy that uses Cross-Entropy Loss"""    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating the Cross-Entropy Loss")
            f1_micro = log_loss(y_true, y_pred)
            logging.info(f"Cross-Entropy Loss : {f1_micro}")
        except Exception as e:
            logging.error("Error occured while calculating the cross entropy loss: {}".format(e))
            raise e