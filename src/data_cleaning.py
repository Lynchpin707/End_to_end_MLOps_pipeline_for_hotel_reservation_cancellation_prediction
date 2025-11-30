import logging
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
class DataPreProcessStrategy(DataStrategy):
    """_summary_

    Args:
        DataStrategy (_type_): ABC 
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforms categorical data into numerical data and elects only the relevant features

        Args:
            df (pd.DataFrame): The initial df

        Raises:
            e: error occured during data preprocessing

        Returns:
            pd.DataFrame: preprocessed df
        """
        try:
            data["type_of_meal_plan"] = data["type_of_meal_plan"].map({"Meal Plan 1" : 1, "Meal Plan 2" : 2, "Meal Plan 3" : 3, "Not Selected":0 })
            
            data['room_type_reserved'] = data['room_type_reserved'].map({'Room_Type 1':1, 'Room_Type 2':2,'Room_Type 3':3,'Room_Type 4':4, 'Room_Type 5':5, 'Room_Type 6':6, 'Room_Type 7':7})
            
            data = pd.get_dummies(df, columns=['market_segment_type'], drop_first=False, dtype=int)
            
            data["booking_status"] = data["booking_status"].map({"Not_Canceled":0, "Canceled":1})
            
            data = data.drop(columns=['market_segment_type_Aviation', 'arrival_month', 'arrival_date'])
            return df
        except Exception as e:
            logging.error("Error occured during data preprocessing: {}".format(e))
            raise e

        
class DataDevideStrategy(DataStrategy):
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            y = data['booking_status']
            X = data.loc[:, data.columns != 'booking_status']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error occured while dividing data: {}".format(e))
            raise e
    
class DataCleaning:
    """
        Data cleaning class
    """
    def __init__(self, data:pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
    
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handles the data

        Returns:
            Union[pd.DataFrame, pd.Series]: _description_
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise(e)
        
