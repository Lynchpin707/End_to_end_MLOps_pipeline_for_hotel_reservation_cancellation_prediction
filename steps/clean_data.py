import logging
import pandas as pd
from zenml import step

from src.data_cleaning import DataCleaning, DataDevideStrategy, DataPreProcessStrategy

from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.DataFrame, "y_train"],
    Annotated[pd.DataFrame, "y_test"]
]:
    
    """_summary_

    Raises:
        e: _description_
    """
    try:
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, preprocess_strategy)
        processed_data = data_cleaning.handle_data()
        
        divide_strategy = DataDevideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed")
        
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e
    