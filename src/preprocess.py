import configparser
import pandas as pd
from typing import Tuple
from logger import Logger
from sklearn.model_selection import train_test_split

SHOW_LOG = True

class DataProcessor:
    """A class for loading and preprocessing text data."""
    
    def __init__(self, dataset_path: str, target_column: str = 'category'):
        """
        Initialization of the data handler.
        Automatically reads the dataset along the specified path.
        """
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.df = pd.read_csv(self.dataset_path)
    
    def split(self, value_column: str = 'question', test_size: float = 0.2) -> Tuple:
        """Divides the dataset into training and test samples.

        Args:
            value_column (str): columns of features.
            test_size (float): percentage of the test sample.

        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        X = self.df[value_column]
        y = self.df[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        return X_train, X_test, y_train, y_test
        
    def balance(self, min_examples: int = 50) -> None:
        """Removes rare classes from the dataset.

        Args:
            min_examples (int): The minimum number of examples per class.
        """
        category_counts = self.df[self.target_column].value_counts()
        valid_categories = category_counts[category_counts >= min_examples].index
        self.df = self.df[self.df[self.target_column].isin(valid_categories)].copy()

    def get_top_n_categories(self, top_n: int = 5) -> None:
        """Leaves only the top N most popular categories in the dataset.

        Args:
            top_n (int): The number of categories to save.
        """
        top_categories = self.df[self.target_column].value_counts().head(top_n).index
        self.df = self.df[self.df[self.target_column].isin(top_categories)].copy()