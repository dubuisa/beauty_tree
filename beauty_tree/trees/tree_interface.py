from abc import ABC, abstractmethod
import numpy as np
from typing import List, Mapping

class ITree(ABC):
    
    def __init__(self,
                 tree_model,
                 feature_names: List[str] = None,
                 class_names: List[str] = None,
                 target_name: str = None,
                 tree_index : int =  None
                 ):
        
        if not self.is_fit(tree_model):
            raise ValueError(f"Model {tree_model} is not fitted")
        
        self.tree_model = tree_model
        self.feature_names = feature_names
        self.class_names = class_names
        self.tree_index = tree_index
        self.target_name = target_name
        
        
    
    @abstractmethod
    def is_fit(self) -> bool:
        pass
    
    
    @abstractmethod
    def is_classifier(self) -> bool:
        """Checks if the tree model is a classifier."""
        pass
    
    @abstractmethod
    def is_regressor(self) -> bool:
        """Checks if the tree model is a regressor."""
        pass
    
    @abstractmethod
    def to_df(self) -> bool:
        pass
    
    @abstractmethod
    def get_features(self) -> np.ndarray:
        """Returns feature indexes for tree's nodes.
        Ex. features[i] holds the feature index to split on
        """
        pass

    @abstractmethod
    def criterion(self) -> str:
        """Returns the function to measure the quality of a split.
        Ex. Gini, entropy, MSE, MAE
        """
        pass
    
    @abstractmethod
    def nclasses(self) -> int:
        """Returns the number of classes.
        Ex. 2 for binary classification or 1 for regression.
        """
        pass
    
    @abstractmethod
    def classes(self) -> np.ndarray:
        """Returns the tree's classes values in case of classification.
        Ex. [0,1] in class of a binary classification
        """
        pass
 
    
    def is_categorical_split(self, id) -> bool:
        """Checks if the node split is a categorical one.
        This method needs to be overloaded only for shadow tree implementation which contain categorical splits,
        like Spark.
        """
        return False