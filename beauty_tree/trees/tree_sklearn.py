from .tree_interface import ITree
from sklearn import is_classifier, is_regressor
import numpy as np
import pandas as pd


from typing import List, Mapping

class SklearnTree(ITree):

    def __init__(self,
                 tree_model,
                 feature_names: List[str] = None,
                 class_names: List[str] = None,
                 target_name: str = None,
                 tree_index : int =  None):
         super().__init__(tree_model, feature_names, target_name, class_names, tree_index)
         self.tree_ = self.__get_tree()


    def is_fit(self) -> bool:
        return getattr(self.tree_model, 'tree_') is not None
    
    
    def is_classifier(self) -> bool:
        return is_classifier(self.tree_)
    
    
    def is_regressor(self) -> bool:
        return is_regressor(self.tree_) 
    
    
    
    def get_features(self) -> np.ndarray:
        return self.tree_.feature


    def criterion(self) -> str:
        return self.tree_model.criterion.upper()

    
    def nclasses(self) -> int:
        return self.tree_.n_classes[0]


    def classes(self) -> np.ndarray:
        if self.is_classifier():
            return self.tree_model.classes_

    
    def to_df(self) -> bool:
        if not self.feature_names:
            raise ValueError("You must provide a list of features_name for scikit-learn trees!")
        
        n_nodes = self.tree_.node_count
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right
        feature = self.tree_.feature
        threshold = self.tree_.threshold
        n_node_samples = self.tree_.n_node_samples
        values = self.tree_.value[:].reshape(n_nodes, -1)
        features = dict(zip(np.arange(len(self.feature_names)), self.feature_names))

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)

        parents = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0, -1)]
        while len(stack) > 0:
            
            node_id, depth, parent = stack.pop()
            node_depth[node_id] = depth
            parents[node_id] = parent
            
            
            is_split_node = children_left[node_id] != children_right[node_id]
            
            if is_split_node:
                stack.append((children_left[node_id], depth + 1, node_id))
                stack.append((children_right[node_id], depth + 1, node_id))
            else:
                is_leaves[node_id] = True


        elements = []
        for i in range(n_nodes):

            elements.append(
                {
                    "node_index": i,
                    "node_depth": node_depth[i],
                    "left_child": children_left[i],
                    "right_child": children_right[i],
                    "split_feature": features.get(feature[i], np.nan),
                    "decision_type": "<=",
                    "threshold": threshold[i],
                    "count": n_node_samples[i],
                    "value": np.max(values[i])/np.sum(values[i]),
                    "is_leaf": is_leaves[i],
                    "parent_index": parents[i],
                }
            )
            
        df = pd.DataFrame.from_records(elements).assign(
            left_child=lambda X: X["left_child"].replace(-1, np.nan),
            right_child=lambda X: X["right_child"].replace(-1, np.nan),
            parent_index=lambda X: X["parent_index"].replace(-1, np.nan),
        )
        df.loc[df["split_feature"].isna(), ["decision_type", "threshold"]] = (np.nan, np.nan)
        return df 
    
    
    def __get_tree(self):
        if hasattr(self.tree_model, 'estimators_'):
            return self.tree_model.estimators_[self.tree_idx].tree_
        return self.tree_model.tree_