from sklearn.base import is_classifier, is_regressor
from lightgbm.basic import Booster
from sklearn.tree._tree import Tree
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm as cmx

import plotly.graph_objects as go

import re

from functools import singledispatchmethod


def _get_cmap(tx_cmap="RdYlGn", vmin=0, vmax=1):
    print(vmin)
    cmap = plt.get_cmap(tx_cmap)
    cNorm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    return cmap


def _to_decision(
    node_index,
    split_feature,
    decision_type,
    threshold,
    value,
    is_leaf,
    label="Accepté",
    label_option="value",
):
    if is_leaf:
        return {
            "full": f"{node_index} : {value*100:.2f}% {label}",
            "value": f"{value}%",
        }.get(label_option, "")
    if decision_type == '==':
        return f'{split_feature} IN ({threshold.replace("||", ",")})'
    if threshold > 10e10:
        threshold = 'INF'
    else:
         threshold = f'{threshold:.2f}'
    return f"{split_feature} {decision_type} {threshold}"


def _get_link_color(node_index, to, is_leaf, lookup=None):
    idx = to
    if is_leaf:
        idx = node_index
    return lookup[lookup["node_index"] == idx].iloc[0].color


def _val_to_color(values, cmap):
    to_rgb = (
        lambda r, g, b, a: f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a:.2})"
    )
    return [to_rgb(r, g, b, a) for r, g, b, a in cmap.to_rgba(values, alpha=0.5)]


def _plot(nodes, connections):
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=30,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=nodes["condition"].to_list(),
                    color=nodes["color"].to_list(),
                ),
                link=dict(
                    source=connections["node_index"].to_list(),
                    target=connections["to"].to_list(),
                    value=connections["count"].to_list(),
                    color=connections["color_link"].to_list(),
                ),
            )
        ]
    )

    fig.update_layout(title_text="Churn segmentation", font_size=12,
                      #width=2000,height=1000
                      )

    fig.show()
    fig.write_image("churn.jpeg")

def _tree_to_viz(df, label_option="value", tx_cmap="RdYlGn"):

    nodes = (
        # df2.assign(node_pos=lambda X: X["node_index"].str.extract(r"\d+\-[A-Z](\d+)$"))
        # .sort_values(by=["node_depth", "node_pos"])
        df.sort_values(by=["node_depth"])
        #.assign(value=lambda X: 1 / (1 + 1 / np.exp(X["value"])))

        .assign(
            condition=lambda X: X[
                ["node_index", "split_feature", "decision_type", "threshold", "value", "is_leaf"]
            ].apply(lambda x: _to_decision(*x, label_option=label_option), axis=1)
        )
        .reset_index()
    )
    mapping_nodes_idx = dict(zip(nodes["node_index"].to_list(), range(len(nodes))))

    nodes = nodes.assign(
        node_index=lambda X: X["node_index"].map(mapping_nodes_idx),
        left_child=lambda X: X["left_child"].map(mapping_nodes_idx),
        right_child=lambda X: X["right_child"].map(mapping_nodes_idx),
        parent_index=lambda X: X["parent_index"].map(mapping_nodes_idx),
        color=lambda X: _val_to_color(X["value"], _get_cmap(tx_cmap, vmin=min(X['value']))),
    )
    conn = (
        nodes.melt(
            id_vars=["node_index", "count", "condition", "is_leaf"],
            value_vars=["left_child", "right_child"],
            value_name="to",
        )
        .dropna()
        .assign(
            color_link=lambda X: X[["node_index", "to", "is_leaf"]].apply(
                lambda x: _get_link_color(*x, lookup=nodes), axis=1
            )
        )
    )
    conn = (
        nodes[["parent_index", "node_index", "count"]]
        .merge(
            conn.drop(columns=["count"]),
            how="left",
            left_on=["parent_index", "node_index"],
            right_on=["node_index", "to"],
        )
        .dropna(subset=["parent_index"])
        .rename(columns={"node_index_y": "node_index"})
    )
    _plot(nodes, conn)
    return nodes, conn



class BeautyTree:
    
    def __init__(self):
        pass
    
    def render(self, model, tree_idx=0, features_name=[], label_option="acc", tx_cmap="RdYlGn"):
        tree = model
        if is_classifier(model) or is_regressor(model):
            if hasattr(model, 'estimators_'):
                tree = model.estimators_[tree_idx].tree_
            else:
                tree = model.tree_
                
        df = self._tree_to_df(tree, features_name)
        nodes, connections = _tree_to_viz(df, label_option=label_option, tx_cmap=tx_cmap)
        
        return nodes, connections
    
    
    @singledispatchmethod
    def _tree_to_df(self, arg):
        raise NotImplementedError(f"Cannot handle type: {type(arg)}")
    
    
    @_tree_to_df.register
    def _(self, arg: Booster, features_name):
        return (
            Booster.trees_to_dataframe(arg)
                .fillna(value=np.nan)
                .assign(is_leaf = lambda X: X["split_feature"].isna())
        )


    @_tree_to_df.register
    def _(self, arg: Tree, features_name):
        if not features_name:
            raise ValueError("You must provide a list of features_name for scikit-learn trees!")
        
        n_nodes = arg.node_count
        children_left = arg.children_left
        children_right = arg.children_right
        feature = arg.feature
        threshold = arg.threshold
        n_node_samples = arg.n_node_samples
        #impurity = arg.impurity
        values = arg.value[:].reshape(n_nodes, -1)
        features = dict(zip(np.arange(len(features_name)), features_name))

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
    



        
