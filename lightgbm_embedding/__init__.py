__version__ = "0.1.0"


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import type_of_target
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.special import expit
import pandas as pd
import warnings


def EstimatorFactory(target_type, n_dim=100, args={}):
    """Factory Function"""

    if target_type is "binary":
        model = LGBMClassifier(n_estimators=n_dim, **args)
    elif target_type is "multiclass":
        model = LGBMClassifier(n_estimators=n_dim, **args)
    elif target_type is "continuous":
        model = LGBMRegressor(n_estimators=n_dim, **args)
    else:
        raise ValueError(f"Target type: {target_type} is not supported.")

    return model


class LightgbmEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, n_dim=100, **lgb_kwargs):
        if ("n_estimators" in lgb_kwargs) and (n_dim is not None):
            warnings.warn(
                f"Ignoring n_estimators in lgb_kwargs because n_dim is provided. n_dim={n_dim}"
            )
            del lgb_kwargs["n_estimators"]

        self.n_dim = n_dim
        self.lgb_kwargs = lgb_kwargs

    def _handle_object_type(self, X):
        for col in X.columns:
            if X[col].dtype.name in ["object", "string"]:
                X[col] = X[col].astype("category")

        return X

    def fit(self, X, y, target_type=None):
        # X = self._handle_object_type(X)

        if target_type is not None:
            self.target_type = target_type
        else:
            self.target_type = type_of_target(y)

        self._model = EstimatorFactory(
            self.target_type, n_dim=self.n_dim, args=self.lgb_kwargs
        )

        if self.target_type is "multiclass":
            self.n_dim = self.n_dim * y.nunique()
            print("Target is multiclass. Setting n_dim = n_dim * num_classes.")

        self._model.fit(X, y)
        self._booster = self._model._Booster

        self._tree_dict = self._get_tree_leaf_value_dict()

    def transform(self, X, y=None):

        preds = self._model.predict(X, pred_leaf=True)

        embeds = []
        for pred in preds:
            embed = [
                self._tree_dict[f"{tree_id}_{leaf_id}"]
                for tree_id, leaf_id in enumerate(pred)
            ]

            if self.target_type in ["binary", "multiclass"]:
                embed = expit(embed)
            embeds.append(embed)

        return pd.DataFrame(embeds, columns=[f"dim_{i}" for i in range(self.n_dim)])

    def _get_tree_leaf_value_dict(self):
        tree_df = self._booster.trees_to_dataframe()[
            ["tree_index", "node_index", "value", "weight", "count"]
        ]

        # Ignore not terminal nodes to leave out only leaves
        tree_df["node_index"] = (
            tree_df["node_index"]
            .str.replace(r"\d+-L(\d+)", "\\1", regex=True)
            .replace(r"\d+-S(\d+)", np.nan, regex=True)
        )

        # Create index with treeid_nodeid
        tree_df = tree_df[tree_df["node_index"].notnull()]
        tree_df.index = (
            tree_df["tree_index"].apply(str) + "_" + tree_df["node_index"].apply(str)
        )

        # To be able to use to_dict directly
        tree_df = tree_df["value"]
        tree_dict = tree_df.to_dict()

        return tree_dict
