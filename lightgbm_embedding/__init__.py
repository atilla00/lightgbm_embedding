__version__ = "0.1.2"

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.special import expit
import pandas as pd
import multiprocessing as mp

__all__ = ["LightgbmEmbedding"]


def LightgbmFactory(target_type, n_dim=100, args={}):
    """LightGBM Factory"""

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
    """
    LightGBM Feature Embeddings class. Leaf values for each trees are used for embedding.


    Parameters
    -----------
    n_dim: int, default = 100
        Number of dimensions to use. *Note: For multiclass problems dimension will be n_dim*num_classes

    **lgb_kwargs: dict, default = {}
        LightGBM arguments in dictionary format


    Notes
    ----------
    Scikit-learn util ``type_of_target`` fails at identifying regression task with array of integers.
    If that's the case, provide target type in fit method ``fit(X, y, target_type='continuous')``.
     For scikit-learn pipelines pipeline.fit(X, y, model__target_type='continuous'))


    Examples
    -----------
    >>> from lightgbm_embedding import LightgbmEmbedding
    >>> from sklearn.datasets import load_breast_cancer
    >>> X, y = load_breast_cancer(return_X_y=True)

    >>> embedder = LightgbmEmbedding().fit(X, y)
    >>> emb.transform(X)
    """

    def __init__(self, n_dim=100, lgb_kwargs={}):
        if "n_estimators" in lgb_kwargs:
            print(
                f"Ignoring n_estimators in lgb_kwargs because n_dim is provided. n_dim={n_dim}"
            )

            lgb_kwargs["n_estimators"] = n_dim

        self.n_dim = n_dim
        self.lgb_kwargs = lgb_kwargs
        self._is_fit = False

    def fit(self, X, y, target_type=None):
        """
        Fit LightGBM Embedder.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            2-D (n_samples, n_features) feature array.

        y : Union[pd.Series, np.ndarray]
            1-D target array.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if target_type is not None:
            self.target_type = target_type
        else:
            self.target_type = type_of_target(y)

        self._model = LightgbmFactory(
            self.target_type, n_dim=self.n_dim, args=self.lgb_kwargs
        )

        if self.target_type is "multiclass":
            self.n_dim = self.n_dim * y.nunique()
            print("Target is multiclass. Setting n_dim = n_dim * num_classes.")

        self._model.fit(X, y)
        self._booster = self._model._Booster

        # self._tree_dict = self._get_tree_leaf_value_dict()
        self._leaf_hash_map = self._get_tree_leaf_value_dict_list()
        self._is_fit = True

        return self

    def _get_leaf_value(self, row: list):
        """Get leaf values from row of leaf index. Input row shape (n_trees,)"""
        embed = [
            self._tree_dict[f"{tree_id}_{leaf_id}"]
            for tree_id, leaf_id in enumerate(row)
        ]

        return embed

    def transform(self, X, y=None, n_jobs=1):
        """
        Get embeddings from fitted estimator.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            2-D (n_samples, n_features) feature array.

        y : None
            Exists for compatibility with pipelines.

        Returns
        ----------
        X_embeddings : pd.DataFrame
            Embeddings dataframe with shape (n_samples, n_dim)

        """

        check_is_fitted(self)

        preds = self._model.predict(X, pred_leaf=True)
        preds = pd.DataFrame(preds)

        for col in preds.columns:
            preds[col] = preds[col].map(self._leaf_hash_map[col])

        preds.columns = [f"dim_{i}" for i in range(self.n_dim)]

        if self.target_type in ["binary", "multiclass"]:
            preds = expit(preds)

        return preds

    def _get_tree_leaf_value_dict_list(self):
        """Returns list with [index:tree_id, values:leaf values]"""
        tree_df = self._booster.trees_to_dataframe()[
            ["tree_index", "node_index", "value", "weight", "count"]
        ]

        # Ignore not terminal nodes to leave out only leaves
        tree_df["node_index"] = (
            tree_df["node_index"]
            .str.replace(r"\d+-L(\d+)", "\\1", regex=True)
            .replace(r"\d+-S(\d+)", np.nan, regex=True)
        )

        tree_df = tree_df[tree_df["node_index"].notnull()]
        tree_df["node_index"] = tree_df["node_index"].apply(int)

        hash_map = []

        for i in range(100):
            tmp = tree_df[tree_df["tree_index"] == i].reset_index(drop=True)
            hash_map.append(pd.Series(tmp["value"], index=tmp["node_index"]).to_dict())

        return hash_map

    def _get_tree_leaf_value_dict(self):
        """Returns leaf values for in dictionary format. Type: List[f"{tree_id}_{leaf_id}", leaf_value]"""
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

    def __sklearn_is_fitted__(self):
        """Sklearn util method for is_fitted validation."""
        return True if self._is_fit else False
