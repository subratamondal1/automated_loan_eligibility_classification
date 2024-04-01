from typing import Self

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from prediction_model.config import config

class MeanImputer(BaseEstimator, TransformerMixin):
    """
    Custom Data Transformer for Imputing Missing Values in Numerical Features.

    This transformer replaces missing values in numerical features with their respective
    mean values.

    Parameters
    ----------
    numerical_features : list of str
        List of column names in the input data that should be treated as numerical
        features and imputed using this transformer.

    Attributes
    ----------
    mean_dict_ : dict of str: float
        Dictionary containing the mean values for each numerical feature.
    """
    def __init__(self, numerical_features: list[str]) -> None:
        self.numerical_features: list[str] = numerical_features
        super().__init__()

    def fit(self, X, y=None) -> 'MeanImputer':
        """
        Learn the mean values for each numerical feature.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The input data containing the numerical features to be imputed.

        y : None
            Ignored in this transformer.

        Returns
        -------
        self : MeanImputer
            The fitted transformer object.
        """
        self.mean_dict_ = {}
        for col in self.numerical_features:
            self.mean_dict_[col] = X[col].mean()
        return self

    def transform(self, X) -> pd.DataFrame:
        """
        Replace missing values in each numerical feature with their respective mean values.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The input data containing the numerical features to be imputed.

        Returns
        -------
        X_imputed : pandas.DataFrame of shape (n_samples, n_features)
            The imputed input data with missing values replaced.
        """
        X = X.copy()
        for col in self.numerical_features:
            X[col].fillna(self.mean_dict_[col], inplace=True)
        return X

class ModeImputer(BaseEstimator, TransformerMixin):
    """
    Custom Data Transformer for Imputing Missing Values in Categorical Features.

    This transformer replaces missing values in categorical features with their respective
    mode values.

    Parameters
    ----------
    categorical_features : list of str
        List of column names in the input data that should be treated as categorical
        features and imputed using this transformer.

    Attributes
    ----------
    mode_dict_ : dict of str: Any
        Dictionary containing the mode values for each categorical feature.
    """
    def __init__(self, categorical_features: list[str]) -> None:
        self.categorical_features: list[str] = categorical_features
        super().__init__()

    def fit(self, X, y=None) -> 'ModeImputer':
        """
        Learn the mode values for each categorical feature.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The input data containing the categorical features to be imputed.

        y : None
            Ignored in this transformer.

        Returns
        -------
        self : ModeImputer
            The fitted transformer object.
        """
        self.mode_dict_ = {}
        for col in self.categorical_features:
            self.mode_dict_[col] = X[col].mode()[0]
        return self

    def transform(self, X) -> pd.DataFrame:
        """
        Replace missing values in each categorical feature with their respective mode values.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The input data containing the categorical features to be imputed.

        Returns
        -------
        X_imputed : pandas.DataFrame of shape (n_samples, n_features)
            The imputed input data with missing values replaced.
        """
        X = X.copy()
        for col in self.categorical_features:
            X[col].fillna(self.mode_dict_[col], inplace=True)
        return X

    
class DropColumns(BaseEstimator, TransformerMixin):
    """
    Custom Data Transformer for Dropping Columns.

    This transformer drops the specified columns from the input data.

    Parameters
    ----------
    columns_to_drop : list of str
        List of column names in the input data to be dropped.
    """
    def __init__(self, columns_to_drop: list[str]) -> None:
        self.columns_to_drop: list[str] = columns_to_drop
        super().__init__()

    def fit(self, X, y=None) -> 'DropColumns':
        """
        No need of learning and hence returning 'self'.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The input data containing the columns to be dropped.

        y : None
            Ignored in this transformer.

        Returns
        -------
        self : DropColumns
            The fitted transformer object.
        """
        return self

    def transform(self, X) -> pd.DataFrame:
        """
        Drop the specified columns from the input data.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The input data containing the columns to be dropped.

        Returns
        -------
        X_dropped : pandas.DataFrame of shape (n_samples, n_features - n_dropped)
            The input data with the specified columns dropped.
        """
        X = X.copy()
        X = X.drop(columns=self.columns_to_drop)
        return X

    
class CombineColumns(BaseEstimator, TransformerMixin):
    """
    Feature Engineering Transformer to combine two columns in the input data.

    This transformer adds the values of one column to another column, effectively
    combining their information into a single column.

    Parameters
    ----------
    columnA : str
        The name of the column in the input data to which the values of `columnB`
        will be added.

    columnB : str
        The name of the column in the input data whose values will be added to
        `columnA`.
    """
    def __init__(self, columnA: str, columnB: str) -> None:
        self.columnA: str = columnA
        self.columnB: str = columnB
        super().__init__()

    def fit(self, X, y=None) -> 'CombineColumns':
        """
        No need of learning and hence returning 'self'.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The input data containing the columns to be combined.

        y : None
            Ignored in this transformer.

        Returns
        -------
        self : CombineColumns
            The fitted transformer object.
        """
        return self

    def transform(self, X) -> pd.DataFrame:
        """
        Combine the values of `columnB` with `columnA` by adding them together.
        df[columnA] += df[columnB]

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The input data containing the columns to be combined.

        Returns
        -------
        X_combined : pandas.DataFrame of shape (n_samples, n_features)
            The combined input data with the updated values in `columnA`.
        """
        X = X.copy()
        X[self.columnA] += X[self.columnB]
        return X

    
class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Custom Label Encoder to encode categorical features to numerical features.

    This transformer replaces each unique categorical value with a numerical index,
    based on the sorted order of their frequency in the input data.

    Parameters
    ----------
    categorical_features : list of str
        List of column names in the input data that should be treated as categorical
        features and encoded using this transformer.

    Attributes
    ----------
    label_dict_ : dict of dict
        Mapping of each categorical feature to a dictionary of its unique values and
        their corresponding numerical indices.
"""
    def __init__(self, categorical_features: list[str]) -> None:
        self.categorical_features: list[str] = categorical_features
        super().__init__()

    def fit(self, X, y=None) -> 'CustomLabelEncoder':
        """
        Learn the mapping of each categorical feature to its unique values and their
        corresponding numerical indices.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The input data containing the categorical features to be encoded.

        y : None
            Ignored in this transformer.

        Returns
        -------
        self : CustomLabelEncoder
            The fitted transformer object.
        """
        self.label_dict_ = {}
        for col in self.categorical_features:
            t = X[col].value_counts().sort_values(ascending=True).index
            self.label_dict_[col] = {value: index for index, value in enumerate(iterable=t, start=0)}
        return self

    def transform(self, X) -> pd.DataFrame:
        """
        Replace each unique categorical value with its corresponding numerical index.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, n_features)
            The input data containing the categorical features to be encoded.

        Returns
        -------
        X_encoded : pandas.DataFrame of shape (n_samples, n_features)
            The encoded input data with numerical features.
        """
        X = X.copy()
        for col in self.categorical_features:
            X[col] = X[col].map(self.label_dict_[col])
        return X