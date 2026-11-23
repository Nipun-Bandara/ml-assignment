"""Shared helpers for the notebook and the Streamlit dashboard (so saved joblib models load in both)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

# The models were saved with scikit-learn 1.6. Newer versions (needed on Python 3.14, e.g. Streamlit Cloud)
# removed a small helper class that the saved ColumnTransformers refer to, so I add it back before loading.
# Checked: every model gives exactly the same test predictions with this in place.
from collections import UserList
import sklearn.compose._column_transformer as _ct
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(UserList):
        pass
    _ct._RemainderColsList = _RemainderColsList

CAT_COLS = ["market", "admin1", "admin2", "category", "commodity", "unit"]
NUM_COLS = ["latitude", "longitude", "lag_1", "lag_2", "lag_3", "roll_mean_3", "roll_std_3", "pct_change_1",
            "nat_median_lag1", "rel_to_national", "nat_change_1", "ly_change", "lag1_vs_roll",
            "months_since_last", "month_num", "month_sin", "month_cos"]
FEATURES = CAT_COLS + NUM_COLS


def make_target(df):
    """Target = log of (this month's price / last month's price): the model predicts the % change."""
    return np.log(df["price"].values / df["lag_1"].values)


def to_price(pred, df):
    """Convert a predicted log-change back to a price in LKR."""
    return df["lag_1"].values * np.exp(pred)


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred),
            "MAPE_%": np.mean(np.abs(y_true - y_pred) / y_true) * 100}


def set_categories(df, categories):
    """Give categorical columns a fixed category list so train, test and dashboard inputs match."""
    df = df.copy()
    for c in CAT_COLS:
        df[c] = pd.Categorical(df[c].astype(str), categories=categories[c])
    return df


class CatBoostWrap(RegressorMixin, BaseEstimator):
    """Thin sklearn wrapper: CatBoost changes `cat_features` internally, which breaks sklearn's clone()
    inside StackingRegressor. This wrapper keeps the parameters untouched."""

    def __init__(self, cat_features=None, iterations=800, learning_rate=0.05, depth=6,
                 l2_leaf_reg=3.0, loss_function="RMSE", random_state=42):
        self.cat_features = cat_features
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.loss_function = loss_function
        self.random_state = random_state

    def _prep(self, X):
        X = X.copy()
        for c in self.cat_features or []:
            X[c] = X[c].astype(str)
        return X

    def fit(self, X, y):
        from catboost import CatBoostRegressor
        self.model_ = CatBoostRegressor(iterations=self.iterations, learning_rate=self.learning_rate,
                                        depth=self.depth, l2_leaf_reg=self.l2_leaf_reg,
                                        loss_function=self.loss_function, cat_features=list(self.cat_features or []),
                                        random_seed=self.random_state, verbose=0, thread_count=-1)
        self.model_.fit(self._prep(X), y)
        self.feature_names_in_ = np.array(X.columns)
        return self

    def predict(self, X):
        return self.model_.predict(self._prep(X))

    @property
    def feature_importances_(self):
        return self.model_.get_feature_importance()


class RFANNResidualHybrid(RegressorMixin, BaseEstimator):
    """RF + ANN residual hybrid.
    1) Random Forest makes a first prediction.
    2) An MLP neural network learns the RF's *out-of-fold* errors (residuals), using the original features
       plus the RF prediction as an extra input.
    3) Final prediction = RF prediction + MLP residual correction."""

    def __init__(self, rf=None, ann=None, n_folds=5):
        self.rf = rf
        self.ann = ann
        self.n_folds = n_folds

    def _with_rf(self, X, rf_pred):
        X = X.copy()
        X["rf_pred"] = rf_pred
        return X

    def fit(self, X, y):
        oof = cross_val_predict(clone(self.rf), X, y, cv=KFold(self.n_folds, shuffle=False))  # honest RF errors
        self.rf_ = clone(self.rf).fit(X, y)
        self.ann_ = clone(self.ann).fit(self._with_rf(X, oof), y - oof)
        return self

    def predict(self, X):
        rf_pred = self.rf_.predict(X)
        return rf_pred + self.ann_.predict(self._with_rf(X, rf_pred))


class WeightedBlend(RegressorMixin, BaseEstimator):
    """w * model_a + (1 - w) * model_b, with w chosen beforehand on out-of-fold predictions."""

    def __init__(self, model_a=None, model_b=None, weight=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.weight = weight

    def fit(self, X, y):
        self.a_ = clone(self.model_a).fit(X, y)
        self.b_ = clone(self.model_b).fit(X, y)
        return self

    def predict(self, X):
        return self.weight * self.a_.predict(X) + (1 - self.weight) * self.b_.predict(X)
