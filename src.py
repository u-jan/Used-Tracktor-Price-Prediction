""" This solution makes heavy use of sklearn's Pipeline class.
    You can find documentation on using this class here:
    http://scikit-learn.org/stable/modules/pipeline.html
"""
from datetime import timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


class DataType(BaseEstimator, TransformerMixin):
    """Cast the data types of the id and data source columns to strings
    from numerics.
    """
    col_types = {'str': ['MachineID', 'ModelID', 'datasource']}

    def fit(self, X, y):
        return self

    def transform(self, X):
        for col_type, column in self.col_types.items():
            X[column] = X[column].astype(col_type)
        X['saledate_converted'] = pd.to_datetime(X.saledate)
        return X


class FilterColumns(BaseEstimator, TransformerMixin):
    """Only keep columns that don't have NaNs.
    """
    def fit(self, X, y):
        column_counts = X.count(axis=0)
        self.keep_columns = column_counts[column_counts == column_counts.max()]
        return self

    def transform(self, X):
        return X.loc[:, self.keep_columns.index]


class ReplaceOutliers(BaseEstimator, TransformerMixin):
    """Replace year made when listed as earlier than 1900, with
    mode of years after 1900. Also add imputation indicator column.
    """
    def fit(self, X, y):
        self.replace_value = X.YearMade[X.YearMade > 1900].mode()
        return self

    def transform(self, X):
        condition = X.YearMade > 1900
        X['YearMade_imputed'] = 0
        X.loc[~condition, 'YearMade'] = self.replace_value[0]
        X.loc[~condition, 'YearMade_imputed'] = 1
        return X


class ComputeAge(BaseEstimator, TransformerMixin):
    """Compute the age of the vehicle at sale.
    """
    def fit(self, X, y):
        return self

    def transform(self, X):
        saledate = pd.to_datetime(X.saledate)
        X['equipment_age'] = saledate.dt.year - X.YearMade
        return X


class ComputeNearestMean(BaseEstimator, TransformerMixin):
    """Compute a mean price for similar vehicles.
    """
    def __init__(self, window=5):
        self.window = window

    def get_params(self, **kwargs):
        return {'window': self.window}

    def fit(self, X, y):
        X = X.sort_values(by=['saledate_converted'])
        g = X.groupby('ModelID')['SalePrice']
        m = g.apply(lambda x: x.rolling(self.window).agg([np.mean]))

        ids = X[['saledate_converted', 'ModelID', 'SalesID']]
        z = pd.concat([m, ids], axis=1)
        z['saledate_converted'] = z.saledate_converted + timedelta(1)

        # Some days will have more than 1 transaction for a particular model,
        # take the last mean (which has most info)
        z = z.drop('SalesID', axis=1)
        groups = ['ModelID', 'saledate_converted']
        self.averages = z.groupby(groups).apply(lambda x: x.tail(1))

        # This is kinda unsatisfactory, but at least ensures
        # we can always make predictions
        self.default_mean = X.SalePrice.mean()
        return self

    def transform(self, X):
        near_price = pd.merge(self.averages, X, how='outer',
                              on=['ModelID', 'saledate_converted'])
        nxcols = ['ModelID', 'saledate_converted']
        near_price = near_price.set_index(nxcols).sort_index()
        g = near_price['mean'].groupby(level=0)
        filled_means = g.transform(lambda x: x.fillna(method='ffill'))
        near_price['filled_mean_price'] = filled_means
        near_price = near_price[near_price['SalesID'].notnull()]
        missing_mean = near_price.filled_mean_price.isnull()
        near_price['no_recent_transactions'] = missing_mean
        near_price['filled_mean_price'].fillna(self.default_mean, inplace=True)
        return near_price


class ColumnFilter(BaseEstimator, TransformerMixin):
    """Only use the following columns.
    """
    columns = ['YearMade', 'YearMade_imputed', 'equipment_age',
               'filled_mean_price', 'no_recent_transactions']

    def fit(self, X, y):
        # Get the order of the index for y.
        return self

    def transform(self, X):
        X = X.set_index('SalesID')[self.columns].sort_index()
        return X


def rmsle(y_hat, y, y_min=5000):
    """Calculate the root mean squared log error between y
    predictions and true ys.
    (hard-coding y_min for dumb reasons, sorry)
    """

    if y_min is not None:
        y_hat = np.clip(y_hat, y_min, None)
    log_diff = np.log(y_hat+1) - np.log(y+1)
    return np.sqrt(np.mean(log_diff**2))


if __name__ == '__main__':
    df = pd.read_csv('data/Train.csv')
    df = df.set_index('SalesID').sort_index()
    y = df.SalePrice

    # Modeling decision: if our model outputs anything lower
    # than this value, round up to this instead
    # (should be part of the pipeline, sorry)
    y_min_cutoff = 5000

    # This is for predefined split... we want -1 for our training split,
    # 0 for the test split.
    cv_cutoff_date = pd.to_datetime('2011-01-01')
    cv = -1*(pd.to_datetime(df.saledate) < cv_cutoff_date).astype(int)

    cross_val = PredefinedSplit(cv)

    p = Pipeline([
        ('filter', FilterColumns()),
        ('type_change', DataType()),
        ('replace_outliers', ReplaceOutliers()),
        ('compute_age', ComputeAge()),
        ('nearest_average', ComputeNearestMean()),
        ('columns', ColumnFilter()),
        ('lm', LinearRegression())
    ])
    df = df.reset_index()

    # GridSearch
    params = {'nearest_average__window': [3, 5, 7]}

    # Turns our rmsle func into a scorer of the type required
    # by gridsearchcv.
    rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

    gscv = GridSearchCV(p, param_grid=params,
                        scoring=rmsle_scorer,
                        cv=cross_val,
                        n_jobs=-1)
    clf = gscv.fit(df.reset_index(), y)

    print('Best parameters: {}'.format(clf.best_params_))
    print('Best RMSLE: {}'.format(clf.best_score_))

    test = pd.read_csv('data/test.csv')
    test = test.sort_values(by='SalesID')

    test_predictions = np.clip(clf.predict(test), y_min_cutoff, None)
    test['SalePrice'] = test_predictions
    outfile = 'data/solution_benchmark.csv'
    test[['SalesID', 'SalePrice']].to_csv(outfile, index=False)