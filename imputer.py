import warnings

import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import IterativeImputer
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings(action='ignore')


class Imputer(object):

    _scaler = MinMaxScaler(feature_range=(0, 1))

    def __init__(self,
                 features_num: int,
                 estimator_name: str,
                 ):

        self.feature_num = features_num
        self.estimator_name = estimator_name
        self._estimator = None
        self._imputer = None
        self._create_imputer()

    def _create_imputer(self):
        if self.estimator_name == 'elastic_net':
            self._estimator = ElasticNet(random_state=1)

        if self.estimator_name == 'knn':
            self._estimator = KNeighborsRegressor(n_jobs=-1)

        if self.estimator_name == 'extra_trees':
            self._estimator = ExtraTreesRegressor(n_jobs=-1, random_state=1)

        self._imputer = IterativeImputer(self._estimator,
                                         max_iter=100,
                                         initial_strategy='median',
                                         imputation_order='descending',
                                         skip_complete=True,
                                         min_value=0,
                                         random_state=1)

    def predict(self, df: pd.DataFrame, target: str, features_num: int) -> pd.DataFrame:
        df = self._select_features(df, target, features_num)
        x = self._imputer.fit_transform(df)
        df = pd.DataFrame(x, index=df.index, columns=df.columns)
        return df

    @classmethod
    def _select_features(cls, df: pd.DataFrame, target: str, features_num: int) -> pd.DataFrame:
        s = df[target]
        df.drop(columns=[target], inplace=True)
        df.dropna(axis='columns', how='all', inplace=True)

        num = df.count()
        corr = abs(df.corrwith(s, drop=True, method='pearson'))
        rank = pd.concat(objs=[num, corr], axis=1)
        rank.columns = ['num', 'corr']
        rank.dropna(axis='index', how='any', inplace=True)
        x = cls._scaler.fit_transform(rank)
        rank = pd.DataFrame(data=x, index=rank.index, columns=rank.columns)

        rank = rank['num'] + rank['corr']
        rank.sort_values(ascending=False, inplace=True)
        rank = rank.head(features_num)

        features = list(rank.index)
        df = df[features]
        df.insert(loc=0, column=target, value=s.array)
        return df
