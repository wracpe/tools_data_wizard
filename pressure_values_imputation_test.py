import cufflinks as cf
import math
import pandas as pd
import pathlib
import plotly as pl
import plotly.graph_objects as go
# import warnings


from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor


cf.go_offline()
pd.set_option('display.max_rows', 200)
layout = go.Layout(width=1450)
# warnings.filterwarnings('ignore')

path = pathlib.Path.cwd()
fields = ['kholmogorskoe', 'otdelnoe', 'romanovskoe']  # kholmogorskoe, otdelnoe, romanovskoe
target = 'Давление забойное от Pпр'

half_range = 0.5

fields_dict = dict()
wells_total = list()

for field in fields:
    df = pd.read_excel(io=path / field / 'day.xlsx',
                       header=0,
                       dtype={0: str},
                       parse_dates=[1])

    df.drop(columns=['Дебит жидкости (ТМ)',
                     'Дебит жидкости',
                     'Дебит нефти (ТМ)',
                     'Дебит газа попутного',
                     'Давление на входе ЭЦН (ТМ)'], inplace=True)

    wells_names = df['Скв'].unique()
    wells_target_fullness = pd.Series(index=wells_names)

    for well_name in wells_names:
        df_well = df[df['Скв'] == well_name].copy()
        s_target = df_well[target]
        fullness = s_target.count() / len(s_target)
        wells_target_fullness[well_name] = fullness

    wells_target_fullness.sort_values(ascending=False, inplace=True)
    wells_names = wells_target_fullness[wells_target_fullness > 0.8].index
    fields_dict[field] = {'df': df, 'wells_names': wells_names}

    wells = [f'{field}_' + well for well in wells_names]
    wells_total.extend(wells)

# wells_names = ['2153']
s_well_error = pd.Series(index=wells_total)


def select_features(df_well):
    s_target = df_well[target]
    df_well.drop(columns=[target], inplace=True)
    df_well.dropna(axis='columns', how='all', inplace=True)

    num = df_well.count()
    corr = abs(df_well.corrwith(s_target, drop=True, method='pearson'))
    rank = pd.concat(objs=[num, corr], axis='columns')
    rank.columns = ['num', 'corr']
    rank.dropna(axis='index', how='any', inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(rank)
    rank = pd.DataFrame(data=X, index=rank.index, columns=rank.columns)

    rank = rank['num'] + rank['corr']
    rank.sort_values(ascending=False, inplace=True)
    rank = rank.head(5)

    features = list(rank.index)
    df_well = df_well[features]
    df_well = df_well.join(s_target)
    return df_well


def drop_values_from_target(df_well):
    center = df_well.shape[0] / 2
    left = int(center * (1 - half_range))
    right = int(center * (1 + half_range))
    df_well_drop = df_well.copy()
    indexes_drop = df_well_drop.iloc[left : right].index
    # indexes_drop = df_well_drop[target].iloc[left : right].dropna().index
    df_well_drop[target].loc[indexes_drop] = math.nan
    return indexes_drop, df_well_drop


# estimator = BayesianRidge(compute_score=True)
# estimator = DecisionTreeRegressor(criterion='mae', max_depth=5, random_state=1)
estimator = ExtraTreesRegressor(n_jobs=8, random_state=1)
# estimator = KNeighborsRegressor()

iter_imp = IterativeImputer(estimator,
                            # sample_posterior=True,
                            max_iter=100,
                            tol=0.01,
                            initial_strategy='median',
                            imputation_order='descending',
                            skip_complete=True,
                            random_state=1)


for field in fields:
    df = fields_dict[field]['df']
    wells_names = fields_dict[field]['wells_names']

    for well_name in wells_names:
        df_well = df[df['Скв'] == well_name].copy()
        df_well.drop(columns=['Скв'], inplace=True)
        df_well.set_index(keys='Дата', inplace=True, verify_integrity=True)
        df_well = df_well.astype(dtype='float64')

        df_well = select_features(df_well)
        # names = list(df_well.columns)
        # names.remove(target)
        # df_well[names].interpolate(method='linear', axis='index', inplace=True, limit_direction='both')
        df_well.interpolate(method='linear', axis='index', inplace=True, limit_direction='both')               

        indexes_drop, df_well_drop = drop_values_from_target(df_well)

        X = iter_imp.fit_transform(df_well_drop)
        df_well_imp = pd.DataFrame(X, index=df_well_drop.index, columns=df_well_drop.columns)

        s_true = df_well[target].loc[indexes_drop]
        s_imp = df_well_imp[target].loc[indexes_drop]
        s_error = abs(s_true - s_imp).divide(s_imp)

        s_well_error.loc[f'{field}_{well_name}'] = s_error.mean()

        s = df_well_imp[target]
        s.name = 'imp'
        df_compare = df_well.join(s)

        fig = df_compare.iplot(layout=layout, theme='ggplot', asFigure=True)
        pl.io.write_html(fig, f'{field}_{well_name}.html')
        # s_error.iplot(layout=layout, mode='lines+markers', size=3)

        print(well_name)
    print(field)

mean_error = s_well_error.mean()
x0 = s_well_error.index[0]
x1 = s_well_error.index[-1]
fig = s_well_error.iplot(layout=layout, kind='bar', theme='ggplot', asFigure=True)
fig.add_shape(type='line', x0=x0, x1=x1, y0=mean_error, y1=mean_error)
pl.io.write_html(fig, 'performance.html')

# cf.getThemes()
