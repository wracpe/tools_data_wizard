import math
import pathlib

import cufflinks as cf
import numpy as np
import pandas as pd
import plotly as pl
import plotly.graph_objects as go

from datetime import timedelta
from plotly import subplots

from imputer import Imputer


target = 'Давление забойное от Pпр'
# target = 'Дебит жидкости среднесуточный'
fields = [
    'kholmogorskoe',
    'otdelnoe',
    'romanovskoe',
]

estimator_name = 'simple'
# estimator_name = 'elastic_net'
# estimator_name = 'knn'
# estimator_name = 'extra_trees'
features_num = 5
# imputer = Imputer(features_num, estimator_name)


fullness_min = 0.95
drop_percent = 0.33
cf.go_offline()
path = pathlib.Path.cwd()
path_data = path / 'data'
path_results = path / 'results' / f'{estimator_name}_{target}'
data = list()
s_well_error = pd.Series(dtype=float)


def read_chess(field: str) -> pd.DataFrame:
    df = pd.read_csv(filepath_or_buffer=path_data / field / 'град_штр.csv',
                     sep=';',
                     dtype={0: str},
                     parse_dates=[1],
                     dayfirst=True,
                     encoding='windows-1251')

    drop_cols = [
        'Давление в линии (нефт)',
        'Давление на входе ЭЦН (ТМ)',
        'Давление на приеме насоса',
        'Давление на БГ КНС',
        'Давление на БГ куста / ГЗУ',
        'Дебит газа',
        'Дебит газа (ТМ)',
        'Дебит газа попутного',
        'Дебит газлифтного газа',
        'Дебит газового конденсата',
        'Дебит жидкости',
        'Дебит жидкости (ТМ)',
        'Дебит нефти (ТМ)',
        'Дебит нефти расчетный',
        'Дебит пластового газа',
        'Дебит стабильного конденсата',
        'Дебит сухого газа',
        'Дебит сырого г/к',
        'Масса реагента УДР (ТМ)',
        'Напряжение BC (ТМ)',
        'Напряжение CA (ТМ)',
        'Расход реагента УДР (ТМ), л/сут',
        'Состояние УДР',
        'Ток фазы B (ТМ)',
        'Ток фазы C (ТМ)',
        'Сила тока ЭЦН',
        'Сила тока ЭЦН (ТМ)',
        'Удельный расход электроэнергии',
    ]
    df.drop(columns=drop_cols, inplace=True)
    df.dropna(axis='columns', how='all', inplace=True)
    return df


def read_status(field: str) -> pd.DataFrame:
    df = pd.read_excel(io=path_data / field / 'град_состояния.xlsx',
                       usecols=[1, 2, 3, 5],
                       dtype={1: str},
                       skiprows=2,
                       parse_dates=[1, 2])

    df.drop(index=[0, 1], inplace=True)
    df = df[df['Сост. скв.'] == 'В работе']
    return df


def slice_chess(well: str, df_ch: pd.DataFrame, df_st: pd.DataFrame) -> pd.DataFrame:
    df_ch_well = df_ch[df_ch['Скв'] == well]
    df_ch_well.drop(columns='Скв', inplace=True)
    df_ch_well.set_index(keys='Дата', inplace=True, verify_integrity=False)
    df_ch_well.sort_index(inplace=True)
    df_ch_well = df_ch_well.astype(dtype='float64')

    date_start = df_ch_well.index[0]
    df_st_well = df_st[(df_st['Скважина'] == well) & (df_st['Дата начала'] >= date_start)]
    if df_st_well.empty:
        return None
    dfs = list()

    for i in df_st_well.index:
        date_1 = df_st_well.at[i, 'Дата начала'].date()
        date_2 = df_st_well.at[i, 'Дата конца'].date()
        if date_1 == date_2:
            continue
        df = df_ch_well.loc[date_1:date_2 - timedelta(days=1)]
        dfs.append(df)
    df_ch_well = pd.concat(dfs)
    return df_ch_well


def drop_values_from_target(df_well: pd.DataFrame) -> (pd.DataFrame, pd.Index):
    index_size = len(df_well.index)
    drop_number = int(index_size * drop_percent)
    df_well_drop = df_well.copy()
    drop_indexes = df_well_drop[target].tail(drop_number).dropna().index
    df_well_drop[target].loc[drop_indexes] = math.nan
    return df_well_drop, drop_indexes


def create_plot(well: str, df_well: pd.DataFrame, df_well_imp: pd.DataFrame):
    fig = subplots.make_subplots(rows=features_num + 1,
                                 cols=1,
                                 shared_xaxes=True,
                                 vertical_spacing=0.02,
                                 specs=[[{}]] + [[{}]] * features_num)

    fig.layout.template = 'seaborn'
    fig.update_layout(width=1450,
                      title=dict(text=f'{well} for {estimator_name}', font=dict(size=20), x=0.05, xanchor='left'),
                      font=dict(size=11),
                      hovermode='x')

    i = 1
    for col in df_well_imp.columns:
        x_imp = df_well_imp[col].index
        y_imp = df_well_imp[col].array
        trace = go.Scatter(name=col, mode='lines', x=x_imp, y=y_imp)
        fig.add_trace(trace, row=i, col=1)

        x_true = df_well[col].index
        y_true = df_well[col].array
        trace = go.Scatter(name='true', showlegend=False, mode='markers', x=x_true, y=y_true, marker=dict(size=4))
        fig.add_trace(trace, row=i, col=1)
        i += 1

    pl.io.write_html(fig, f'{path_results}\\{well}.html')


def fill_simple(df_well_drop, drop_indexes):
    df_well_imp = df_well_drop[[target, 'Давление забойное от Hд']]
    df_well_imp[target].loc[drop_indexes] = df_well_imp['Давление забойное от Hд'].loc[drop_indexes]
    return df_well_imp


def calc_mape(y_true: pd.Series, y_imp: pd.Series) -> float:
    y_true, y_imp = np.array(y_true), np.array(y_imp)
    errors = np.abs((y_true - y_imp) / y_true)
    mape = np.mean(errors[~np.isnan(errors)]) * 100
    return mape


for field in fields:
    df_ch = read_chess(field)
    df_st = read_status(field)
    wells = df_ch['Скв'].unique()
    # TODO: Delete.
    wells = wells.tolist()
    if field == 'kholmogorskoe':
        wells.remove('95')
    if field == 'romanovskoe':
        wells.remove('1031_1')
        wells.remove('1105')
        wells.remove('2156')

    for well in wells:
        df_well = slice_chess(well, df_ch, df_st)
        if df_well is None:
            continue
        s_target = df_well[target]
        fullness = s_target.count() / len(s_target)
        if fullness >= fullness_min:
            well = f'{field}_' + well
            print(well)

            df_well_drop, drop_indexes = drop_values_from_target(df_well)
            # TODO: Refactor.
            df_well_imp = fill_simple(df_well_drop, drop_indexes)
            # df_well_imp = imputer.predict(df_well_drop, target, features_num)

            create_plot(well, df_well, df_well_imp)

            s_true = df_well[target].loc[drop_indexes]
            s_imp = df_well_imp[target].loc[drop_indexes]
            mape = calc_mape(s_true, s_imp)
            s_well_error.loc[well] = mape

            row = [well, mape] + list(df_well_imp.columns)
            data.append(row)


# features_cols = [f'feature_{x}' for x in range(1, features_num+1)]
# data = pd.DataFrame(data, columns=['well', 'MAPE', 'target'] + features_cols)
# s = pd.concat(objs=[data[f] for f in features_cols], ignore_index=True)
# num = s.value_counts()
# data.to_excel(excel_writer=f'{path_results}\\_features.xlsx', float_format='%.3f')
# num.to_excel(excel_writer=f'{path_results}\\_features_main.xlsx')

layout = go.Layout(width=1450, hovermode='x')
fig = s_well_error.iplot(layout=layout, kind='bar', theme='ggplot', asFigure=True)
fig.update_layout(title_text=f'MAPE by wells for {estimator_name}')

mean_error = s_well_error.mean()
x0 = s_well_error.index[0]
x1 = s_well_error.index[-1]
fig.add_shape(type='line', x0=x0, x1=x1, y0=mean_error, y1=mean_error)
pl.io.write_html(fig, f'{path_results}\\_performance.html')
