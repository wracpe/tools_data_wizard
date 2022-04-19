import datetime
import pathlib
import warnings
import numpy as np
import pandas as pd
import plotly as pl
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from typing import Dict, List, Tuple


warnings.filterwarnings('ignore')


class Imputer(object):
    """Класс, содержащий логику заполнения пропусков в данных дебита и давления.

    Класс предназначен для заполнения пропусков в суточных временных данных
    конкретной нефтяной или нагнетательной скважины месторождения. Класс
    работает с набором данных папки data в корне проекта.

    Parameters
    ----------
    field_name: str
        Наименование месторождения. Должно соответствовать наименованию папки,
        содержащей все необходимые данные по месторождению.

    year_month_start_sh: tuple[int, int]
        Год и месяц начальной даты в таблице данных sh.

    year_month_end: tuple[int, int]
        Год и месяц последней даты во всех таблицах данных. Все таблицы должны
        быть согласованы по последней дате.

    estimator_name: {"ela", "knn", "tree"}
        Наименование модели, используемой для заполнения пропусков.

    References
    ----------

    .. [1] https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html

    """

    _path_general = pathlib.Path.cwd() / 'data'
    _fonds = [
        'Нефтяные',
        'Нагнетательные',
    ]
    _sosts = [
        'В работе',
        'Освоение прошлых лет',
        'Освоение текущего года',
    ]

    def __init__(
            self,
            field_name: str,
            year_month_start_sh: Tuple[int, int],
            year_month_end: Tuple[int, int],
            estimator_name: str,
    ):
        self._field_name = field_name
        self._folder_name = '%s_%s_%s_%s' % (year_month_start_sh + year_month_end)
        self._estimator_name = estimator_name
        self._counter = {}
        self._run()

    def _run(self) -> None:
        self._check_create_dir_existence()
        self._read_fond_sost_merop_sh()
        self._crop_fond_sost_merop()
        self._select_well_names_to_impute()
        self._impute_save()

    def _check_create_dir_existence(self) -> None:
        self._path_current = self._path_general / self._field_name / self._folder_name
        if not self._path_current.exists():
            raise FileNotFoundError(
                f'Директория "{self._path_current}" не существует. '
                f'Проверьте вводимые "year_month_start_sh" и "year_month_end".'
            )
        self._path_imputation_plots = self._path_current / 'well_sh_imputation_plots'
        if not self._path_imputation_plots.exists():
            self._path_imputation_plots.mkdir(parents=False)

    def _read_fond_sost_merop_sh(self) -> None:
        """Читает все необходимые для расчета входные таблицы данных."""
        self._df_fond = pd.read_feather(self._path_current / 'fond.feather')
        self._df_sost = pd.read_feather(self._path_current / 'sost.feather')
        self._df_merop = pd.read_feather(self._path_current / 'merop.feather')
        self._df_sh = pd.read_feather(self._path_current / 'sh.feather')
        self._check_sh()

    def _check_sh(self) -> None:
        """Проверяет соответствие входных дат наименованию папки с данными.

        Проверяются даты, поданные в конструктор класса (кортежи год, дата), и
        наименование папки внутри папки месторождения на соответствие.
        """
        dates_sh = self._df_sh['dt']
        self._date_sh_min = min(dates_sh)
        self._date_sh_max = max(dates_sh)
        date_sh_min_str = f'{self._date_sh_min.year}_{self._date_sh_min.month}'
        date_sh_max_str = f'{self._date_sh_max.year}_{self._date_sh_max.month}'
        if self._folder_name != f'{date_sh_min_str}_{date_sh_max_str}':
            logger.error('Min и max даты таблицы sh не соответствуют названию папки.')
            logger.error([self._field_name, date_sh_min_str, date_sh_max_str])
            raise AssertionError('Min и max даты таблицы sh не соответствуют названию папки.')

    def _crop_fond_sost_merop(self) -> None:
        """Приводит все таблицы к одним датам."""
        self._df_fond = self._crop_specific(self._df_fond, 'fond')
        self._df_sost = self._crop_specific(self._df_sost, 'sost')
        self._df_merop = self._df_merop.loc[
            (self._df_merop['dtend'] < self._date_sh_max) &
            (self._df_merop['dtend'] > self._date_sh_min)
        ]

    def _crop_specific(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        df = df.loc[
            (df['dtstart'] < self._date_sh_max) &
            (df['dtend'] > self._date_sh_min)
        ]
        df.dropna(inplace=True)  # TODO: Проверить необходимость данной процедуры.
        if df.empty:
            raise AssertionError(f'Таблица {df_name} не соответствует таблице sh по датам.')
        else:
            for well_name in df['well.ois'].unique():
                df_well = df.loc[df['well.ois'] == well_name]
                if df_well['dtstart'].iloc[0] < self._date_sh_min:
                    df_well['dtstart'].iloc[0] = self._date_sh_min  # Изменение dtstart.
                if df_well['dtend'].iloc[-1] > self._date_sh_max:
                    df_well['dtend'].iloc[-1] = self._date_sh_max  # Изменение dtend.
                df.update(df_well)
            df['well.ois'] = df['well.ois'].astype('int64')
        return df

    def _select_well_names_to_impute(self) -> None:
        """Выбирает номера скважин месторождения для заполнения.

        Проверяется наличие данных скважины в таблице sh, ее состояние в таблице
        sost, ее фонд в таблице fond.
        """
        well_names_by_sh = self._df_sh['well.ois'].unique()
        well_names_by_sost = self._df_sost.loc[self._df_sost['ssid.name'].isin(self._sosts)]['well.ois'].unique()
        self._df_fond = self._df_fond.loc[
            (self._df_fond['charwork.name'].isin(self._fonds)) &
            (self._df_fond['well.ois'].isin(sorted(set(well_names_by_sh) & set(well_names_by_sost))))
        ]
        self._well_names = self._df_fond['well.ois'].unique().tolist()
        if not self._well_names:
            raise AssertionError(
                f'Месторождение "{self._field_name}" не содержит данных sh '
                f'по работающим скважинам фондов {self._fonds}.'
            )
        else:
            print(f'Количество скважин для заполнения: {len(self._well_names)}.')

    def _impute_save(self) -> None:
        dfs = []
        dfs_origin = []
        for ind, well_name in enumerate(self._well_names):
            logger.info(f'{ind} out of {len(self._well_names)}: {well_name}')
            try:
                imputer = _ImputerByWellSh(
                    well_name,
                    self._estimator_name,
                    self._path_imputation_plots,
                    self._df_fond,
                    self._df_sost,
                    self._df_merop,
                    self._df_sh,
                    self._sosts,
                    self._counter
                )
                dfs.append(imputer.df_sh)
                dfs_origin.append(imputer.df_sh_origin)
            except Exception as exc:
                logger.exception(exc)
                logger.info([self._field_name, len(self._well_names), self._counter])
                continue
        self._df_sh_sost_fond = pd.concat(objs=dfs, ignore_index=True)
        self._df_sh_sost_fond.fillna(value=np.nan, inplace=True)
        self._df_sh_sost_fond.to_feather(f'{self._path_current}\\sh_sost_fond.feather')
        self._df_sh_sost_fond.to_excel(f'{self._path_current}\\sh_sost_fond.xlsx', engine='openpyxl')

        df_sh_sost_fond_origin = pd.concat(objs=dfs_origin, ignore_index=True)
        df_sh_sost_fond_origin.fillna(value=np.nan, inplace=True)
        df_sh_sost_fond_origin.to_excel(f'{self._path_current}\\sh_sost_fond_origin.xlsx', engine='openpyxl')


class _ImputerByWellSh(object):

    def __init__(
            self,
            well_name: int,
            estimator_name: str,
            path_imputation_plots: pathlib.Path,
            df_fond: pd.DataFrame,
            df_sost: pd.DataFrame,
            df_merop: pd.DataFrame,
            df_sh: pd.DataFrame,
            sosts: List[str],
            counter
    ):
        self._counter = counter
        self.well_name = well_name
        self._estimator_name = estimator_name
        self._path_imputation_plots = path_imputation_plots
        self._set_fond_sost_by_well(df_fond, df_sost, sosts)
        self._set_merop_by_well(df_merop)
        self._set_sh_by_well(df_sh)
        self._run()

    def _set_fond_sost_by_well(self, df_fond: pd.DataFrame, df_sost: pd.DataFrame, sosts: List[str]) -> None:
        self._df_fond = df_fond.loc[df_fond['well.ois'] == self.well_name]
        self._df_work = df_sost.loc[
            (df_sost['well.ois'] == self.well_name) &
            (df_sost['ssid.name'].isin(sosts))
        ]
        self._df_stop = df_sost.loc[
            (df_sost['well.ois'] == self.well_name) &
            (~df_sost['ssid.name'].isin(sosts))
        ]
        self._df_fond.drop(columns='well.ois', inplace=True)
        self._df_work.drop(columns='well.ois', inplace=True)
        self._df_stop.drop(columns='well.ois', inplace=True)

    def _set_merop_by_well(self, df_merop: pd.DataFrame) -> None:
        self._df_merop = df_merop.loc[df_merop['well.ois'] == self.well_name]
        self._df_merop.drop(columns='well.ois', inplace=True)
        self._df_merop.drop_duplicates(subset=['dtend', 'merid.name'], inplace=True)
        self._df_merop.set_index(keys='dtend', inplace=True, verify_integrity=False)
        # TODO: Добавить вывод уведомления о том, что обнаружены мероприятия на одну и ту же дату.
        self._df_merop = self._df_merop[~self._df_merop.index.duplicated(keep='first')]

    def _set_sh_by_well(self, df_sh: pd.DataFrame) -> None:
        self.df_sh = df_sh.loc[df_sh['well.ois'] == self.well_name]
        self.df_sh.drop(columns='well.ois', inplace=True)
        self.df_sh.set_index(keys='dt', inplace=True, verify_integrity=True)
        self._df_sh_copy = self.df_sh.copy()

    def _run(self) -> None:
        self._set_dates_sh_work()
        self._add_cols_sh()
        self._impute_sh()
        self._prepare_sh()

    def _set_dates_sh_work(self) -> None:
        dates_sh = self.df_sh.index.to_list()
        dates_work = []
        for i, r in self._df_work.iterrows():
            dates = pd.date_range(r['dtstart'], r['dtend'], freq='D').date.tolist()
            dates_work.extend(dates)
        self._dates_sh_work = sorted(set(dates_sh) & set(dates_work))

    def _add_cols_sh(self) -> None:
        self.df_sh['sost'] = 'Остановлена'
        self.df_sh.loc[self._dates_sh_work, 'sost'] = 'В работе'
        self.df_sh.loc[self._df_merop.index, 'merid.name'] = self._df_merop['merid.name']

    def _impute_sh(self) -> None:
        self._imputers_fonds = {}
        for i, r in self._df_fond.iterrows():
            fond_name = r['charwork.name']
            dates_fond = pd.date_range(r['dtstart'], r['dtend'], freq='D').date.tolist()
            dates_sh_work_fond = sorted(set(self._dates_sh_work) & set(dates_fond))
            df_sh_work_fond = self._df_sh_copy.loc[dates_sh_work_fond]
            self.df_sh.loc[dates_fond, 'charwork.name'] = fond_name
            self.df_sh_origin = self.df_sh.copy()
            try:
                targets = self._get_check_targets(fond_name, df_sh_work_fond)
                imputer = _ImputerByWellShWorkFond(
                    df_sh_work_fond,
                    targets,
                    self._estimator_name,
                )
            except Exception as exc:
                self.df_sh.loc[dates_sh_work_fond, 'sost'] = 'Остановлена'
                print(exc)
                print('Расчет по периоду пропущен. Состояние по скважине изменено на "Остановлена".')
                continue
            else:
                df_sh, df_stop, df_merop = self._prepare_plotter_data(dates_sh_work_fond)
                _PlotterByImputation(
                    self.well_name,
                    fond_name,
                    imputer,
                    df_sh,
                    df_stop,
                    self._path_imputation_plots,
                )
                self._imputers_fonds[fond_name] = imputer
                self.df_sh.update(imputer.df)

    def _get_check_targets(self, fond_name, df_sh_work_fond: pd.DataFrame) -> List[str]:
        dates = df_sh_work_fond.index
        targets_rate = self._get_targets_rate_as_dict()
        targets = targets_rate[fond_name]
        for target in targets:
            s_target = df_sh_work_fond[target].dropna()
            if s_target.empty:
                if target not in self._counter.keys():
                    self._counter[target] = 1
                else:
                    self._counter[target] += 1
                logger.error(f'Нет значений по "{target}" с {dates[0]} по {dates[-1]}.')
                raise AssertionError(f'Нет значений по "{target}" с {dates[0]} по {dates[-1]}.')

        if fond_name == 'Нефтяные':
            # Попытка добавить в targets забойное давление.
            targets_bhp = self._get_targets_bhp_as_dict()
            df = df_sh_work_fond[targets_bhp.values()].dropna(how='all')
            if df.empty:
                raise AssertionError(f'Нет значений по забойному давлению с {dates[0]} по {dates[-1]}.')
            targets.append(self._get_bhp(df_sh_work_fond))

        return targets

    def _prepare_plotter_data(self, dates: List[datetime.date]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        date_min = min(dates)
        date_max = max(dates)
        df_sh = self.df_sh.loc[date_min:date_max]
        df_stop = self._df_stop.loc[
            (self._df_stop['dtstart'] < date_max) &
            (self._df_stop['dtend'] > date_min)
        ]
        if not df_stop.empty:
            if df_stop['dtstart'].iloc[0] < date_min:
                df_stop['dtstart'].iloc[0] = date_min  # Изменение dtstart.
            if df_stop['dtend'].iloc[-1] > date_max:
                df_stop['dtend'].iloc[-1] = date_max  # Изменение dtend.
        df_merop = self._df_merop.loc[date_min:date_max]
        return df_sh, df_stop, df_merop

    def _prepare_sh(self) -> None:
        self.df_sh.reset_index(drop=False, inplace=True)
        self.df_sh['well.ois'] = self.well_name
        self.df_sh['Давление забойное'] = self.df_sh[self._get_bhp(self.df_sh)]

        self.df_sh_origin.reset_index(drop=False, inplace=True)
        self.df_sh_origin['well.ois'] = self.well_name
        self.df_sh_origin['Давление забойное'] = self.df_sh_origin[self._get_bhp(self.df_sh_origin)]
        cols_to_use = [
            'dt',
            'well.ois',
            'charwork.name',
            'sost',
            'merid.name',
            'Давление забойное',
        ]
        cols_to_use.extend(self._get_targets_bhp_as_dict().values())
        targets_rate = self._get_targets_rate_as_dict()
        for fond_name in targets_rate.keys():
            cols_to_use.extend(targets_rate[fond_name])
        self.df_sh = self.df_sh[cols_to_use]
        self.df_sh_origin = self.df_sh_origin[cols_to_use]

    def _get_bhp(self, df: pd.DataFrame) -> str:
        targets_bhp = self._get_targets_bhp_as_dict()
        bhp_pump = targets_bhp['pump']
        bhp_hdyn = targets_bhp['hdyn']
        bhp_pump_count = df[bhp_pump].count()
        bhp_hdyn_count = df[bhp_hdyn].count()
        if bhp_hdyn_count > 0:
            if bhp_pump_count / bhp_hdyn_count > 0.75:
                return bhp_pump
            else:
                return bhp_hdyn
        else:
            return bhp_pump

    @staticmethod
    def _get_targets_rate_as_dict() -> Dict[str, List[str]]:
        return {
            'Нефтяные': [
                'Дебит жидкости среднесуточный',
                'Дебит нефти расчетный',
                'Газовый фактор рабочий (ТМ)',
                'Дебит газа (ТМ)',
                'Дебит газа попутного',
            ],
            'Нагнетательные': [
                'Приемистость среднесуточная',
            ],
        }

    @staticmethod
    def _get_targets_bhp_as_dict() -> Dict[str, str]:
        return {
            'pump': 'Давление забойное от Pпр',
            'hdyn': 'Давление забойное от Hд',
        }


class _ImputerByWellShWorkFond(object):

    _estimators = {
        'ela':
            ElasticNet(
                random_state=1,
            ),
        'knn':
            KNeighborsRegressor(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1,
            ),
        'tree':
            ExtraTreesRegressor(
                n_estimators=10,
                max_depth=5,
                max_features='sqrt',
                n_jobs=-1,
                random_state=1,
            ),
    }
    _nan_percent = 0.8
    _corr_value = 0.3
    _max_features = 10

    def __init__(
            self,
            df: pd.DataFrame,
            targets: List[str],
            estimator_name: str,
    ):
        self.df = df.copy()
        self.df_limited = df.copy()
        self.targets = targets
        self.estimator_name = estimator_name
        self._n = len(df)
        self._k = int(self._n * self._nan_percent)
        self._create()
        self._run()

    def _create(self) -> None:
        estimator = self._estimators[self.estimator_name]
        self._imputer = IterativeImputer(
            estimator,
            max_iter=1000,
            initial_strategy='median',
            imputation_order='ascending',
            skip_complete=True,
            min_value=0,
            random_state=1,
        )

    def _run(self) -> None:
        self._set_features()
        self._impute_by_estimator()
        self._limit_df()

    def _set_features(self) -> None:
        df_features = self.df.drop(columns=self.targets).dropna(axis=1, thresh=self._k)
        df_rank = pd.DataFrame(index=df_features.columns, columns=self.targets)
        for target in self.targets:
            df_rank[target] = abs(df_features.corrwith(self.df[target], drop=True, method='kendall'))
        s_rank = df_rank.max(axis=1)
        s_rank = s_rank.loc[s_rank > self._corr_value].sort_values(ascending=False).head(self._max_features)
        self.features = s_rank.index.to_list()

    def _impute_by_estimator(self) -> None:
        self.params = self.targets + self.features
        df = self.df[self.params]
        # TODO: Проверить необходимость простой интерполяции перед заполнением основным методом.
        df.interpolate(axis=0, limit=7, inplace=True, limit_direction='both')
        mx = self._imputer.fit_transform(df)
        self.df[df.columns] = pd.DataFrame(data=mx, index=df.index, columns=df.columns)

    def _limit_df(self) -> None:
        self.df_limited = self.df_limited[self.params].copy()
        for param in self.params:
            new_col = f'{param}_imp'
            idx_nan = self.df_limited[param].loc[self.df_limited[param].isna()].index
            self.df_limited[new_col] = self.df[param].loc[idx_nan]


class _PlotterByImputation(object):

    def __init__(
            self,
            well_name: int,
            fond_name: str,
            imputer: _ImputerByWellShWorkFond,
            df_sh: pd.DataFrame,
            df_stop: pd.DataFrame,
            path_imputation_plots: pathlib.Path,
    ):
        self._well_name = well_name
        self._fond_name = fond_name
        self._imputer = imputer
        self._df_sh = df_sh
        self._df_stop = df_stop
        self._path = str(path_imputation_plots)
        self._run()

    def _run(self) -> None:
        self._figure = go.Figure(layout=go.Layout(
            font=dict(size=10),
            hovermode='x',
            template='seaborn',
            title=dict(text=f'Скважина {self._well_name}, фонд "{self._fond_name}"', x=0.05, xanchor='left'),
        ))
        self._set_file_name()
        self._create_shrunk_chess_plot()
        # self._create_extend_chess_plot()

    def _set_file_name(self) -> None:
        self._file_name = f'{self._path}\\{self._well_name}_{self._fond_name}'

    def _create_shrunk_chess_plot(self) -> None:
        fig = self._create_subplot_figure(self._imputer.targets)
        pl.io.write_image(fig, f'{self._file_name}.png', width=1450, height=700, scale=2, engine='kaleido')

    def _create_extend_chess_plot(self) -> None:
        fig = self._create_subplot_figure(self._imputer.params)
        pl.io.write_html(fig, f'{self._file_name}.html', auto_open=False)

    def _create_subplot_figure(self, cols: List[str]) -> go.Figure:
        n_params = len(cols)
        h0 = 0.05
        row_heights = [h0] + [(1 - h0) / n_params] * n_params
        figure = go.Figure(self._figure)
        fig = make_subplots(
            rows=n_params + 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=row_heights,
            figure=figure,
        )
        fig.update_xaxes(dtick='M1', tickformat="%b\n%Y")
        fig.update_yaxes(row=1, col=1, range=[0, 1], showticklabels=False)

        s = self._df_sh['merid.name'].dropna()
        trace = go.Scatter(
            name='мероприятие',
            x=s.index,
            y=[0.2] * len(s),
            mode='markers+text',
            marker=dict(size=8),
            text=s.array,
            textposition='top center',
            textfont=dict(size=8),
        )
        fig.add_trace(trace, row=1, col=1)

        for i in range(n_params):
            param1 = cols[i]
            param2 = param1 + '_imp'
            x1 = self._df_sh.index.to_list()
            y1 = self._df_sh[param1]
            x2 = self._imputer.df_limited.index.to_list()
            y2 = self._imputer.df_limited[param2]
            trace1 = go.Scatter(name=param1, x=x1, y=y1, mode='markers', marker=dict(size=3))
            trace2 = go.Scatter(name=param2, x=x2, y=y2, mode='markers', marker=dict(size=3))
            fig.add_trace(trace1, row=i + 2, col=1)
            fig.add_trace(trace2, row=i + 2, col=1)

        for i, r in self._df_stop.iterrows():
            x0 = r['dtstart']
            x1 = r['dtend']
            fig.add_vrect(x0=x0, x1=x1, fillcolor='red', opacity=0.10, line_width=0)

        return fig
