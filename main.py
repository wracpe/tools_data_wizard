import time
from loguru import logger
from imputer import Imputer

logger.add('log.log', level='DEBUG', format="{time:HH:mm:ss} {level} {message}")

# Наименования месторождений для расчета.
# Должны быть согласованы с наименованием папок по месторождениям в data.
# Пример: Отдельное
field_names = [
    # 'Валынтойское',
    # 'Вынгаяхинское',
    # 'Вынгапуровское',
    # 'Западно-Чатылькинское',
    'Крайнее',
    # 'Новопортовское',
    # 'Оренбургское',
    # 'Отдельное',
    # 'Романовское',
    # 'Холмистое',
    # 'Холмогорское',
    # 'Чатылькинское',
]


# Даты расчета.
# Должны быть согласованы с наименованием папки данных внутри папки месторождения.
# Пример: 2018_1_2021_4
year_month_start_sh = (2018, 1)
year_month_end = (2021, 8)


for field_name in field_names:
    time_start = time.time()
    logger.info(f'Старт расчета для месторождения "{field_name}".')
    try:
        imp = Imputer(
            field_name,
            year_month_start_sh,
            year_month_end,
            estimator_name='knn',
        )
    except Exception as exc:
        print(exc)
        print('Расчет по месторождению не выполнен.')
        continue
    else:
        run_time = round((time.time() - time_start) / 60, 1)
        logger.success(f'Время выполнения: {run_time} мин.')
    logger.success([f'Конец расчета для месторождения "{field_name}".', imp.counter])
