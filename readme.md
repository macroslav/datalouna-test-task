# Файлы с предиктами лежат в __data/predicts__

# Для запуска решения:
+ Либо Jupyter Notebook _notebooks/solution.ipynb_
+ Либо скрипт _run.py_

# Использованные модели и метрики:
+ CatboostClassifier - **0.63**
+ Dense neural network - **0.62**
+ LogisticRegression - **0.63**

# Как можно улучшить:
+ В текущей версии сравниваю каждого игрока с каждым - можно сравнить и команды в целом друг с другом (брать среднее значение статистики по всей команде и сравнивать с таким же средним у каждого противника/команды противник)
+ Добавить фичи с датой - появятся временные ряды, можно будет посчитать скользящие средние за промежуток времени, например, текущий стрик команды (победы/поражения)
+ Больше данных - лучше модель, 730 строчек мало и для катбуста, и для нейронки
+ Придумать, как сделать хороший эмбединг игрока, команды, карты, на нем обучать нейронки