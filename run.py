import pandas as pd
from pathlib import Path
import logging

from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier, Pool
from tensorflow.keras import layers
from tensorflow import keras

from src.data.data_preprocessors import DataPreprocessor
from src.data.data_loaders import DataLoader
from src.data.data_savers import DataSaver

DATA_DIR = Path('data/raw').absolute()
PREDICTS_DIR = Path('data/predicts').absolute()
BASIC_PARAMS = {'verbose': 0,
                'iterations': 1000}


def main():
    logging.info("Loading data ...")
    data_loader = DataLoader()
    train_data, players_data, test_data = data_loader.load_data(data_dir=DATA_DIR)
    test_data = test_data.drop(columns='index')

    logging.info("Preprocessing data ...")
    preprocessor = DataPreprocessor()
    players_data_processed = preprocessor.preprocess(matches_data=train_data,
                                                     players_data=players_data)

    x_test = preprocessor.merge_players_with_matches(test_data)
    x_test = x_test.drop(columns=[f"p{i}_id_{side}" for i in range(1, 6) for side in ['first', 'second']])
    x_test = preprocessor.add_features(x_test)
    target = 'who_win'

    x, x_test = prepare_data_for_models(test_data=x_test, train_data=players_data_processed)
    y = players_data_processed[[target]]

    train_pool = Pool(data=x, label=y)

    logging.info("Fitting models")
    model_ctb = train_catboost_model(train_pool=train_pool,
                                     params=BASIC_PARAMS)
    model_nn = train_nn_model(x, y)
    model_lr = train_lr_model(x, y)

    logging.info("Making predicts")
    predicts_list = list()
    test_data[target] = model_lr.predict_proba(x_test)[:, 1]
    predicts_list.append(test_data.loc[:, ['map_id', target]])

    test_data[target] = model_ctb.predict_proba(x_test)[:, 1]
    predicts_list.append(test_data.loc[:, ['map_id', target]])

    test_data[target] = model_nn.predict(x_test)
    predicts_list.append(test_data.loc[:, ['map_id', target]])

    logging.info("Saving predicts")
    data_saver = DataSaver()
    data_saver.save(save_dir=PREDICTS_DIR, content=predicts_list)


def train_catboost_model(train_pool: Pool, params: dict) -> CatBoostClassifier:
    """
    Fit and return fitted CatboostClassifier model.
    params:
        x_train: pd.DataFrame,
        y_train: pd.DataFrame.
    return: pd.DataFrame.
    """
    model_ctb = CatBoostClassifier(**params)
    model_ctb.fit(train_pool)

    return model_ctb


def train_nn_model(x_train: pd.DataFrame, y_train: pd.DataFrame) -> keras.Sequential:
    """
    Fit and return fitted dense neural network model.
    params:
        x_train: pd.DataFrame,
        y_train: pd.DataFrame.
    return: pd.DataFrame.
    """
    model_nn = keras.Sequential([
        layers.Dense(x_train.shape[1], activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ])

    model_nn.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    history = model_nn.fit(
        x_train, y_train,
        batch_size=1024,
        epochs=30,
        verbose=0
    )

    return model_nn


def train_lr_model(x_train: pd.DataFrame, y_train: pd.DataFrame) -> LogisticRegression:
    """
    Fit and return fitted LogisticRegression model.
    params:
        x_train: pd.DataFrame,
        y_train: pd.DataFrame.
    return: pd.DataFrame.
    """
    lr_model = LogisticRegression()
    lr_model.fit(x_train, y_train)

    return lr_model


def prepare_data_for_models(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat_features = ['team1_id', 'team2_id', 'map_name']
    target = 'who_win'
    x = train_data.drop(columns=[target, 'map_id'] + cat_features)
    x_test = test_data.drop(columns=['map_id'] + cat_features)
    x = x.fillna(0)

    return x, x_test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
