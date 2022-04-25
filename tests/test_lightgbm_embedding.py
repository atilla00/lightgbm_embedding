from lightgbm_embedding import __version__, LightgbmEmbedding
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def test_version():
    assert __version__ == "0.1.0"


def test_binary_classification():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    )
    cols = [
        "PassengerId",
        "Pclass",
        "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Ticket",
        "Fare",
        "Cabin",
        "Embarked",
    ]
    target = "Survived"

    for col in df.columns:
        if df[col].dtype.name == "object":
            df[col] = df[col].astype("category")

    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

    n_dim = 20
    emb = LightgbmEmbedding(n_dim=n_dim)
    emb.fit(X_train[cols], X_train[target], target_type="continuous")

    X_train_embed = emb.transform(X_train[cols])
    X_test_embed = emb.transform(X_test[cols])

    assert np.shape(X_train_embed)[0] == np.shape(X_train)[0]
    assert np.shape(X_test_embed)[0] == np.shape(X_test)[0]

    assert np.shape(X_train_embed)[1] == n_dim
    assert np.shape(X_test_embed)[1] == n_dim


def test_multiclass_classification():
    df = pd.read_csv(
        "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
    )
    cols = df.columns[:-1]
    target = df.columns[-1]
    num_classes = df[target].nunique()

    X_train, X_test = train_test_split(
        df, test_size=0.2, stratify=df[target], random_state=42
    )

    n_dim = 20
    emb = LightgbmEmbedding(n_dim=n_dim)
    emb.fit(X_train[cols], X_train[target])

    X_train_embed = emb.transform(X_train[cols])
    X_test_embed = emb.transform(X_test[cols])

    assert np.shape(X_train_embed)[0] == np.shape(X_train)[0]
    assert np.shape(X_test_embed)[0] == np.shape(X_test)[0]

    assert np.shape(X_train_embed)[1] == n_dim * num_classes
    assert np.shape(X_test_embed)[1] == n_dim * num_classes


def test_regression():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    )
    cols = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "ocean_proximity",
    ]
    target = "median_house_value"
    df["ocean_proximity"] = df["ocean_proximity"].astype("category")

    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

    n_dim = 20
    emb = LightgbmEmbedding(n_dim=20)
    emb.fit(X_train[cols], X_train[target], target_type="continuous")

    X_train_embed = emb.transform(X_train[cols])
    X_test_embed = emb.transform(X_test[cols])

    assert np.shape(X_train_embed)[0] == np.shape(X_train)[0]
    assert np.shape(X_test_embed)[0] == np.shape(X_test)[0]

    assert np.shape(X_train_embed)[1] == n_dim
    assert np.shape(X_test_embed)[1] == n_dim
