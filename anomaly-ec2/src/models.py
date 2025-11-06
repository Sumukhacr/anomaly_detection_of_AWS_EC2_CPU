
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from tensorflow import keras
from keras import layers

def isolation_forest(df: pd.DataFrame, feature_cols, contamination=0.01):
    X = df[feature_cols].values
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=42)
    iso.fit(X)
    scores = -iso.score_samples(X)
    threshold = np.percentile(scores, 99)
    df["iso_score"] = scores
    df["anomaly_iso"] = (scores >= threshold).astype(int)
    return df, iso

def lstm_autoencoder(df: pd.DataFrame, feature_cols, seq_len=24, epochs=10, batch_size=64):
    X = df[feature_cols].values
    seq_data = np.array([X[i:i+seq_len] for i in range(len(X)-seq_len)])
    inputs = keras.Input(shape=(seq_len, len(feature_cols)))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(32)(x)
    x = layers.RepeatVector(seq_len)(x)
    x = layers.LSTM(32, return_sequences=True)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    outputs = layers.TimeDistributed(layers.Dense(len(feature_cols)))(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    model.fit(seq_data, seq_data, epochs=epochs, batch_size=batch_size, verbose=0)

    recon = model.predict(seq_data, verbose=0)
    mse = ((seq_data - recon)**2).mean(axis=(1,2))
    idx = df.index[seq_len:]
    df.loc[idx, "ae_score"] = mse
    thr = np.percentile(mse, 99)
    df["anomaly_ae"] = (df["ae_score"] >= thr).astype(int)
    return df, model
