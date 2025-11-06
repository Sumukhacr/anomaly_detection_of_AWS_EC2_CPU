
import matplotlib.pyplot as plt

def plot_anomalies(df, method_col, title):
    plt.figure(figsize=(12,4))
    plt.plot(df.index, df["value"], label="Value")
    plt.scatter(df.index[df[method_col]==1], df.loc[df[method_col]==1,"value"], 
                color="red", marker="x", label="Anomaly")
    plt.title(title)
    plt.legend()
    plt.show()
