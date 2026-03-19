import pandas as pd

def load_and_clean_data(path):
    df = pd.read_csv(path)

    df.columns = [
        "house_no",
        "space",
        "rooms",
        "floors",
        "location",
        "price"
    ]

    df = df.dropna()
    df = pd.get_dummies(df, columns=["location"], drop_first=True)

    return df