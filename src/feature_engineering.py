def add_features(df):
    df["price_per_sqft"] = df["price"] / df["space"]
    df["room_density"] = df["rooms"] / df["space"]

    df = df.drop("house_no", axis=1)

    return df