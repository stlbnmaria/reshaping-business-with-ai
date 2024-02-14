from datetime import timedelta
from typing import Tuple

import pandas as pd

from config.config import DATA_FILE, THRESHOLD


def get_exogenous_data(
    df_sum: pd.DataFrame, max_stamp: pd.Timestamp, min_stamp: pd.Timestamp
) -> pd.DataFrame:
    data = df_sum.loc[
        (df_sum.date_order >= min_stamp) & (df_sum.date_order < max_stamp)
    ]
    data = data.drop(columns="date_order")
    data = data.groupby("client_id").sum()

    return data


def get_historical_buys_ratio(
    df_sum: pd.DataFrame, X: pd.DataFrame, max_stamp: pd.Timestamp
) -> pd.DataFrame:
    all_buys = (
        df_sum.loc[
            (df_sum.date_order < max_stamp),
            ["client_id", "sales_net", "quantity", "nr_orders"],
        ]
        .groupby("client_id")
        .sum()
    )
    all_buys.columns = ["perc_sales_net", "perc_quantity", "perc_nr_orders"]
    X = X.merge(all_buys, left_index=True, right_index=True, how="left")
    X["perc_sales_net"] = X["sales_net"] / X["perc_sales_net"]
    X["perc_quantity"] = X["quantity"] / X["perc_quantity"]
    X["perc_nr_orders"] = X["nr_orders"] / X["perc_nr_orders"]
    X = X.drop(columns=["nr_orders"])

    return X


def get_avg_buy_time(
    df_sum: pd.DataFrame, X: pd.DataFrame, max_stamp: pd.Timestamp
) -> pd.DataFrame:
    avg_buy_time = df_sum[df_sum.date_order < max_stamp].copy()

    # Sort data by 'client_id' and 'date_order' and drop multiple orders per day
    avg_buy_time = avg_buy_time.sort_values(["client_id", "date_order"])
    df_day = avg_buy_time.drop_duplicates(["date_order", "client_id"])

    # Calculate the time difference between consecutive purchases for each customer
    df_day["avg_time_purchase"] = df_day.groupby("client_id")[
        "date_order"
    ].diff()

    # Calculate the mean over times per customer
    time_to_buy = (
        df_day.groupby("client_id")["avg_time_purchase"].mean().dt.days
    )

    X = X.merge(
        time_to_buy, left_on="client_id", right_index=True, how="left"
    ).fillna(0)

    return X


def get_returns(
    df: pd.DataFrame,
    X: pd.DataFrame,
    max_stamp: pd.Timestamp,
    min_stamp: pd.Timestamp = None,
) -> pd.DataFrame:
    if not min_stamp:
        min_stamp = df.date_order.min()
        out_cols = None
    else:
        out_cols = ["client_id", "return_count_now", "ret_pct_now"]

    returns = df[
        (df.date_order >= min_stamp) & (df.date_order < max_stamp)
    ].copy()

    # Count the number of returns for each day, client, and product
    return_counts = (
        returns[returns["sales_net"] < 0]
        .groupby(["client_id"])
        .size()
        .rename("return_count")
        .reset_index()
    )

    # Count the total number of purchases for each day and each client
    total_purchases = (
        returns[returns.sales_net > 0]
        .groupby(["client_id"])
        .size()
        .rename("total_purchases")
        .reset_index()
    )

    # Merge return_counts and total_purchases back to the main DataFrame on 'client_id'
    client_returns = return_counts.merge(
        total_purchases, on="client_id", how="outer"
    )

    # Fill NaN values with 0 (for pairs with no client_returns)
    client_returns["return_count"] = client_returns["return_count"].fillna(0)
    client_returns["total_purchases"] = client_returns[
        "total_purchases"
    ].fillna(0)

    # Calculate return percentage (ret_pct) for each day and each client-product pair
    client_returns["ret_pct"] = (
        client_returns["return_count"] / client_returns["total_purchases"]
    ).fillna(0)

    client_returns = client_returns.drop(columns=["total_purchases"])
    if out_cols:
        client_returns.columns = out_cols

    X = X.merge(client_returns, on="client_id", how="left")

    return X


def dataloader(stamp: pd.Timestamp = None) -> Tuple[pd.DataFrame]:
    df = pd.read_csv(DATA_FILE, sep=";")
    df["date_order"] = pd.to_datetime(df["date_order"])

    if stamp:
        df = df[df.date_order < stamp].copy()

    # get a list of all clients in data
    all_clients = df.client_id.unique()

    # filter only for sales and not returns
    df_ml = df[df.sales_net > 0].copy()

    # get dummies for order channel to count orders per channel
    df_ml = pd.concat(
        [df_ml, pd.get_dummies(df_ml["order_channel"], dtype=int)], axis=1
    )
    df_ml["nr_orders"] = 1

    # drop unnecessary columns
    df_ml = df_ml.drop(
        columns=["date_invoice", "branch_id", "order_channel", "product_id"]
    )

    # create daily sum by customer
    df_sum = df_ml.groupby(["date_order", "client_id"]).sum().reset_index()

    ########################
    # create test set
    test_stamp = df_sum.date_order.max() - timedelta(days=THRESHOLD)
    train_stamp = test_stamp - timedelta(days=THRESHOLD)

    customers_test = df_sum[df_sum.date_order >= test_stamp].client_id.unique()
    y_test = pd.DataFrame({"client_id": all_clients})
    y_test["churn"] = ~y_test.client_id.isin(customers_test)

    X_test = get_exogenous_data(df_sum, test_stamp, train_stamp)

    X_test = get_historical_buys_ratio(df_sum, X_test, test_stamp)

    X_test = get_avg_buy_time(df_sum, X_test, test_stamp)

    # get all returns in the last 2 years
    X_test = get_returns(df, X_test, test_stamp, min_stamp=None)

    # get returns for the last x days
    X_test = get_returns(df, X_test, test_stamp, min_stamp=train_stamp)

    # merge label to exogenous features to have the same clients
    test = X_test.merge(y_test, on="client_id", how="left")

    ######################
    # create train set
    y_train = pd.DataFrame({"client_id": all_clients})
    y_train["churn"] = ~y_train.client_id.isin(test.client_id)

    lowest_stamp = train_stamp - timedelta(days=THRESHOLD)
    X_train = get_exogenous_data(df_sum, train_stamp, lowest_stamp)

    X_train = get_historical_buys_ratio(df_sum, X_train, train_stamp)

    X_train = get_avg_buy_time(df_sum, X_train, train_stamp)

    # get all returns in the last 2 years
    X_train = get_returns(df, X_train, train_stamp, min_stamp=None)

    # get returns for the last x days
    X_train = get_returns(df, X_train, train_stamp, min_stamp=lowest_stamp)

    # merge label to exogenous features to have the same clients
    train = X_train.merge(y_train, on="client_id", how="left")

    print(" Train shape", train.shape)
    print(f" Train Churn Rate {train.churn.mean():.2f}")
    print(" Test shape", test.shape)
    print(f" Test Churn Rate {test.churn.mean():.2f}")
    print(" -----------------------")

    return train, test, test_stamp
