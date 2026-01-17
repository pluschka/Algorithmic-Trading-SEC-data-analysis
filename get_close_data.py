import pandas as pd
import yfinance as yf
import datetime
import numpy as np


def get_close_data(month='2018-01'):
    # load insider data
    insider_data = pd.read_csv(f'data/relevant_{month}.csv',
                               delimiter=',')

    # get financial information for each ticker
    ticker = set(insider_data['issuer.tradingSymbol'])
    start_date = min(insider_data['transactionDate'])
    close_data = yf.download(ticker, start=start_date, period='2y')[['Close']]
    close_data.columns = close_data.columns.droplevel(0)
    close_data = close_data.reset_index()
    close_data['Date'] = pd.to_datetime(close_data['Date']).dt.date

    tickers = close_data.columns[1:]
    close_data_rel = pd.DataFrame()

    for ticker in tickers:
        # find filing date for ticker in insider_data
        filing_date = pd.to_datetime(
            insider_data.loc[insider_data['issuer.tradingSymbol'] == ticker,
                             'transactionDate'].iloc[0]
        ).date()

        # find prices in close_data relative to filing date
        row_data = {}
        for num in range(365):
            date = filing_date + datetime.timedelta(days=num)
            price_series = close_data.loc[close_data['Date'] == date, ticker]
            value = price_series.squeeze() if not price_series.empty else None

            # select last price if no values for this date
            if not isinstance(value, (float, np.floating)):
                if num > 0:
                    value = row_data[f"{num-1}"]
                else:
                    value = None

            row_data[f"{num}"] = value

        # add join key
        row_data["ticker"] = ticker

        df = pd.DataFrame([row_data])
        close_data_rel = pd.concat([close_data_rel, df], ignore_index=True)

    close_data = pd.merge(insider_data, close_data_rel,
                          left_on='issuer.tradingSymbol',
                          right_on='ticker',
                          how='inner').drop('ticker', axis=1)

    close_data.to_csv(f'data/close_data_{month}.csv',
                      header=True,
                      index=False)


def concat_close_data():
    # list names of close_data
    months = ['01', '02', '03', '04', '05', '06',
              '07', '08', '09', '10', '11', '12']
    years = ['09', '10', '11', '12', '13', '14', '15', '16',
             '17', '18', '19', '20', '21', '22', '23']
    all_names_of_close_data_files = []
    for year in years:
        for month in months:
            name_of_close_data_file = 'close_data_20' + year + '-' + month
            all_names_of_close_data_files.append(name_of_close_data_file)

    # no data later than 2023-08, therefore removed from list
    all_names_of_close_data_files = all_names_of_close_data_files[:-4]

    # concatenate all data
    all_df_of_close_data = pd.DataFrame()
    expected_final_row_number = 0
    for name in all_names_of_close_data_files:
        df = pd.read_csv(f'data/{name}.csv', index=False)
        all_df_of_close_data = pd.concat([all_df_of_close_data, df],
                                         ignore_index=True)
        expected_final_row_number += df.shape[0]

    # remove tickers that are not found in yfinace or lacking complete data
    for num in range(364):
        all_df_of_close_data = all_df_of_close_data[
            all_df_of_close_data[f'{num}'].notna()].copy()

    # drop NAs because we have enough data anyways
    all_df_of_close_data = all_df_of_close_data[all_df_of_close_data[
        'postTransactionAmounts.sharesOwnedFollowingTransaction']
        .notna()].copy()

    # remove implausible data
    all_df_of_close_data = all_df_of_close_data[pd.to_datetime(
        all_df_of_close_data['transactionDate']) < pd.Timestamp('2023-09-01')]

    return all_df_of_close_data


#### GPT-5.2 Make over:

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def _ensure_row_id(df: pd.DataFrame) -> pd.DataFrame:
    if "row_id" in df.columns:
        return df

    key_cols = [
        "issuer.tradingSymbol",
        "reportingOwner.name",
        "transactionDate",
        "amounts.shares",
        "amounts.pricePerShare",
    ]
    tmp = df.reindex(columns=key_cols).copy()

    tmp["transactionDate"] = pd.to_datetime(tmp["transactionDate"], errors="coerce").dt.strftime("%Y-%m-%d")
    tmp["amounts.shares"] = pd.to_numeric(tmp["amounts.shares"], errors="coerce").round(6)
    tmp["amounts.pricePerShare"] = pd.to_numeric(tmp["amounts.pricePerShare"], errors="coerce").round(6)

    row_hash = pd.util.hash_pandas_object(tmp.fillna(""), index=False).astype("uint64").astype(str)
    dup_idx = row_hash.groupby(row_hash).cumcount().astype(str)

    df = df.copy()
    df["row_id"] = row_hash + ":" + dup_idx
    return df


def _download_close_matrix(tickers, start, end) -> pd.DataFrame:
    tickers = sorted({t for t in tickers if isinstance(t, str) and t.strip()})
    if not tickers:
        return pd.DataFrame()

    px = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    if px.empty:
        return pd.DataFrame()

    # Multi-ticker -> MultiIndex columns: (ticker, field)
    if isinstance(px.columns, pd.MultiIndex):
        close = px.xs("Close", axis=1, level=1, drop_level=True)
    else:
        # Single ticker -> columns like ["Open","High","Low","Close",...]
        close = px[["Close"]].rename(columns={"Close": tickers[0]})

    close.index = pd.to_datetime(close.index, errors="coerce").normalize()
    close = close.sort_index()
    return close


def get_close_data(month: str = "2018-01", horizon_days: int = 365) -> pd.DataFrame:
    insider_data = pd.read_csv(f"data/relevant_{month}.csv")
    insider_data = _ensure_row_id(insider_data)

    insider_data["issuer.tradingSymbol"] = (
        insider_data["issuer.tradingSymbol"]
        .astype(str)
        .str.upper()
        .str.strip()
    )
    insider_data["transactionDate"] = pd.to_datetime(insider_data["transactionDate"], errors="coerce").dt.normalize()
    insider_data = insider_data.dropna(subset=["issuer.tradingSymbol", "transactionDate", "row_id"])

    tickers = insider_data["issuer.tradingSymbol"].unique().tolist()

    min_date = insider_data["transactionDate"].min()
    max_date = insider_data["transactionDate"].max()
    start = (min_date - pd.Timedelta(days=7)).date()
    end = (max_date + pd.Timedelta(days=horizon_days + 7)).date()

    close_mat = _download_close_matrix(tickers, start=start, end=end)
    if close_mat.empty:
        out = insider_data.copy()
        out.to_csv(f"data/close_data_{month}.csv", index=False)
        return out

    # build relative close rows for each unique (ticker, transactionDate)
    unique_keys = insider_data[["issuer.tradingSymbol", "transactionDate"]].drop_duplicates()
    rel_rows = []

    for _, r in unique_keys.iterrows():
        t = r["issuer.tradingSymbol"]
        d0 = r["transactionDate"]

        if t not in close_mat.columns:
            continue

        s = close_mat[t].dropna()
        if s.empty:
            continue

        dates = pd.date_range(d0, periods=horizon_days, freq="D")
        vals = s.reindex(dates, method="ffill")  # forward fill to last known close
        row = {str(i): v for i, v in enumerate(vals.to_numpy())}
        row["issuer.tradingSymbol"] = t
        row["transactionDate"] = d0
        rel_rows.append(row)

    close_rel = pd.DataFrame(rel_rows)

    out = insider_data.merge(
        close_rel,
        on=["issuer.tradingSymbol", "transactionDate"],
        how="inner",
        validate="many_to_one",
    )

    out.to_csv(f"data/close_data_{month}.csv", index=False)
    return out


def concat_close_data(data_dir: str = "data") -> pd.DataFrame:
    p = Path(data_dir)
    files = sorted(p.glob("close_data_????-??.csv"))

    all_df = []
    for f in files:
        df = pd.read_csv(f)
        all_df.append(df)

    if not all_df:
        return pd.DataFrame()

    out = pd.concat(all_df, ignore_index=True)

    # require complete horizon columns (0..363) present and non-null
    horizon_cols = [str(i) for i in range(364)]
    existing = [c for c in horizon_cols if c in out.columns]
    if existing:
        out = out[out[existing].notna().all(axis=1)].copy()

    out = out[out["postTransactionAmounts.sharesOwnedFollowingTransaction"].notna()].copy()
    out["transactionDate"] = pd.to_datetime(out["transactionDate"], errors="coerce")
    out = out[out["transactionDate"] < pd.Timestamp("2023-09-01")].copy()
    out.to_csv("data/2026_01/sec_close.csv", index=False)

    return out
