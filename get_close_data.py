import pandas as pd
import yfinance as yf
import datetime
import numpy as np


def get_close_data(filename='2018-01'):
    # read insider data 
    insider_data = pd.read_csv(f'data/relevant_{filename}.csv',
                            delimiter=',')

    # get financal information for each ticker
    ticker = set(insider_data['issuer.tradingSymbol'])
    start_date = min(insider_data['transactionDate'])
    close_data = yf.download(ticker, start=start_date, period='2y')[['Close']]
    close_data.columns = close_data.columns.droplevel(0)
    close_data = close_data.reset_index()
    close_data['Date'] = pd.to_datetime(close_data['Date']).dt.date

    tickers = close_data.columns[1:]
    close_data_rel = pd.DataFrame()

    for ticker in tickers:
        # find filling date for ticker in insider_data
        filing_date = pd.to_datetime(
            insider_data.loc[insider_data['issuer.tradingSymbol'] == ticker,
                            'transactionDate'].iloc[0]
        ).date()

        # find prices in close_data relative to filling date
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

    close_data = pd.merge(insider_data, close_data_rel, left_on='issuer.tradingSymbol',
                        right_on='ticker', how='inner').drop('ticker', axis=1)

    close_data.to_csv(f'data/close_data_{filename}.csv',
                    header=True, index=False)


def concat_close_data():
    # list of all names of close_data
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    years = ['09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
    all_names_of_close_data_files = []
    for year in years:
        for month in months:
            name_of_close_data_file = 'close_data_20'+ year + '-' + month
            all_names_of_close_data_files.append(name_of_close_data_file)

    # no data later than 2023-08, therefore removed from list
    all_names_of_close_data_files = all_names_of_close_data_files[:-4]

    # concatinate all data
    all_df_of_close_data = pd.DataFrame()
    expected_final_row_number = 0
    for name in all_names_of_close_data_files:
        df = pd.read_csv(f'data/{name}.csv', index=False)
        all_df_of_close_data = pd.concat([all_df_of_close_data, df], ignore_index=True)
        expected_final_row_number += df.shape[0]

    # remove tickers that are not found in yfinace or have no complete close data
    for num in range(364):
        all_df_of_close_data = all_df_of_close_data[all_df_of_close_data[f'{num}'].notna()].copy()
    
    # remove rest of NAs because we have enough data anyways
    all_df_of_close_data = all_df_of_close_data[all_df_of_close_data['postTransactionAmounts.sharesOwnedFollowingTransaction'].notna()].copy()

    # remove inplausible data
    all_df_of_close_data = all_df_of_close_data[ pd.to_datetime(all_df_of_close_data['transactionDate'])< pd.Timestamp('2023-09-01')]

    return all_df_of_close_data
