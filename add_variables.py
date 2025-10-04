import pandas as pd
import numpy as np
all_df_of_close_data = pd.read_csv('data/all_df_of_close_data.csv')

# month variable
all_df_of_close_data['transaction_month'] = pd.DatetimeIndex(all_df_of_close_data['transactionDate']).month

# recession dummy from https://fred.stlouisfed.org/release?rid=242
recession_dummy = pd.read_csv('data/recession_dummy/daily,_7-day.csv')
recession_dummy['USRECD'] = recession_dummy['USRECD'].astype('int8')
all_df_of_close_data = all_df_of_close_data.merge(recession_dummy,
                                                  left_on='transactionDate',
                                                  right_on='observation_date',
                                                  how="left").drop(columns=['observation_date'])

# count of fillings per person
# clean names becouse no id exported
all_df_of_close_data['reportingOwner.name'] = (all_df_of_close_data['reportingOwner.name']
             .str.replace(r'[^A-Za-z ]+', '', regex=True)
             .str.replace(r'\s+', ' ', regex=True)
             .str.strip()
             .str.upper())


count_trades_tbl = (all_df_of_close_data
       .groupby('reportingOwner.name')['issuer.tradingSymbol']
       .count()
       .rename('count')
       .reset_index())

all_df_of_close_data = all_df_of_close_data.merge(
    count_trades_tbl.rename(columns={"count": "filing_count_reportingOwner.name"}),
    on="reportingOwner.name", how="left"
)

# Clusterbuys in past 14 days
df = all_df_of_close_data.copy()
df['transactionDate'] = pd.to_datetime(df['transactionDate'], errors='coerce')

# calculate daily fillings of ticker
daily = (df.groupby(['issuer.tradingSymbol', 'transactionDate']).size()
           .rename('n').reset_index())

# rolling count for past 14 days of fillings for this ticker
roll = (daily.set_index('transactionDate')
              .groupby('issuer.tradingSymbol')['n']
              .rolling('14D').sum()
              .rename('trades_14d')
              .reset_index())

df = df.merge(roll, on=['issuer.tradingSymbol', 'transactionDate'], how='left')

all_df_of_close_data = df

# high price dummy
all_df_of_close_data['high_price'] = (all_df_of_close_data['0'] > all_df_of_close_data['0'].median()).astype('int8')