import pandas as pd
import numpy as np
all_df_of_close_data = pd.read_csv('data/all_df_of_close_data.csv')

# change string into bool, D = direct, I = indirect
all_df_of_close_data['direct_ownership'] = (
    all_df_of_close_data['ownershipNature.directOrIndirectOwnership'].eq('D')
).astype('int8')

# month variable
all_df_of_close_data['transaction_month'] = pd.DatetimeIndex(
    all_df_of_close_data['transactionDate']).month

# recession dummy from https://fred.stlouisfed.org/release?rid=242
recession_dummy = pd.read_csv('data/recession_dummy/daily,_7-day.csv')
recession_dummy['USRECD'] = recession_dummy['USRECD'].astype('int8')
all_df_of_close_data['transactionDate'] = (
    pd.to_datetime(all_df_of_close_data['transactionDate'], utc=True)
)
recession_dummy['observation_date'] = pd.to_datetime(
    recession_dummy['observation_date'], utc=True).dt.normalize()
all_df_of_close_data = (
    all_df_of_close_data
    .merge(
        recession_dummy,
        left_on="transactionDate",
        right_on="observation_date",
        how="left",
    )
    .drop(columns=["observation_date"])
)

# count of fillings per person
# clean names becouse no id exported
all_df_of_close_data['reportingOwner.name'] = (
    all_df_of_close_data['reportingOwner.name']
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
    right=count_trades_tbl.rename(
        columns={"count": "filing_count_reportingOwner.name"},
    ),
    on="reportingOwner.name",
    how="left",
)

# high freuency trader
median = all_df_of_close_data['filing_count_reportingOwner.name'].median()

all_df_of_close_data['high_frequency_trader'] = (
    all_df_of_close_data['filing_count_reportingOwner.name']
    .gt(median)
    .astype('int8')
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

# Clusterbuy dummy
all_df_of_close_data['clusterbuy'] = (
    all_df_of_close_data['trades_14d'].gt(1)
    .astype('int8')
)

# high price dummy
median = all_df_of_close_data['0'].median()

all_df_of_close_data['high_price'] = (
    all_df_of_close_data['0']
    .gt(median)
    .astype('int8')
)
# postTransactionAmounts.sharesOwnedFollowingTransaction is the amount of
# shares after each filling
post_shares = 'postTransactionAmounts.sharesOwnedFollowingTransaction'
shares = 'amounts.shares'

holdings_before_filing = (
    all_df_of_close_data[post_shares]
    - all_df_of_close_data[shares]
)

# calculation: (amount of shares in this filling/old amount holdings)*100 for
# percent to know how much the person baught in comparison to what they owned
# old amount of shares = post_shares - shares
# if old amount of shares = 0 then division by 0 would cause problems
all_df_of_close_data['holding_change_percent'] = np.where(
    holdings_before_filing == 0, 0, (all_df_of_close_data['amounts.shares'] /
                                     holdings_before_filing) * 100)

# remove inplausible data (132 cases)
all_df_of_close_data = all_df_of_close_data[
    all_df_of_close_data['holding_change_percent'] >= 0]

# high freqency trader
all_df_of_close_data['high_frequency_trader'] = (
    all_df_of_close_data['filing_count_reportingOwner.name'] >
    all_df_of_close_data['filing_count_reportingOwner.name']
    .median()).astype('int8')

# high change in holdings dummy
all_df_of_close_data['high_change_in_holdings'] = (
    all_df_of_close_data['holding_change_percent'] >
    all_df_of_close_data['holding_change_percent']
    .median()).astype('int8')

# Target analysis
days = [4, 198]  # interesting spike after 4 days and maximum return on day 198
target_help_cols = [f'percent_change_since_{d}d' for d in days]

# calculate percent change from day 4 and 198 to filling date
for d in days:
    col = f'percent_change_since_{d}d'
    all_df_of_close_data[col] = (
        (all_df_of_close_data[str(d)] - all_df_of_close_data['0'])
        / all_df_of_close_data['0'] * 100
    )

# compute target variables
target_help_cols = [f'percent_change_since_{d}d' for d in days]

for p in range(1, 11):  # .gt() -> >=
    target = all_df_of_close_data[target_help_cols].ge(p).astype('int8')
    target.columns = [f't_{p}_{c}' for c in target_help_cols]
    all_df_of_close_data[target.columns] = target

# find most interesting target variables
target_cols = [f't_{p}_percent_change_since_{d}d'
               for d in days for p in range(1, 11)]

n = len(all_df_of_close_data)
ones = all_df_of_close_data[target_cols].sum().astype(int)
zeros = n - ones

summary = pd.DataFrame({
    'Target Name': target_cols,
    'ones':  ones.values,
    'zeros': zeros.values,
    'ones_rate':  (ones / n).values,
})

# keep only relevant targets
target_cols = [f't_{p}_percent_change_since_{d}d'
               for d in days for p in range(1, 11)]
target_cols.remove('t_1_percent_change_since_4d')
target_cols.remove('t_10_percent_change_since_198d')
all_df_of_close_data = all_df_of_close_data.drop(columns=target_cols)
