import pandas as pd
import numpy as np


df0 = pd.read_csv('data/inner_close_sec.csv')
keep_cols = [c for c in df0.columns if not (str(c).isdigit() and 0 <= int(c) <= 750)]
df = df0.loc[:, keep_cols]

# change string into bool, D = direct, I = indirect
df['direct_ownership'] = (
    df['ownershipNature.directOrIndirectOwnership'].eq('D')
).astype('int8')

# for model readability month_sin and month_cos
df['transaction_month'] = pd.DatetimeIndex(
    df['transactionDate']).month
df["month_sin"] = np.sin(2 * np.pi * df["transaction_month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["transaction_month"] / 12)

"""
plt.plot(df["month_sin"], df["month_cos"])
plt.title("Transformation of transaction_month in month_sin and month_cos")
plt.xlabel("month_sin")
plt.ylabel("month_cos")
plt.show()

df = df.drop(columns="transaction_month")

"""

# count of fillings per person
# clean names because no id exported
df['reportingOwner.name'] = (
    df['reportingOwner.name']
    .str.replace(r'[^A-Za-z ]+', '', regex=True)
    .str.replace(r'\s+', ' ', regex=True)
    .str.strip()
    .str.upper())

count_trades_tbl = (df
                    .groupby('reportingOwner.name')['issuer.tradingSymbol']
                    .count()
                    .rename('count')
                    .reset_index())
df = df.merge(
    right=count_trades_tbl.rename(
        columns={"count": "filing_count_reportingOwner.name"},
    ),
    on="reportingOwner.name",
    how="left",
)

# high frequency trader
median = df['filing_count_reportingOwner.name'].median()

df['high_frequency_trader'] = (
    df['filing_count_reportingOwner.name']
    .gt(median)
    .astype('int8')
)

# Cluster buys in past 14 days
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

# Cluster buys dummy
df['cluster_buy'] = (
    df['trades_14d'].gt(1)  # .gt() -> grater than
    .astype('int8')
)

# high price dummy
median = df['0'].median()

df['high_price'] = (
    df['0']
    .gt(median)
    .astype('int8')
)

# postTransactionAmounts.sharesOwnedFollowingTransaction is the amount of
# shares after each filling
post_shares = 'postTransactionAmounts.sharesOwnedFollowingTransaction'
shares = 'amounts.shares'

holdings_before_filing = (
    df[post_shares]
    - df[shares]
)

# calculation: (amount of shares in this filling/pre amount holdings)*100 for
# percent to know how much the person bought in comparison to what they owned
# pre amount of shares = post_shares - shares
# if old amount of shares = 0 then division by 0 would cause problems
df['holding_change_percent'] = np.where(
    holdings_before_filing == 0, 0, (df['amounts.shares'] /
                                     holdings_before_filing) * 100)
# remove implausible data
df = df[
    df['holding_change_percent'] >= 0]


# high change in holdings dummy
df['high_change_in_holdings'] = (
    df['holding_change_percent'] >
    df['holding_change_percent']
    .median()).astype('int8')

df.to_csv("data/final_final_df.csv")