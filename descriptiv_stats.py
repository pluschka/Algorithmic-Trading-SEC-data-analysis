import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
from data_cleaning import outlier_strategy

# plot total price development of a 300 sample
all_df_of_close_data = pd.read_csv('data/all_df_of_close_data.csv')

# remove implausible data
for num in range(364):
    all_df_of_close_data = all_df_of_close_data[
        all_df_of_close_data[f'{num}'] >= 0]

# remove upper outliers
df = all_df_of_close_data.copy()
for num in range(364):
    q75, q25 = np.quantile(df[f'{num}'], [0.75, 0.25])
    iqr = q75 - q25
    df = df[df[f'{num}'] <= q75 + 1.5 * iqr]

all_df_of_close_data = df
df = all_df_of_close_data.copy()

day_cols = [str(i) for i in range(364)]

# data is to large, plot only a n=300 sample
sample = df.sample(n=300, random_state=69)

# with matrix or else warning
abs_mat = sample[day_cols].to_numpy(dtype=float)

# calculate demeaned values
row_mean = np.nanmean(abs_mat, axis=1, keepdims=True)
demean_mat = abs_mat - row_mean

# calculate percentage since day 0
ref = sample[day_cols[0]].to_numpy(dtype=float)[:, None]
ref = np.where(ref == 0, np.nan, ref)
percent_mat = (abs_mat - ref) / ref

x = np.arange(364)

# plot absolute, demeaned and percentage in one pic
fig, axs = plt.subplots(1, 3, figsize=(22, 6), sharex=True)

axs[0].plot(x, abs_mat.T, linewidth=0.6, alpha=0.2)
axs[0].set_title('Absolute')
axs[0].set_xlabel('Days since Filing Date')
axs[0].set_ylabel('Close Price')

axs[1].plot(x, demean_mat.T, linewidth=0.6, alpha=0.2)
axs[1].set_title('Demeaned')
axs[1].set_xlabel('Days since Filing Date')
axs[1].set_ylabel('Close Price (demeaned)')

axs[2].plot(x, percent_mat.T, linewidth=0.6, alpha=0.2)
axs[2].set_title('Percent')
axs[2].set_xlabel('Days since Filing Date')
axs[2].set_ylabel('Percent since Day 0')

for ax in axs:
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.savefig('exports/descriptive_random_state_69.png', dpi=500)
plt.show()


# Mean Returns for one share each filling
x_0 = all_df_of_close_data['0'].to_numpy()[:, None]
day_cols = [str(i) for i in range(1, 364)]
Xt = all_df_of_close_data[day_cols].to_numpy()
return_matrix = Xt - x_0
return_df = pd.DataFrame(return_matrix)
mean_return = return_df.mean()

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.plot(mean_return)
ax.set_title('Mean Returns for one share each filling')
ax.set_xlabel('Days')
ax.set_ylabel('Mean of Price_0 - Price_t\nin $')

for s in ax.spines.values():
    s.set_visible(False)

plt.tight_layout()
plt.show()


# Total Mean Returns: actual return with amount of shares
n = all_df_of_close_data['amounts.shares'].to_numpy()[:, None]
x_0 = all_df_of_close_data['0'].to_numpy()[:, None]
day_cols = [str(i) for i in range(1, 364)]
Xt = all_df_of_close_data[day_cols].to_numpy()
return_matrix = (Xt - x_0)*n
return_df = pd.DataFrame(return_matrix)
mean_return = return_df.mean()

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.plot(mean_return)
ax.set_title('Total Mean Returns')
ax.set_xlabel('Days')
ax.set_ylabel('Mean of return times the amount of shares in $')

for s in ax.spines.values():
    s.set_visible(False)

plt.tight_layout()
plt.show()


# Histogram of shares
relevant_data_uncleaned = pd.read_csv('data/relevant_data_uncleaned.csv')
relevant_data_uncleaned_outliers_removed = \
    outlier_strategy(relevant_data_uncleaned,
                     default_outlier_strategy="delete",
                     except_replace_0=[],
                     except_replace_mean=[],
                     except_delete=[],
                     ignore=[])
relevant_data_uncleaned_outliers_removed.shape
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
fig.suptitle("Histogram of Number of Shares", fontsize=14)
ax.hist(relevant_data_uncleaned_outliers_removed['amounts.shares'], bins=50)
ax.set_xlabel('Amount of shares')
ax.set_ylabel('Count')

median = relevant_data_uncleaned_outliers_removed['amounts.shares'].median()

for s in ax.spines.values():
    s.set_visible(False)
ax.axvline(median, linestyle="--", lw=1, c="red")
ax.text(median,
        ax.get_ylim()[1]*+1.12,
        f"median\n{median:.0f}",
        va="top",
        ha="center",
        color="red")

mean = relevant_data_uncleaned_outliers_removed['amounts.shares'].mean()

for s in ax.spines.values():
    s.set_visible(False)
ax.axvline(mean, linestyle="--", lw=1, c="green")
ax.text(mean,
        ax.get_ylim()[1]*+1.12,
        f"mean\n{mean:.0f}",
        va="top",
        ha="center",
        color="green")

plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=2.0)
plt.show()


# histogram of price
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
fig.suptitle("Histogram of Number of Price per Share", fontsize=14)
ax.hist(relevant_data_uncleaned_outliers_removed['amounts.pricePerShare'],
        bins=50)
ax.set_xlabel('Amount of Price per share')
ax.set_ylabel('Count')

median = relevant_data_uncleaned_outliers_removed['amounts.pricePerShare']\
    .median()

for s in ax.spines.values():
    s.set_visible(False)

ax.axvline(median, linestyle="--", lw=1, c="red")
ax.text(median,
        ax.get_ylim()[1] * 1.12,
        f"median\n{median:.0f}",
        va="top", ha="right", color="red", clip_on=False)

mean = relevant_data_uncleaned_outliers_removed['amounts.pricePerShare'].mean()

for s in ax.spines.values():
    s.set_visible(False)
ax.axvline(mean, linestyle="--", lw=1, c="green")
ax.text(mean,
        ax.get_ylim()[1] * 1.12,
        f"mean\n{mean:.0f}",
        va="top", ha="left", color="green", clip_on=False)

plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=2.0)
plt.show()


# Histogram of Post Transaction Holdings of Issuer
fig, ax = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle("Histogram of Post Transaction Holdings of Issuer", fontsize=14)
ax[0].hist(relevant_data_uncleaned_outliers_removed['postTransactionAmounts.sharesOwnedFollowingTransaction'], bins=100)
ax[0].set_title('Outlier removed once', y=1.1)
ax[0].set_xlabel('Amount of Holdings after Filing')
ax[0].set_ylabel('Count')

# remove upper outliers in postTransactionAmounts.sharesOwnedFollowingTransaction for better look
df = relevant_data_uncleaned_outliers_removed.copy()
q75, q25 = np.quantile(df['postTransactionAmounts.sharesOwnedFollowingTransaction'], [0.75, 0.25])
iqr = q75 - q25
df = df[df['postTransactionAmounts.sharesOwnedFollowingTransaction'] <= q75 + 1.5 * iqr]

ax[1].hist(df['postTransactionAmounts.sharesOwnedFollowingTransaction'], bins=100)
ax[1].set_title('Outlier removed twice', y=1.3)
ax[1].set_xlabel('Amount of Holdings after Filing')
ax[1].set_ylabel('Count')


axes = [0, 1]
for axe in axes:
    for s in ax[axe].spines.values():
        s.set_visible(False)
    median = relevant_data_uncleaned_outliers_removed['postTransactionAmounts.sharesOwnedFollowingTransaction'].median()
    ax[axe].axvline(median, linestyle="--", lw=1, c="red")
    ax[axe].text(median,
                 ax[axe].get_ylim()[1]*+1.12,
                 f"median\n{median:.0f}",
                 va="top",
                 ha="center",
                 color="red")

    mean = relevant_data_uncleaned_outliers_removed['postTransactionAmounts.sharesOwnedFollowingTransaction'].mean()
    ax[axe].axvline(mean, linestyle="--", lw=1, c="green")
    ax[axe].text(mean,
                 ax[axe].get_ylim()[1] * 1.12,
                 f"mean\n{mean:.0f}",
                 va="top", ha="center", color="green", clip_on=False)

plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=2.0)
plt.show()


# Distribution of direct_ownership
ser = relevant_data_uncleaned_outliers_removed['direct_ownership']
counts = ser.value_counts(dropna=False).sort_index()

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(counts.index.astype(str), counts.values, width=0.6)

ax.bar_label(bars, labels=[f"{v:,}" for v in counts.values], padding=2)
ax.set_ylabel('Count')
ax.set_title('Distribution of direct_ownership')
for s in ax.spines.values():
    s.set_visible(False)
plt.tight_layout(); plt.show()


# Distribution of is_direct, officer, 10percent, other
vars = ['reportingOwner.relationship.isDirector',
        'reportingOwner.relationship.isOfficer',
        'reportingOwner.relationship.isTenPercentOwner',
        'reportingOwner.relationship.isOther']

titles = ['Director', 'Officer', '10% Owner', 'Other']

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
axes = ax.ravel()

for i, v in enumerate(vars):
    ser = relevant_data_uncleaned_outliers_removed[v]
    counts = ser.value_counts(dropna=False).sort_index()

    for s in axes[i].spines.values():
        s.set_visible(False)

    bars = axes[i].bar(counts.index.astype(str), counts.values, width=0.6)
    axes[i].bar_label(bars, labels=[f"{int(val):,}" for val in counts.values], padding=2)
    axes[i].set_ylabel('Amount of Filings per Insider')
    axes[i].set_title(titles[i])

plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=2.0)
plt.show()


# Does the filing month affect the target
counts_t_1_percent_change_since_4d = (
    all_df_of_close_data
    .groupby(['transaction_month', 't_1_percent_change_since_4d'])
    .size()
    .unstack(fill_value=0))

counts_t_10_percent_change_since_198d = (
    all_df_of_close_data
    .groupby(['transaction_month', 't_10_percent_change_since_198d'])
    .size()
    .unstack(fill_value=0))

months = np.arange(1, 13)
x = np.arange(len(months))
width = 0.4

colors = ["#E1DEDE", 'tab:blue']

fig, ax = plt.subplots(2, 1, figsize=(12, 10))
for i, col in enumerate(counts_t_1_percent_change_since_4d.columns):
    ax[0].bar(x + i*width, counts_t_1_percent_change_since_4d[col].values,
              width,
              label=f"target={col}",
              color=colors[i])
    ax[1].bar(x + i*width, counts_t_10_percent_change_since_198d[col].values,
              width,
              label=f"target={col}",
              color=colors[i])

ax[0].set_ylabel('Count')
ax[0].set_title(
    'Does the filing month affect the target '
    't_1_percent_change_since_4d?'
)
ax[0].set_xticks(x + width/2, months)
ax[0].legend(loc='upper left', ncols=2)
ax[0].set_ylim(0, counts_t_1_percent_change_since_4d.values.max()*1.1)
ax[1].set_ylabel('Count')
ax[1].set_title('Does the filing month affect the target '
                't_10_percent_change_since_198d?')
ax[1].set_xticks(x + width/2, months)
ax[1].legend(loc='upper left', ncols=2)
ax[1].set_ylim(0, counts_t_1_percent_change_since_4d.values.max()*1.1)

for s in ax[0].spines.values():
    s.set_visible(False)

for s in ax[1].spines.values():
    s.set_visible(False)
plt.show()

# Does the economic cycle effect the target
counts_t_1_percent_change_since_4d = (
    all_df_of_close_data
    .groupby(['USRECD', 't_1_percent_change_since_4d'])
    .size()
    .unstack(fill_value=0))

counts_t_10_percent_change_since_198d = (
    all_df_of_close_data
    .groupby(['USRECD', 't_10_percent_change_since_198d'])
    .size()
    .unstack(fill_value=0))

label = ['expansionary period', 'recessionary period']
x = np.arange(len(label))
width = 0.4

colors = ["#E1DEDE", 'tab:blue']

fig, ax = plt.subplots(2, 1, figsize=(12, 10))
for i, col in enumerate(counts_t_1_percent_change_since_4d.columns):
    ax[0].bar(x + i*width, counts_t_1_percent_change_since_4d[col].values,
              width,
              label=f"target={col}",
              color=colors[i])
    ax[1].bar(x + i*width, counts_t_10_percent_change_since_198d[col].values,
              width,
              label=f"target={col}",
              color=colors[i])

ax[0].set_ylabel('Count')
ax[0].set_title('Does the economic cycle effect the target '
                't_1_percent_change_since_4d ?')
ax[0].set_xticks(x + width/2, label)
ax[0].legend(loc='upper right', ncols=2)
ax[0].set_ylim(0, counts_t_1_percent_change_since_4d.values.max()*1.1)
ax[1].set_ylabel('Count')
ax[1].set_title('Does the economic cycle effect the target '
                't_10_percent_change_since_198d ?')
ax[1].set_xticks(x + width/2, label)
ax[1].set_ylim(0, counts_t_1_percent_change_since_4d.values.max()*1.1)

for s in ax[0].spines.values():
    s.set_visible(False)

for s in ax[1].spines.values():
    s.set_visible(False)
plt.show()


# filing_count_reportingOwner.name
fig, ax = plt.subplots(3, 2, figsize=(12, 12))
fig.suptitle('Is there a relationship between the target variable and the '
             'trading behavior approximated by the frequency of trades of an '
             'insider?',
             fontsize=14)
sns.boxplot(data=all_df_of_close_data,
            x='t_1_percent_change_since_4d',
            y='filing_count_reportingOwner.name',
            ax=ax[0, 0])
sns.boxplot(data=all_df_of_close_data,
            x='t_10_percent_change_since_198d',
            y='filing_count_reportingOwner.name',
            ax=ax[0, 1])


# remove upper outliers in filing_count_reportingOwner.name
# for better look at the boxplot
df = all_df_of_close_data.copy()
q75, q25 = np.quantile(df[f'{'filing_count_reportingOwner.name'}'],
                       [0.75, 0.25])
iqr = q75 - q25
df = df[df[f'{'filing_count_reportingOwner.name'}'] <= q75 + 1.5 * iqr]

sns.boxplot(data=df,
            x='t_1_percent_change_since_4d',
            y='filing_count_reportingOwner.name',
            ax=ax[1, 0])
sns.boxplot(data=df,
            x='t_10_percent_change_since_198d',
            y='filing_count_reportingOwner.name',
            ax=ax[1, 1])

# remove upper outliers once more in filing_count_reportingOwner.name
# for better look at the boxplot
df1 = df.copy()
q75, q25 = np.quantile(df1[f'{'filing_count_reportingOwner.name'}'],
                       [0.75, 0.25])
iqr = q75 - q25
df1 = df1[df1[f'{'filing_count_reportingOwner.name'}'] <= q75 + 1.5 * iqr]

sns.boxplot(data=df1,
            x='t_1_percent_change_since_4d',
            y='filing_count_reportingOwner.name',
            ax=ax[2, 0])
sns.boxplot(data=df1,
            x='t_10_percent_change_since_198d',
            y='filing_count_reportingOwner.name',
            ax=ax[2, 1])

axes = [(0, 0), (1, 0), (1, 1), (0, 1), (2, 0), (2, 1)]
for axe in axes:
    for s in ax[axe].spines.values():
        s.set_visible(False)
    ax[axe].set_title(None)
    ax[axe].set_ylabel('Amount of Fillings per Insider')

plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=2.0)
plt.show()


# Do High Frequency trader have different trades than low frequency traders
# regarding the target variables?
counts_t_1_percent_change_since_4d = (
    all_df_of_close_data
    .groupby(['high_frequency_trader', 't_1_percent_change_since_4d'])
    .size()
    .unstack(fill_value=0))

counts_t_10_percent_change_since_198d = (
    all_df_of_close_data
    .groupby(['high_frequency_trader', 't_10_percent_change_since_198d'])
    .size()
    .unstack(fill_value=0))

months = np.arange(2)
x = np.arange(2)
width = 0.4

colors = ["#E1DEDE", 'tab:blue']

fig, ax = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Do High Frequency trader have different trades than low '
             'frequency traders regarding the target variables?',
             fontsize=14)
for i, col in enumerate(counts_t_1_percent_change_since_4d.columns):
    ax[0].bar(x + i*width,
              counts_t_1_percent_change_since_4d[col].values,
              width,
              label=f"target={col}",
              color=colors[i])
    ax[1].bar(x + i*width,
              counts_t_10_percent_change_since_198d[col].values,
              width,
              label=f"target={col}",
              color=colors[i])

ax[0].set_ylabel('Count')
ax[0].set_title('t_1_percent_change_since_4d')
ax[0].set_xticks(x + width/2, months)
ax[0].legend(loc='upper right', ncols=2)
ax[0].set_ylim(0, counts_t_1_percent_change_since_4d.values.max()*1.1)
ax[1].set_ylabel('Count')
ax[1].set_title('t_10_percent_change_since_198d')
ax[1].set_xticks(x + width/2, months)
ax[1].set_ylim(0, counts_t_1_percent_change_since_4d.values.max()*1.1)

for s in ax[0].spines.values():
    s.set_visible(False)

for s in ax[1].spines.values():
    s.set_visible(False)

plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=2.0)
plt.show()

# histogram of cluster buys
fig, ax = plt.subplots(2, 1, figsize=(12, 6))
fig.suptitle("Histogram of cluster buys", fontsize=14)
ax[0].hist(all_df_of_close_data['trades_14d'], bins=100)
ax[0].set_title('Raw data')
ax[0].set_xlabel('Amount of fillings on one Ticker in past 14 days')
ax[0].set_ylabel('Count')

# remove upper outliers in trades_14d for better look
df = all_df_of_close_data.copy()
q75, q25 = np.quantile(df['trades_14d'], [0.75, 0.25])
iqr = q75 - q25
df = df[df['trades_14d'] <= q75 + 1.5 * iqr]

ax[1].hist(df['trades_14d'], bins=100)
ax[1].set_title('Without outliers')
ax[1].set_xlabel('Amount of fillings on one Ticker in past 14 days')
ax[1].set_ylabel('Count')

median = all_df_of_close_data['trades_14d'].median()
axes = [0, 1]
for axe in axes:
    for s in ax[axe].spines.values():
        s.set_visible(False)
    ax[axe].axvline(median, linestyle="--", lw=1, c="red")
    ax[axe].text(median,
                 ax[axe].get_ylim()[1]*+1.12,
                 f"median {median}",
                 va="top",
                 ha="center",
                 color="red")

plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=2.0)
plt.show()

# Do cluster buys hit often target variables then no cluster buys?
counts_t_1_percent_change_since_4d = (
    all_df_of_close_data
    .groupby(['cluster_buy', 't_1_percent_change_since_4d'])
    .size()
    .unstack(fill_value=0))

counts_t_10_percent_change_since_198d = (
    all_df_of_close_data
    .groupby(['cluster_buy', 't_10_percent_change_since_198d'])
    .size()
    .unstack(fill_value=0))


x = np.array([0, 1])
labels = ['No cluster buy', 'cluster buy']
width = 0.4

colors = ["#E1DEDE", 'tab:blue']

fig, ax = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle("Do cluster buys hit often target variables then other buys?",
             fontsize=14)
for i, col in enumerate(counts_t_1_percent_change_since_4d.columns):
    ax[0].bar(x + i*width,
              counts_t_1_percent_change_since_4d[col].values,
              width,
              label=f"target={col}",
              color=colors[i])
    ax[1].bar(x + i*width,
              counts_t_10_percent_change_since_198d[col].values,
              width,
              label=f"target={col}",
              color=colors[i])

ax[0].set_ylabel('Count')
ax[0].set_title('t_1_percent_change_since_4d')
ax[0].set_xticks(x + width/2)
ax[0].set_xticklabels(labels, rotation=0)
ax[0].legend(loc='upper right', ncols=2)
ax[0].set_ylim(0, counts_t_1_percent_change_since_4d.values.max()*1.1)
ax[1].set_ylabel('Count')
ax[1].set_title('t_10_percent_change_since_198d')
ax[1].set_xticks(x + width/2)
ax[1].set_ylim(0, counts_t_1_percent_change_since_4d.values.max()*1.1)
ax[1].set_xticklabels(labels, rotation=0)

for s in ax[0].spines.values():
    s.set_visible(False)

for s in ax[1].spines.values():
    s.set_visible(False)

plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=2.0)
plt.show()


# Do high prices hit often target variables then trades with low prices?
counts_t_1_percent_change_since_4d = (
    all_df_of_close_data
    .groupby(['high_price', 't_1_percent_change_since_4d'])
    .size()
    .unstack(fill_value=0))

counts_t_10_percent_change_since_198d = (
    all_df_of_close_data
    .groupby(['high_price', 't_10_percent_change_since_198d'])
    .size()
    .unstack(fill_value=0))

x = np.array([0, 1])
labels = ['low price', 'high price']
width = 0.4

colors = ["#E1DEDE", 'tab:blue']

fig, ax = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Do high prices hit often target variables then trades with low '
             'prices?',
             fontsize=14)
for i, col in enumerate(counts_t_1_percent_change_since_4d.columns):
    ax[0].bar(x + i*width,
              counts_t_1_percent_change_since_4d[col].values,
              width,
              label=f"target={col}",
              color=colors[i])
    ax[1].bar(x + i*width,
              counts_t_10_percent_change_since_198d[col].values,
              width,
              label=f"target={col}",
              color=colors[i])

ax[0].set_ylabel('Count')
ax[0].set_title('t_1_percent_change_since_4d')
ax[0].set_xticks(x + width/2)
ax[0].set_xticklabels(labels, rotation=0)
ax[0].legend(loc='upper right', ncols=2)
ax[0].set_ylim(0, counts_t_1_percent_change_since_4d.values.max()*1.1)
ax[1].set_ylabel('Count')
ax[1].set_title('t_10_percent_change_since_198d')
ax[1].set_xticks(x + width/2)
ax[1].set_ylim(0, counts_t_1_percent_change_since_4d.values.max()*1.1)
ax[1].set_xticklabels(labels, rotation=0)

for s in ax[0].spines.values():
    s.set_visible(False)

for s in ax[1].spines.values():
    s.set_visible(False)

plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=2.0)
plt.show()

# Histogram of change in holdings
fig, ax = plt.subplots(2, 1, figsize=(12, 6))
fig.suptitle("Histogram of Change in Holdings", fontsize=14)
ax[0].hist(all_df_of_close_data['holding_change_percent'], bins=100)
ax[0].set_title('Raw data')
ax[0].set_xlabel('Change of holding of one insider due to this filling in '
                 'percent')
ax[0].set_ylabel('Count')

# remove upper outliers in trades_14d for better look
df = all_df_of_close_data.copy()
q75, q25 = np.quantile(df['holding_change_percent'], [0.75, 0.25])
iqr = q75 - q25
df = df[df['holding_change_percent'] <= q75 + 1.5 * iqr]

ax[1].hist(df['holding_change_percent'], bins=100)
ax[1].set_title('Without outliers')
ax[1].set_xlabel('Change of holding of one insider due to this filling in '
                 'percent')
ax[1].set_ylabel('Count')

median = all_df_of_close_data['holding_change_percent'].median()
axes = [0, 1]
for axe in axes:
    for s in ax[axe].spines.values():
        s.set_visible(False)
    ax[axe].axvline(median, linestyle="--", lw=1, c="red")
    ax[axe].text(median,
                 ax[axe].get_ylim()[1]*+1.12,
                 f"median {median}",
                 va="top",
                 ha="center",
                 color="red")

plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=2.0)
plt.show()

# Do high change in holdings hit often target variables then trades with low
# change in holdings?
counts_t_1_percent_change_since_4d = (
    all_df_of_close_data
    .groupby(['high_change_in_holdings', 't_1_percent_change_since_4d'])
    .size()
    .unstack(fill_value=0))

counts_t_10_percent_change_since_198d = (
    all_df_of_close_data
    .groupby(['high_change_in_holdings', 't_10_percent_change_since_198d'])
    .size()
    .unstack(fill_value=0))

x = np.array([0, 1])
labels = ['low change in holdings', 'high change in holdings']
width = 0.4

colors = ["#E1DEDE", 'tab:blue']

fig, ax = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Do high change in holdings hit often target variables then '
             'trades with low change in holdings?', fontsize=14)
for i, col in enumerate(counts_t_1_percent_change_since_4d.columns):
    ax[0].bar(x + i*width,
              counts_t_1_percent_change_since_4d[col].values,
              width,
              label=f"target={col}",
              color=colors[i])
    ax[1].bar(x + i*width,
              counts_t_10_percent_change_since_198d[col].values,
              width,
              label=f"target={col}",
              color=colors[i])

ax[0].set_ylabel('Count')
ax[0].set_title('t_1_percent_change_since_4d')
ax[0].set_xticks(x + width/2)
ax[0].set_xticklabels(labels, rotation=0)
ax[0].legend(loc='upper right', ncols=2)
ax[0].set_ylim(0, counts_t_1_percent_change_since_4d.values.max()*1.1)
ax[1].set_ylabel('Count')
ax[1].set_title('t_10_percent_change_since_198d')
ax[1].set_xticks(x + width/2)
ax[1].set_ylim(0, counts_t_1_percent_change_since_4d.values.max()*1.1)
ax[1].set_xticklabels(labels, rotation=0)

for s in ax[0].spines.values():
    s.set_visible(False)

for s in ax[1].spines.values():
    s.set_visible(False)

plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=2.0)
plt.show()


def scatterplot(df, variable=None, target_compare="target"):
    """
    Creates scatter plots for every combination of the provided variables and
    saves them together as a grid.

    Parameters:
        df (pd.DataFrame): The current dataset.
        variable (list): A list of two variables for a single scatter plot; if
        None, plot all pairwise combinations.
        target_compare (str): Target variable used for coloring (hue).
    """
    if variable is not None:
        x, y = variable
        sns.scatterplot(data=df,
                        x=x,
                        y=y,
                        hue=target_compare if target_compare in df.columns
                        else None)
        plt.tight_layout()
        plt.savefig(f'exports/scatterplot/scatter_{x}_und_{y}.png', dpi=300)
        plt.close()
        return

    # plot for all variables
    variable = df.select_dtypes(include='number').columns.tolist()
    combination = list(itertools.combinations(variable, 2))

    for v in variable:
        relevant_combination = [(x_var, y_var) for x_var, y_var in combination
                                if v in (x_var, y_var)]

        fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
        axes = axes.flatten()

        for idx, (x_var, y_var) in enumerate(relevant_combination):
            if idx >= len(axes):
                break
            sns.scatterplot(
                data=df,
                x=x_var,
                y=y_var,
                hue=target_compare if target_compare in df.columns else None,
                ax=axes[idx]
            )

        for ax in axes[len(relevant_combination):]:
            fig.delaxes(ax)

        plt.tight_layout()
        plt.savefig('exports/scatterplot/scatter.png', dpi=300)
        plt.close()

# scatterplot(relevant_data_without_outliers,
# variable=None,
# target_compare="target")
