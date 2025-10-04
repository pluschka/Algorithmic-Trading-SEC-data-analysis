import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

all_df_of_close_data = pd.read_csv('data/all_df_of_close_data.csv')

# remove inplausible data
for num in range(364):
    all_df_of_close_data = all_df_of_close_data[all_df_of_close_data[f'{num}']>= 0]

# remove upper outlires
df = all_df_of_close_data.copy()
for num in range(364):
    q75, q25 = np.quantile(df[f'{num}'], [0.75 ,0.25])
    iqr = q75 - q25
    df = df[df[f'{num}'] <= q75 + 1.5 * iqr]

all_df_of_close_data = df


# 1) Daten vorbereiten
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

plt.savefig('exports/descriptives_random_state_69.png', dpi=500)
plt.show()
