import pandas as pd
import numpy as np
import os
import math
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox, skew, zscore
from sklearn.preprocessing import PowerTransformer
from sklearn.utils import resample
from pandas.api.types import is_numeric_dtype, is_object_dtype
from IPython.display import display


def stats(df):
    """
    Adds isna(), skew(), and var() on top of describe(). An overview of the results is saved in the export folder.
    skew() in Excel = SKEW(A2:A11)
    var() is equivalent to VAR.S in Excel for binary variables and VAR.P for the remaining numeric variables.

    Input:
    df (pd.DataFrame): our current data

    Output:
    pd.DataFrame: overview with descriptive statistics and missing values for each variable
    csv: the same overview saved to export
    """
    # describe dataframes
    description = df.describe(include="all").T

    # compute skewness for numeric columns only
    skew = pd.DataFrame(df.select_dtypes(include=[np.number]).skew(), columns=["skew"])
    
    # compute variance for numeric columns only
    var = pd.DataFrame(df.select_dtypes(include=[np.number]).var(), columns=["var"])
    
    # show column data types
    dtypes = pd.DataFrame(df.dtypes, columns=['dtype'])
    
    # compute NA counts and store as a DataFrame
    na_counts = pd.DataFrame(df.isna().sum(), columns=['na_count'])
    
    # merge statistics
    descriptive = dtypes.join(na_counts).join(description).join(skew).join(var)

    # change float display format since variances are shown in scientific notation
    descriptive['var'] = descriptive['var'].apply(lambda x: f"{x:.6f}" if pd.notnull(x) else x)

    # display all rows and columns
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(descriptive)

def clean_dtypes(df):
    """
    Standardizes data types
    - First, convert all 0/1 columns that are not boolean to boolean
    - Convert string columns to boolean via one-hot encoding
    - Convert all remaining numeric variables uniformly to float

    Input:
        df (pd.DataFrame): our current data
        max_unique_cats (int): Maximum number of categories for one-hot encoding string columns

    Output:
        pd.DataFrame: cleaned DataFrame with numeric and boolean columns
    """

    # detect binary values and convert to boolean
    for col in df.columns:
        if df[col].nunique(dropna=False) == 2 and df[col].dropna().isin([0, 1]).all(): # all() function returns True if all items in an iterable are true, otherwise it returns False.
            df[col] = df[col].astype(bool)

    # process string columns
    string_columns = df.select_dtypes(include='object').columns.tolist()
    
    # if there are string columns, apply one-hot encoding (create a boolean for each category)
    if string_columns:  # get_dummies() converts categorical variables into dummy/indicator variables (one-hot encoding)
        df = pd.get_dummies(df, columns=string_columns, drop_first=True)
    
    # convert everything to numeric where possible
    for col in df.columns:
        if not is_numeric_dtype(df[col]) and not df[col].dtype == bool:
            try:
                # df[col] = df[col].astype(str).str.replace(",", ".").str.replace(" ", "")  # otherwise decimal places get lost
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass

    return df


def missing_values_summary(df):
    """
    Provides an overview of all columns with missing values in a DataFrame.

    Input:
        df (pd.DataFrame): our current data

    Output:
        New df (pd.DataFrame): variable names with their NA count and percentage
        (rounded to 2 decimals), sorted by NA count in descending order.
    """
    total = df.isnull().sum()
    percent = (total / len(df)) * 100
    missing = pd.DataFrame({
        'na_count': total,
        'percent of na': percent.round(2)
    })
    missing = missing[missing['na_count'] > 0]
    missing = missing.sort_values(by='na_count', ascending=False)
    
    return missing


def missing_strategy(df, default_missing_strategy = "delete", except_replace_0=[], except_replace_mean=[],except_delete=[]):
    """
    Handles missing values according to the chosen strategy.

    default_missing_strategy (str) is applied to all variables unless they are listed in the exception lists:
        replace_0: replaces NAs with 0
        replace_mean: replaces NAs with the column mean
        delete: drops every row that contains an NA

    except_replace_0 (list): variables to replace with 0 instead of using the default strategy
    except_replace_mean (list): variables to replace with the mean instead of using the default strategy
    except_delete (list): variables for which rows with NAs should be dropped instead of using the default strategy

    Example:
        missing_strategy(
            diff_df,
            default_missing_strategy="delete",
            except_replace_0=["Variable_1"],
            except_replace_mean=["Variable_2"],
            except_delete=[],
        )

    Input:
        df (pd.DataFrame): our current data

    Output:
        New df (pd.DataFrame): dataset without NAs
    """
    for col in df.columns:
        if df[col].isna().any():
            if col in except_replace_0:
                df.loc[:, col] = df[col].fillna(0)
            elif col in except_replace_mean:
                df.loc[:, col] = df[col].fillna(df[col].mean())
            elif col in except_delete:
                # ~ bitwise negation operator Tilde is bitwise negation operator. 
                # It flips 1's with 0's and vice versa. This operation takes place in binary level.
                df = df[~df[col].isna()]  # drop columns with na
            else:
                # default strategy
                if default_missing_strategy == "replace_0":
                    df.loc[:, col] = df[col].fillna(0)
                elif default_missing_strategy == "replace_mean":
                    df.loc[:, col] = df[col].fillna(df[col].mean())
                elif default_missing_strategy == "delete":
                    df = df[~df[col].isna()]  # drop columns with na
    return df

def plot_heatmap(df):
    """
    Plots a heatmap of the given df. Make sure to remove string variables.
    """
    corr_matrix = df.corr(numeric_only=False).abs()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    plt.show() 

#plot_heatmap(relevant_data_uncleaned)


def remove_correlation(df, correlation_threshold = 0.8, df_name="df", target_col="target", never_drop="target"):
    """
    Computes the correlation matrix and saves it in the notebook directory.
    All variables with a correlation above the threshold (default = 0.8) are removed
    according to the following decision rule:
    If two variables correlate above the threshold, the one with the lower correlation
    to the target variable is removed. The idea is to keep the variable with the
    higher correlation to the target because it provides more explanatory power
    for the model. The output includes a list of removed variables with their
    correlations and the cleaned DataFrame.

    To verify in Excel: =CORREL(--bool_variable_1, --bool_variable_2).
    Use the double unary (--) for boolean values. If missing values are not handled
    beforehand, Excel may yield slightly different correlations. If you pass the data
    with missing values already handled, Excel will produce the same correlations
    as Python.

    Input:
        df (pd.DataFrame): our current data
        correlation_threshold (float): threshold for dropping variables
        df_name (str): name of the DataFrame (e.g., T1_df, T2_df, or diff_df);
                    used for log documentation
        target_col (str): the target variable used to compare correlations
        never_drop (list): variables that must never be dropped

    Output:
        New df (pd.DataFrame): without variables whose correlation exceeds correlation_threshold
        Correlation matrix (Excel file): saved in the notebook directory
        Print statements: list of all dropped variables with correlation details
    """

    # calculate correlation matrix
    corr_matrix = df.corr().abs()
    
    # set the diagonal to 0 to ignore self-correlations
    np.fill_diagonal(corr_matrix.values, 0)

    corr_with_target = df.corr(method='pearson')[target_col].abs()
    to_drop = set()
    removed_details = []

    # for all values below the diagonal
    for i in range(len(corr_matrix.columns)): # range = 0, 1, 2 ... up to the number of columns
        for j in range(i): # range(i) yields 0..i-1; e.g., i=0 -> [], i=1 -> [0], i=2 -> [0, 1], i=5 -> [0, 1, 2, 3, 4]
            var1 = corr_matrix.columns[i]
            var2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]

            # If a very high correlation is found in the correlation matrix...
            if corr_value > correlation_threshold:
                # skip the next part if the variables are already in to_drop or are the target
                if var1 in to_drop or var2 in to_drop or var1 == target_col or var2 == target_col:
                    continue

                # ...compute each variable's correlation with the target
                var1_target_corr = corr_with_target.get(var1, 0)
                var2_target_corr = corr_with_target.get(var2, 0)

                # To decide which variable to drop, compare their correlations with the target
                if var1 and var2 in never_drop: # If both variables are in never_drop, continue
                    continue

                # If var2 is in never_drop or var2's correlation with the target is higher, keep var2 and drop var1
                elif var2 in never_drop or var1_target_corr < var2_target_corr: 
                    remove_var = var1
                    keep_var = var2
                    remove_corr = var1_target_corr
                    keep_corr = var2_target_corr
                
                # If var1 is in never_drop or var1's correlation with the target is higher, keep var1 and drop var2
                elif var1 in never_drop or var2_target_corr <= var1_target_corr:
                    remove_var = var2
                    keep_var = var1
                    remove_corr = var2_target_corr
                    keep_corr = var1_target_corr

                to_drop.add(remove_var)
                removed_details.append((remove_var, keep_var, corr_value, remove_corr, keep_corr))

    #corr_matrix.to_excel("corrmatrix.xlsx", index=True)

    if not to_drop:
        print(f"In {df_name} were no columns removed due to high correlation.")
    else:
        print("This columns were removed due to high correlation:")
        for remove_var, keep_var, corr_value, remove_corr, keep_corr in removed_details:
            print(f"In {df_name} was '{remove_var}' removed because of the correlation with '{keep_var}' (r = {corr_value:.3f})")
            print(f"Correlation of'{remove_var}' with '{target_col}': {remove_corr:.3f}")
            print(f"Correlation of '{keep_var}' with '{target_col}': {keep_corr:.3f}")
            print("-----------------------------------")

    print("-------------------------------------------------------------------------------------------")
    df = df.drop(columns=to_drop)
    df.shape
    return df, removed_details, corr_matrix


def variance(df, 
             variance_threshold = 0.1, 
             target_corr_threshold = 0.2, 
             df_name="df",
             target_col="target_variable",
             never_drop=["target_variable"]):
    """
    Computes the variance of each variable.
    var() is equivalent to VAR.S in Excel for binary variables and VAR.P for the remaining numeric variables.

    Input:
        df (pd.DataFrame): our current data
        variance_threshold (float): variance threshold for dropping variables
        target_corr_threshold (float): correlation-with-target threshold for dropping variables
        df_name (str): name of the DataFrame (e.g., T1_df, T2_df, or diff_df); used for log documentation
        target_col (str): the target variable used to compute correlations
        never_drop (list): variables that must never be dropped

    Output:
        New df (pd.DataFrame): without variables whose variance is below variance_threshold
        Print statements: list of all removed variables
    """
    variances = df.var(skipna=True, ddof=1)  # ddof=1 for sample variance; 0 would be population variance

    low_variance_columns = list(variances[(variances < variance_threshold)].index)

    # compute correlation with the target variable
    corr_with_target = df.corr(method='pearson')[target_col].abs()


    to_drop = []
    
    # only drop variables that also have low correlation with the target
    for var in low_variance_columns:
        if var in never_drop:
            continue  # don't drop because it's in the never_drop list
            
        var_target_corr = corr_with_target.get(var, 0)

        if var_target_corr > target_corr_threshold:
            print(f"In {df_name} has '{var}' a low variance of {variances[var]:.6f}, but a high correlation with {target_col} {var_target_corr:.3f} and is not removed from the df")

        else:
            to_drop.append(var)
       
    # drop low-variance variables from the DataFrame
    df = df.drop(columns=to_drop)
    
    if not to_drop:
        print(f"In {df_name} has no low variance variables and no columns were removed.")
    else:
        print(f"Removed columns in {df_name} with low variance and low correlation with {target_col}:")
        for col in to_drop:
            print(f"{col}: Variance = {variances[col]:.6f}, correlation with {target_col} = {corr_with_target[col]:.3f}")
    print("-------------------------------------------------------------------------------------------")
        
    return df

# variance(relevant_data_without_outlier, 
#         variance_threshold = 0.1,
#         target_corr_threshold = 0.2,
#         df_name="relevant_data_without_outlier",
#         target_col="t_1_percent_change_since_4d",
#         never_drop=["t_1_percent_change_since_4d"])


def outlier_strategy(df,
                     default_outlier_strategy = "delete",
                     except_replace_0=[],
                     except_replace_mean=[],
                     except_delete=[],
                     ignore=[]):
    """
    Removes outliers column-wise using the IQR rule and applies a chosen handling strategy.

    Outlier rule:
        For each numeric column, compute Q1 (25th pct), Q3 (75th pct), IQR = Q3 - Q1.
        Values < Q1 - 1.5*IQR or > Q3 + 1.5*IQR are treated as outliers.

    Input:
        df (pd.DataFrame): input data. All columns are coerced to numeric (non-numeric → NaN) before processing.
        default_outlier_strategy (str): fallback strategy for columns not listed in the exception lists.
            Allowed: "delete" | "replace_0" | "replace_mean"
            - "delete": drop rows where the column has an outlier
            - "replace_0": replace outliers in the column with 0
            - "replace_mean": replace outliers in the column with the column mean
        except_replace_0 (list[str]): columns for which outliers are replaced with 0 (overrides default).
        except_replace_mean (list[str]): columns for which outliers are replaced with the column mean (overrides default).
        except_delete (list[str]): columns for which rows with outliers are dropped (overrides default).
        ignore (list[str]): columns to skip entirely (no outlier handling applied).

    Output:
        pd.DataFrame: DataFrame after outlier handling.

    Notes:
        - Only numeric columns are processed; coercion uses pd.to_numeric(..., errors="coerce").
        - If IQR is 0, no values will be marked as outliers for that column (bounds collapse to Q1=Q3).
    """


    cols = df.columns
    df = df[cols].apply(pd.to_numeric, errors='coerce')

    for cols in df.select_dtypes(include=['number']).columns:
        if cols in ignore: 
            continue
        # calculate Q1, Q3 und IQR 
        Q1 = df[cols].quantile(0.25)
        Q3 = df[cols].quantile(0.75)
        IQR = Q3 - Q1

        # define threshold for outlier
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if cols in except_replace_0:
            df.loc[:, cols] = df[cols].apply(
                lambda x: 0 if x < lower_bound or x > upper_bound else x)
        elif cols in except_replace_mean:
            mean_val = df[cols].mean()
            df.loc[:, cols] = df[cols].apply(
                lambda x: mean_val if x < lower_bound or x > upper_bound else x)
        elif cols in except_delete:
            df = df[~((df[cols] < lower_bound) | (df[cols] > upper_bound))]
        else:
            # strategies
            if default_outlier_strategy == "replace_0":
                df[cols] = df[cols].apply(
                    lambda x: 0 if x < lower_bound or x > upper_bound else x)
            elif default_outlier_strategy == "replace_mean":
                mean_val = df[cols].mean()
                df[cols] = df[cols].apply(
                    lambda x: mean_val if x < lower_bound or x > upper_bound else x)
            elif default_outlier_strategy == "delete":
                df = df[~((df[cols] < lower_bound) | (df[cols] > upper_bound))]
    return df

# outlier_strategy(relevant_data_uncleaned, 
# default_outlier_strategy = "delete",
# except_replace_0=[],
# except_replace_mean=[],
# except_delete=[],
# ignore=[])


def count_outliers(df, columns=None, iqr_k=1.5):
    """
    Makes a table summary of amount of outliers in df.
    """
    cols = df.columns if columns is None else list(columns)

    # fore numeric
    num = df[cols].apply(pd.to_numeric, errors='coerce')

    # calculate quantile/IQR
    Q1 = num.quantile(0.25)
    Q3 = num.quantile(0.75)
    IQR = (Q3 - Q1).astype(float)

    # select outliers
    lb = (Q1 - float(iqr_k) * IQR).astype(float)
    ub = (Q3 + float(iqr_k) * IQR).astype(float)
    mask = num.lt(lb, axis=1) | num.gt(ub, axis=1)

    # summarize outliers table
    return mask.sum().sort_values(ascending=False)

#count_outliers(df_important, column)

def outlier_strategy_comparison(df, name_target="target_variable"):
    variables = df.select_dtypes(include='number').columns.tolist()

    for var in variables:
        # change datatype to float to avoid warnings
        df[var] = df[var].astype(float)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        count_outliers = count_outliers(df, column=var)
        fig.subtitle(f"Outlier strategy for {var} (Outliers: {count_outliers}", fontsize=16)

        # prepare df for each strategy
        df0 = outlier_strategy(df.copy(), default_outlier_strategy="replace_0",
                               except_replace_0=[], except_replace_mean=[],
                               except_delete=[], ignore=[var])

        df1 = outlier_strategy(df.copy(), default_outlier_strategy="replace_0",
                               except_replace_0=[], except_replace_mean=[],
                               except_delete=[], ignore=[])

        df2 = outlier_strategy(df.copy(), default_outlier_strategy="replace_0",
                               except_replace_0=[], except_replace_mean=[var],
                               except_delete=[], ignore=[])

        df3 = outlier_strategy(df.copy(), default_outlier_strategy="replace_0",
                               except_replace_0=[], except_replace_mean=[],
                               except_delete=[var], ignore=[])

        dataframe = [df0, df1, df2, df3]
        title = ["Original data", "replaced with 0", "replaced with mean", "outlier deleted"]

        for i in range(4):
            ax = axes[i]
            if name_target is None:
                sns.histplot(data=dataframe[i], x=var, bins=30, kde=False, ax=ax)
            else:
                sns.histplot(data=dataframe[i], x=var, hue=name_target, bins=30, kde=False, ax=ax)
            ax.set_title(title[i])
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def distribution_overview(df):
    """
    Returns, for each numeric column, the original skewness of the variable
    and the skewness after the following transformations:

    Transformations considered:
    - log(x)
    - log1p(x)
    - sqrt(x)
    - boxcox(x)
    - -log1p(x)
    - log(max - x + 1)
    - power_transform (Yeo-Johnson)

    Note: log(x) and boxcox(x) are not applicable to negative values.

    If a transformation is not possible (e.g., for binary variables), '-' is shown.

    Input:
        df (pd.DataFrame): our current DataFrame

    Output:
        pd.DataFrame: summary with skewness values per transformation
    """

    results = []

    # skip binary variables
    for col in df.columns:
        if df[col].dtype == bool:
            results.append({
                "row": col,
                "original": "-",
                "log(x)": "-",
                "log1p": "-",
                "sqrt": "-",
                "boxcox": "-",
                "power_transform": "-"

            })
            continue

        series = pd.to_numeric(df[col], errors='coerce')
        row = {"Row": col}

        # original
        try:
            # use only useful variables
            #valid_series = series.dropna()
        
            # condition: enough distinct values
            if series.nunique() <= 1:
                row["original"] = "not calculable"
            elif series.var() < 1e-8:
                row["original"] = "almost constant"
            else:
                row["original"] = round(skew(series), 3)
        except:
            row["original"] = "-"

        # log(x) only für x > 0
        try:
            if (series <= 0).any():
                row["log(x)"] = "not applicable because 0 or negative value"
            else:
                transformed = np.log(series)
                if len(transformed) <= 1:
                    row["log(x)"] = "not calculable"
                elif np.var(transformed) < 1e-8:
                    row["log(x)"] = "almost constant"
                else:
                    row["log(x)"] = round(skew(transformed), 3)
        except:
            row["log(x)"] = "-"
                    
        # log1p(x) only für x > -1
        try:        
            if (series <= -1).any():
                row["log1p"] = "not calculable because of values smaller then -1"
            else:
                transformed = np.log1p(series)
                row["log1p"] = round(skew(transformed), 3)
        except:
            row["log1p"] = "-"

        
        # sqrt(x) only für x ≥ 0
        try:
            if (series < 0).any():
                row["sqrt"] = "not calculable because of values smaller then 0"
            else:
                transformed = np.sqrt(series)
                row["sqrt"] = round(skew(transformed), 3)
        except:
            row["sqrt"] = "-"
        
        # boxcox(x) only für x > 0
        try:
            if (series <= 0).any():
                row["boxcox"] = "not calculable because of values smaller then 0"
            else:
                transformed, _ = boxcox(series)
                row["boxcox"] = round(skew(transformed), 3)
        except:
            row["boxcox"] = "-"

        # power_transform (Yeo-Johnson)
        try:
            pt = PowerTransformer(method="yeo-johnson", standardize=False)
            transformed = pt.fit_transform(series.values.reshape(-1, 1)).flatten()
            row["power_transform"] = round(skew(transformed), 3)
        except:
            row["power_transform"] = "-"


        results.append(row)

    return pd.DataFrame(results)


def transform_skewness(df, variable=None, transformation=None):
    """
    Applies a chosen transformation to a given column.

    Input:
        df (pd.DataFrame): original DataFrame
        variable (str): name of the column in the DataFrame (e.g., "Variable_1")
        transformation (str): transformation to improve skewness.
            Options: "log", "log1p", "sqrt", "boxcox", "power_transform"

    Notes:
        - log(x) and boxcox(x) are not applicable to negative values
        (log requires x > 0; Box–Cox requires x > 0).
        - power_transform refers to scikit-learn’s Yeo–Johnson (works with zeros and negatives).

    Output:
        pd.DataFrame: DataFrame with the column transformed (and renamed if applicable)
        print: skewness of the variable before and after the transformation
    """
    series = df[variable]
    
    # original skewness
    if series.nunique() <= 1:
        og_skewness = "not calculable"
    elif series.var() < 1e-8:
        og_skewness = "almost constant"
    else:
        og_skewness = round(skew(series), 3)
    
    transformed = None
    
    # log(x)
    if transformation == "log":
        if (series <= 0).any():
            new_skewness = "not calculable because of values smaller then 0"
        else:
            transformed = np.log(series)
    
    # log1p(x)
    elif transformation == "log1p":
        if (series <= -1).any():
            new_skewness = "not calculable because of values smaller then -1"
        else:
            transformed = np.log1p(series)
    
    # sqrt(x)
    elif transformation == "sqrt":
        if (series < 0).any():
            new_skewness = "not calculable because of values smaller then 0"
        else:
            transformed = np.sqrt(series)
    
    # boxcox(x)
    elif transformation == "boxcox":
        if (series <= 0).any():
            new_skewness = "not calculable because of values smaller then 0"
        else:
            transformed, _ = boxcox(series)
    
    # power_transform (Yeo-Johnson)
    elif transformation == "power_transform":
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        transformed = pt.fit_transform(series.values.reshape(-1, 1)).flatten()
    
    # calculate new skewness after transformation
    if transformed is not None:
        if len(transformed) <= 1:
            new_skewness = "not calculable"
        elif np.var(transformed) < 1e-8:
            new_skewness = "almost constant"
        else:
            new_skewness = round(skew(transformed), 3)
        
        # save transformed variables in data frame
        new_colname = f"{variable} ({transformation})"
        df[new_colname] = transformed
    
    print(f"Skewness of '{variable}' was before {og_skewness} and {transformation} is {new_skewness}")
    
    return df

def balance(df, name_target="target"):

    df_majority = df[df[name_target] == False]
    df_minority = df[df[name_target] == True]
    
    # random sample of majority class same size as majority class
    df_majority_downsampled = resample(df_majority, 
                                       replace=False, # without returning
                                       n_samples=len(df_minority),  # same size as minority class
                                       random_state=420)  # set seed
    
    # union data
    df = pd.concat([df_majority_downsampled, df_minority])
    
    diagram(df, diagram="histplot", variable="target", name_target=None, save_as=None)
    return df


def scaling(df, scale_strategy = "all", columns_to_scale=None):
    """
    Applies Z-score standardization to the provided variables. By default, all non-boolean
    columns are scaled.

    Recommended for: logistic regression and other linear models; typically not needed
    for tree-based models (Decision Trees, Random Forest, XGBoost). Prefer running
    transform_skewness first, then applying the Z-score.

    Input:
        df (pd.DataFrame): original DataFrame
        scale_strategy (str): "all" (scale all non-boolean columns) or "only_chosen_variables" (scale only selected columns)
        variable (list[str]): list of column names to scale when using "only_chosen_variables", e.g., ["Variable_1"]

    Output:
        pd.DataFrame: DataFrame with standardized variables (where applicable)
        print: information about which variables were scaled
        pd.DataFrame via stats(df): overview with descriptive statistics and missing values for each variable
    """
    
    if scale_strategy=="all":
        columns = df.select_dtypes(include=['number']).columns
    elif scale_strategy=="only_chosen_variables":
    
        for var in columns_to_scale:
            df[var] = zscore(df[var])
            print(f"{var} were adjusted with z score")

    stats(df)
    return df


def scatterplot(df, variable=None, name_target="target"):
    """
    Creates scatter plots for every combination of the provided variables and saves them together as a grid.

    Input:
        df (pd.DataFrame): our current data
        variable (list or None): list with two variable names for a single scatter plot; if None, plot all pairwise combinations
        name_target (str): target variable used for color mapping (hue)
    """

    # plots for chosen variables
    if variable is not None:
        x, y = variable
        sns.scatterplot(data=df, x=x, y=y, hue=name_target if name_target in df.columns else None)
        plt.tight_layout()
        plt.savefig(f"scatter_{x}_and_{y}.png", dpi=300)
        plt.close()
        return

    # plots for all variables
    variable = df.select_dtypes(include='number').columns.tolist()
    combination = list(itertools.combinations(variable, 2))

    for v in variable:
        relevant_combination = [(x_var, y_var) for x_var, y_var in combination if v in (x_var, y_var)]

        fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
        axes = axes.flatten()

        for idx, (x_var, y_var) in enumerate(relevant_combination):
            if idx >= len(axes):
                break
            sns.scatterplot(
                data=df,
                x=x_var,
                y=y_var,
                hue=name_target if name_target in df.columns else None,
                ax=axes[idx]
            )

        # remove irrelevant axe
        for ax in axes[len(relevant_combination):]:
            fig.delaxes(ax)

        plt.tight_layout()
        output_path = os.path.join("..", "Reports", "Scatterplot", f"Scatter_{v}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()


def diagram(df,
            diagram="boxplot",
            variable=None,
            name_target="target",
            title=None):
    """
    Creates boxplot or histograms for each variable in the DataFrame, or for a specific
    list of variables passed via `variable`. If `name_target` is provided, plots are
    stratified by that target (e.g., Churner vs. Non-churner); if `name_target=None`,
    the overall distribution is shown.

    Input:
        df (pd.DataFrame): the current dataset
        diagram (str): chart type; one of {"boxplot", "histplot"}
        variable (list[str] or None): variables to plot; e.g., ["Variable_1"].
            If None (default), plot all eligible variables.
        name_target (str or None): target column used to split the plots by class (hue).
            If None, no stratification is applied.
        save_as (str or None): filename for saving the PNG output; if None, do not save

    Output:
        Plots: histogram(s)/boxplot(s) for the requested variable(s), optionally split by `name_target`
    """


    # plot for one  variable
    if variable is not None:
        plt.figure(figsize=(6, 4))
        if diagram == "boxplot":
            if name_target is None:
                ax = sns.boxplot(y=df[variable])
            else:
                ax = sns.boxplot(data=df, x=name_target, y=variable)
        elif diagram == "histplot":
            if name_target is None:
                ax = sns.histplot(data=df, x=variable, bins=30, kde=False)
            else:
                ax = sns.histplot(data=df, x=variable, hue=name_target, bins=30, kde=False)
        title_text = f"{variable}" if title is None else f"{variable} – {title}"
        ax.set_title(title_text)

        plt.tight_layout()
        plt.show()
        return

    
    # plots for all variables
    else:
        
        variable = df.select_dtypes(include='number').columns.tolist()
        if name_target in variable:
            variable.remove(name_target)

        count = len(variable)
        columns = 3  # count rows in grid
        rows = math.ceil(count / columns)
    
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 5, rows * 4))
        axes = axes.flatten()  # makes 2D array to a 1D list
    
        for i, var in enumerate(variable):
            ax = axes[i]
            if diagram == "boxplot":
                if name_target is None:
                    sns.boxplot(y=df[var], ax=ax)
                else:
                    sns.boxplot(data=df, x=name_target, y=var, ax=ax)
    
            elif diagram == "histplot":
                if name_target is None:
                    sns.histplot(data=df, x=var, bins=30, kde=False, ax=ax)
                else:
                    sns.histplot(data=df, x=var, hue=name_target, bins=30, kde=False, ax=ax)
                    
            # title for each subplot
            if title:
                ax.set_title(f"{var} - {title}")
    
        # hide empty axe
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
    plt.show()


def null_analyse(df):
    """
    Analyzes zero values per column and within the positive target class.

    What it does:
        - Computes the overall percentage of zeros for each column.
        - Computes, among rows where target == 1, the percentage of zeros per column.
        - Returns a summary DataFrame sorted by overall zero percentage.
        - Flags "critical" variables (default rule: >99% zeros overall AND >99% zeros when target == 1).
        - Prints a 2x2 contingency table (crosstab) for each critical variable.

    Input:
        df (pd.DataFrame): dataset containing a binary column 'target' (0/1).

    Output:
        pd.DataFrame: summary with columns
            - 'variable'
            - 'percent of 0 in each variable'
            - 'How many percent of the target = 1 have 0 in this variable?'
            - 'critical variable' (boolean)
        print: crosstab for each critical variable
    Notes:
        - Assumes 'target' exists and is binary (0/1).
        - If there are no rows with target == 1, the target-specific percentages are undefined.
    """

    # share of 0 in each variable
    zero_all = (df==0).sum() / len(df) *100
    
    # count target = 1
    target_count = (df["target"] == 1).sum()
    
    # percent target = 1 with 0
    zero_target = (df[df["target"] == 1] == 0).sum() / target_count * 100

    zerodf = pd.DataFrame({
        'variable': df.columns,
        'percent of 0 in each variable': zero_all.round(2).values,
        'How many percent of the target = 1 have 0 in this variable?': zero_target.round(2).values
        
    })
    
    zerodf = zerodf.sort_values(
        by='percent of 0 in each variable',
        ascending=False
    ).reset_index(drop=True)

    zerodf["critical variable"] = (
    (zerodf['percent of 0 in each variable'] > 99)
    &
    (zerodf['How many percent of the target = 1 have 0 in this variable'] > 99)
    )

    # crosstab for critical variables
    critical_vars = zerodf.loc[zerodf["critical Variable"], 'Variable'].tolist()
    
    # Loop for critical variables
    for var in critical_vars:
        print(f"\nCrosstab for {var}:\n")
        table = pd.crosstab(
            df[var] == 0,        
            df['target'],     
            rownames=[f"{var}"],
            colnames=['target'],
            margins=True # adds sum
        )
        print(table)
    return zerodf

def drop_t1(df):
    t1_columns = [col for col in df.columns if ' T1' in col]
    
    df = df.drop(columns=t1_columns)
    return df
