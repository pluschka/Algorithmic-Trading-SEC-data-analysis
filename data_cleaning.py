import pandas as pd
import numpy as np
import os
#import sys
#import re
import math
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import boxcox, skew, zscore
from sklearn.preprocessing import PowerTransformer
from sklearn.utils import resample
from pandas.api.types import is_numeric_dtype, is_object_dtype


def stats(df):
    """
    Ergänzt describe() um isna(), skew(), var(). Eine Übersicht des Ergebnisses wird im Ordner Data/Exporte abgespeichert.
    skew() in Excel =SCHIEFE(A2:A11)
    var() ist äquivalent zu VARIANZA() in Excel für Binäre Variable und VAR.P() für die restlichen numerischen Variablen in Excel.
    
    Input:
        df (pd.DataFrame): unsere aktuellen Daten

    Output:
        pd.DataFrame: Übersicht mit deskreptiven Statistiken und Missings zu jeder Variable
        csv: Übersicht mit deskreptiven Statistiken und Missings zu jeder Variable im Ordner Data/Exporte abgespeichert
    """
    # Beschreibung des DataFrames
    description = df.describe(include="all").T

    # Skewness nur für numerische Spalten berechnen
    skew = pd.DataFrame(df.select_dtypes(include=[np.number]).skew(), columns=["skew"])
    
    # Skewness nur für numerische Spalten berechnen
    var = pd.DataFrame(df.select_dtypes(include=[np.number]).var(), columns=["var"])
    
    # Datentypen der Spalten anzeigen
    dtypes = pd.DataFrame(df.dtypes, columns=['dtype'])
    
    # Anzahl der NAs berechnen und als DataFrame speichern
    na_counts = pd.DataFrame(df.isna().sum(), columns=['na_count'])
    
    # Statistiken verjoinen
    descriptives = dtypes.join(na_counts).join(description).join(skew).join(var)

    # Anzeigeformat für floats ändern, da Varianzen in wissenschaftliche Schreibweise ausgegeben werden
    descriptives['var'] = descriptives['var'].apply(lambda x: f"{x:.6f}" if pd.notnull(x) else x)

    
    # Alle Zeilen und Spalten anzeigen lassen
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(descriptives)

def clean_dtypes(df):
    """
    Überarbeitet die Datentypen
    - Zuerst alle 0/1, die kein bool sind zu bool
    - Strings via One-Hot-Encoding zu bool
    - alle restlichen numerischen Variablen einheitlich zu float umwandeln

    Input:
        df (pd.DataFrame): unsere aktuellen Daten
        max_unique_cats (int): Maximale Anzahl an Kategorien, um String-Spalten zu One-Hot-Encodieren

    Output:
        pd.DataFrame: Bereinigter DataFrame mit numerischen und boolschen Spalten
    """

    # Binärwerte erkennen und in bool umwandeln
    for col in df.columns:
        if df[col].nunique(dropna=False) == 2 and df[col].dropna().isin([0, 1]).all(): # all() function returns True if all items in an iterable are true, otherwise it returns False.
            df[col] = df[col].astype(bool)

    # Strings verarbeiten
    string_columns = df.select_dtypes(include='object').columns.tolist()
    
    # wenn string Variable dann One-Hot-Encoding (zu jeder Ausprägung ein bool generieren)
    if string_columns: #  the get_dummies() function converts categorical variables into dummy/indicator variables (known as one-hot encoding)
        df = pd.get_dummies(df, columns=string_columns, drop_first=True)
    
    # Alles numerisch machen, was geht
    for col in df.columns:
        if not is_numeric_dtype(df[col]) and not df[col].dtype == bool:
            try:
                # df[col] = df[col].astype(str).str.replace(",", ".").str.replace(" ", "") # sonst gehen die Nachkommastellen verloren
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass

    return df


def Differenz(df):
    """
    Bildet paarweise Differenzen zu allen T1 und T2 Werten. Die neuen Differenzspalten werden an den bestehenden Datensatz rangejoint.
    Binäre Variable werden in integer umgewandelt für die Differenzbildung.
    T1 - T2 = diff

    Bei der Selektion aus der IDA müssen alle T1 Variablen einheitlich "(Leerzeichen) T1" benannt werden genauso für T2.  
    T1 = 1 Monat vor Kündigung  
    T2 = 6 Monate vor Kündigung
        
    Input:
        df (pd.DataFrame): unsere aktuellen Daten mit konstanten Variablen, T1 und T2 Variablen
    
    Output:
        Neues df (pd.DataFrame): mit konstanten Variablen, T1, T2 und den neuen diff Variablen
    """
    # Prüfen, ob bereits 'diff'-Spalten existieren
    if any('diff' in col for col in df.columns):
        return df

    
    # Alle Spaltennamen extrahieren
    columns = df.columns
    
    # Spalten, die 'T1' und 'T2' enthalten
    t1_columns = [col for col in columns if 'T1' in col]
    t2_columns = [col for col in columns if 'T2' in col]
    
    # Leere Liste für neue Differenz Spalten
    diff_columns = {}
    
    # Berechne die Differenz für die entsprechenden Paare
    for t2_col in t2_columns:
        # Finde das zugehörige T1
        t1_col = t2_col.replace('T2', 'T1')
        
        if t1_col in t1_columns:
            # Berechne die Differenz und speichere sie mit einem neuen Spaltennamen
            diff_col_name = t2_col.replace('T2', 'diff')
            diff_columns[diff_col_name] = df[t1_col].astype(int) - df[t2_col].astype(int) # True/False Werte kann man nicht subtrahieren deswegen .astype(int)
    
    df = pd.concat([df, pd.DataFrame(diff_columns)], axis=1)

    return df

def split_data(df):
    """
    Teielt den Datensatz auf in T1_data, T2_data und diff_data. Die dfs bestehen aus den konstanten Kennzahlen, die sich nicht über die 
    Zeit ändern.
        
    Input:
        df (pd.DataFrame): unsere aktuellen Daten mit konstanten Kennzahlen, T1, T2 und diff Variablen
    
    Output:
        T1_df (pd.DataFrame): mit konstanten Kennzahlen + T1 Variablen
        T2_df (pd.DataFrame): mit konstanten Kennzahlen + T2 Variablen
        diff_df (pd.DataFrame): mit konstanten Kennzahlen + diff Variablen
    """
    # Alle Spaltennamen extrahieren
    columns = df.columns
    
    # Spalten, die 'T1' und 'T2' enthalten
    t1_columns = [col for col in columns if ' T1' in col]
    t2_columns = [col for col in columns if ' T2' in col]
    diff_columns = [col for col in columns if ' diff' in col]
    konst_columns = [col for col in columns if ' T1' not in col and ' T2' not in col and ' diff' not in col]

    T1_df = df[konst_columns + t1_columns]
    T2_df = df[konst_columns + t2_columns]
    diff_df = df[konst_columns + diff_columns]

    return T1_df, T2_df, diff_df


def missing_values_summary(df):
    """
    Gibt eine Übersicht über alle Spalten mit fehlenden Werten in einem DataFrame.

    Input:
        df (pd.DataFrame): unsere aktuellen Daten

    Output:
        Neues df (pd.DataFrame): Variablennamen mit dazugehörige NA Anzahl und Prozentanteil
    """
    total = df.isnull().sum()
    percent = (total / len(df)) * 100
    missing = pd.DataFrame({
        'Anzahl_NaNs': total,
        'Prozent_NaNs': percent.round(2)
    })
    missing = missing[missing['Anzahl_NaNs'] > 0]
    missing = missing.sort_values(by='Anzahl_NaNs', ascending=False)
    
    return missing


def Missings(df, default_missing_strategy = "delete", execpt_replace_0=[], execpt_replace_mean=[],execpt_delete=[]):
    """
    Behandelt Missings je nach gewählter Strategie
    
    default_missing_strategy (string) wird pauschal für alle Variablen angewand, essei denn sie sind in den anderen Listen gelistet
        replace_0: Ersetzt NAs mit 0
        replace_mean: Ersetzt NAs mit Durchschnittswert der Variable
        delete: Löscht jede Zeile in der ein NA vorkommt

    execpt_replace_0 (list): Diese Variablen werden mit 0 ersetzt, statt mit der default strategie behandelt
    execpt_replace_mean (list): Diese Variablen werden mit dem Durchschnitt ersetzt, statt mit der default strategie behandelt
    execpt_delete (list): Diese Variablen werden gelöscht, statt mit der default strategie behandelt

    Bsp.:  Missings(diff_df, default_missing_strategy = "delete", 
                             execpt_replace_0=["Variable_1"], 
                             execpt_replace_mean=["Variable_1"],
                             execpt_delete=[])

    Input:
        df (pd.DataFrame): unsere aktuellen Daten

    Output:
        Neues df (pd.DataFrame): Ohne NAs
        
    """
    for col in df.columns:
        if df[col].isna().any():
            if col in execpt_replace_0:
                df.loc[:, col] = df[col].fillna(0)
            elif col in execpt_replace_mean:
                df.loc[:, col] = df[col].fillna(df[col].mean())
            elif col in execpt_delete:
                # ~ bitwise negation operator Tilde is bitwise negation operator. 
                # It flips 1's with 0's and vice versa. This operation takes place in binary level.
                df = df[~df[col].isna()]  # Zeilen mit NA in dieser Spalte löschen
            else:
                # Default Strategie
                if default_missing_strategy == "replace_0":
                    df.loc[:, col] = df[col].fillna(0)
                elif default_missing_strategy == "replace_mean":
                    df.loc[:, col] = df[col].fillna(df[col].mean())
                elif default_missing_strategy == "delete":
                    df = df[~df[col].isna()]  # Zeilen mit NA in dieser Spalte löschen
    return df

def plot_heatmap(df):
    corr_matrix = df.corr(numeric_only=False).abs()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    plt.show() 


def Korrelation(df, correlation_threshold = 0.8, df_name="df", target_col="target", never_drop="target"):
    """
    Berechnet die Korrelationsmatrix und speichert sie in dem Notebook Ordner ab.
    Alle Variablen mit einer Korrelation über dem Schwellwert (default = 0.8) werden aus dem Datensatz entfernt unter folgender 
    Entschiedungsregel:
    Wenn eine Variable mit einer anderen über dem threshold korreliert wird anhand der Korrelation mit der Target
    Variable entschieden welche entfernt wird. Die Idee ist, dass man die Variable behält mit einer höheren Korrelation zu
    target, weil diese mehr Erklärungsgehalt für das Modell hat. Die Variable mit der geringeren Korrelation zu
    target wird gelöscht.
    Ausgegeben wird eine Liste der entfernetn Variablen mit den Korrelationen und das bereinihgte df.

    Wenn man das in Excel Überprüfen will =KORREL(--bool_variable_1; --bool_variable_2), das -- bei Bool Werten. Sind die Missings vorher 
    nicht behandelt, dann kommen leicht andere Korrelationen raus. Übergibt man Excel die um Missings bereinigten Daten, dann kommen
    dieselben Korrelationen raus wie in python raus. 
        
    Input:
        df (pd.DataFrame): unsere aktuellen Daten
        correlation_threshold (float): Grenzwert für das Entfernen der Variablen
        df_name (string): Name vom verwendeten df, T1_df, T2_df oder diff_df. Das wird für die Log Dokumentation benötigt.
        target_col (string): als Hilfe für die Berechnung der Korrelation mit der Target Variable
        never_drop (list): Wenn man vermeiden will, dass eine bestimmt Variable gelöst wird, speichert man sie hier als Liste ab
        
    Output:
        Neues df (pd.DataFrame): ohne Variablen mit einer Korrelation über dem Wert festgelegt mit correlation_threshold
        Korrelations Matrix (Excel Datei): Abgespeichert im Notebook Ordner
        Print Statements: Liste aller entfernten Variablen mit Korrelationsinfos
    """
    # Korrelationsmatrix berechnen
    corr_matrix = df.corr().abs()
    
    # Diagonale auf 0 setzen, um sich selbstkorrelationen zu ignorieren
    np.fill_diagonal(corr_matrix.values, 0)

    corr_with_target = df.corr(method='pearson')[target_col].abs()
    to_drop = set()
    removed_details = []

    # Für alle Werte unterhalb der Diagonale
    for i in range(len(corr_matrix.columns)): # range = 0, 1, 2 ... bis Anzahl Spalten
        for j in range(i): # range(i=0) -> 0, dann range(i=1) -> 0, 1 ; dann range(i=2) -> 0, 1,2 :  range(i=5) -> 0, 1, 2, 3, 4, 5 usw
            var1 = corr_matrix.columns[i]
            var2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]

            # Wenn in der Korrelationsmatrix eine sehr hohe Korrelation gefunden wird...
            if corr_value > correlation_threshold:
                # (Skip den nächsten Teil wenn Variablen shcon in to_drop sind oder undere Target)
                if var1 in to_drop or var2 in to_drop or var1 == target_col or var2 == target_col:
                    continue

                # ...berechnen wir die Korrelation beider Variablen mit TF_Kündiger.
                var1_target_corr = corr_with_target.get(var1, 0)
                var2_target_corr = corr_with_target.get(var2, 0)

                # Um zu entscheiden welche Variable rausfliegt vergleiche die Korrelation mit TF_Kündiger
                if var1 and var2 in never_drop: # Wenn beide Varibalen in never_drop sind continue
                    continue

                # Wenn var2 in never_drop ist oder die Korrelation von var2 mit Kündiger höher, dann keep var2 und remove var1
                elif var2 in never_drop or var1_target_corr < var2_target_corr: 
                    remove_var = var1
                    keep_var = var2
                    remove_corr = var1_target_corr
                    keep_corr = var2_target_corr
                
                # Wenn var1 in never_drop ist oder die Korrelation von var1 mit Kündiger höher, dann keep var1 und remove var2    
                elif var1 in never_drop or var2_target_corr <= var1_target_corr:
                    remove_var = var2
                    keep_var = var1
                    remove_corr = var2_target_corr
                    keep_corr = var1_target_corr

                to_drop.add(remove_var)
                removed_details.append((remove_var, keep_var, corr_value, remove_corr, keep_corr))

    corr_matrix.to_excel("Korrelationsmatrix.xlsx", index=True)

    if not to_drop:
        print(f"In {df_name} wurden keine Spalten wegen zu hoher Korrelation entfernt.")
    else:
        print("Entfernte Spalten wegen hoher Korrelation:")
        for remove_var, keep_var, corr_value, remove_corr, keep_corr in removed_details:
            print(f"In {df_name} wurde '{remove_var}' entfernt wegen der Korrelation mit '{keep_var}' (r = {corr_value:.3f})")
            print(f"Korrelation von '{remove_var}' mit '{target_col}': {remove_corr:.3f}")
            print(f"Korrelation von '{keep_var}' mit '{target_col}': {keep_corr:.3f}")
            print("-----------------------------------")

    print("-------------------------------------------------------------------------------------------")
    df = df.drop(columns=to_drop)
    df.shape
    return df, removed_details, corr_matrix



def Varianz(df, variance_threshold = 0.1, target_corr_threshold = 0.2, df_name="df", target_col="TF_Kündiger", never_drop=["TF_Kündiger"]):
    """
    Berechnet die Varianz jeder Variable.
    var() ist äquivalent zu VARIANZA() in Excel für Binäre Variable und VAR.P() für die restlichen numerischen Variablen in Excel.
        
    Input:
        df (pd.DataFrame): unsere aktuellen Daten
        variance_threshold (float): Grenzwert für Varianz für das Entfernen der Variablen 
        target_corr_threshold (float): Grenzwert für Korrelation mit Targetvariable für das Entfernen der Variablen 
        df_name (string): Name vom verwendeten df, T1_df, T2_df oder diff_df. Das wird für die Log Dokumentation benötigt.
        target_col (string): als Hilfe für die Berechnung der Korrelation mit der Target Variable
        never_drop (list): Wenn man vermeiden will, dass eine bestimmt Variable gelöst wird, speichert man sie hier als Liste ab
    
    Output:
        Neues df (pd.DataFrame): ohne Variablen mit niedriger Varianz unter dem Wert festgelegt mit variance_threshold
        Print Statement: Liste aller entfernten Variablen
    """
    variances = df.var(skipna=True, ddof=1) # ddof=1 für Stichprobenvarianz; 0 wäre Populationsvarianz

    low_variance_columns = list(variances[(variances < variance_threshold)].index)

    # Korrelation mit Zielvariable berechnen
    corr_with_target = df.corr(method='pearson')[target_col].abs()

    # Varianz = 0 kann schon mal in die to_drop liste
    to_drop = []
    
    # Nur Variablen entfernen, die auch eine niedrige Korrelation zur Zielvariable haben
    for var in low_variance_columns:
        if var in never_drop:
            continue  # nicht löschen, weil in never_drop-Liste
            
        var_target_corr = corr_with_target.get(var, 0)

        if var_target_corr > target_corr_threshold:
            print(f"In {df_name} hat '{var}' niedrige Varianz mit {variances[var]:.6f}, aber hohe Korrelation mit {target_col} {var_target_corr:.3f} wird behalten.")

        else:
            to_drop.append(var)
       
    # Variablen mit niedriger Varianz aus dem df entfernen
    df = df.drop(columns=to_drop)
    
    if not to_drop:
        print(f"In {df_name} wurden keine Spalten wegen zu geringer Varianz entfernt.")
    else:
        print(f"Entfernte Spalten in {df_name} mit niedriger Varianz und geringer Korrelation mit {target_col}:")
        for col in to_drop:
            print(f"{col}: Varianz = {variances[col]:.6f}, Korrelation mit {target_col} = {corr_with_target[col]:.3f}")
    print("-------------------------------------------------------------------------------------------")
        
    return df


def Ausreißer(df, default_outlier_strategy = "delete", except_replace_0=[], except_replace_mean=[], except_delete=[], ignore=[]):

    for column in df.select_dtypes(include=['number']).columns:
        if column in ignore: 
            continue
        # Q1, Q3 und IQR berechnen
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Definiere die Grenzen für Ausreißer
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if column in except_replace_0:
            df.loc[:, column] = df[column].apply(
                lambda x: 0 if x < lower_bound or x > upper_bound else x)
        elif column in except_replace_mean:
            mean_val = df[column].mean()
            df.loc[:, column] = df[column].apply(
                lambda x: mean_val if x < lower_bound or x > upper_bound else x)
        elif column in except_delete:
            df = df[~((df[column] < lower_bound) | (df[column] > upper_bound))]
        else:
            # Default Strategie
            if default_outlier_strategy == "replace_0":
                df[column] = df[column].apply(
                    lambda x: 0 if x < lower_bound or x > upper_bound else x)
            elif default_outlier_strategy == "replace_mean":
                mean_val = df[column].mean()
                df[column] = df[column].apply(
                    lambda x: mean_val if x < lower_bound or x > upper_bound else x)
            elif default_outlier_strategy == "delete":
                df = df[~((df[column] < lower_bound) | (df[column] > upper_bound))]
    return df


def count_outlires(df, column):
    cols = df.columns

    # 1) Alles numerisch erzwingen (Nicht-Numerisches -> NaN)
    df = df[cols].apply(pd.to_numeric, errors='coerce')
    # Q1, Q3 und IQR berechnen
    Q1 = df[cols].quantile(0.25)
    Q3 = df[cols].quantile(0.75)
    IQR = Q3 - Q1

    # Definiere die Grenzen für Ausreißer
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_count = df[column].apply(lambda x: x < lower_bound or x > upper_bound).sum()
    return outlier_count

#count_outliers(df_important, column)

def Ausreißerstrategievgl(df, kündiger_var="target"):
    import math
    variablen = df.select_dtypes(include='number').columns.tolist()

    for var in variablen:
        # Spalte in float umwandeln, damit replace_0/mean keine Datentyp-Warnung mehr gibt
        df[var] = df[var].astype(float)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        anzahl_ausreißer = count_outlires(df, column=var)
        fig.suptitle(f"Ausreißerstrategien für: {var}  –  Ausreißeranzahl: {anzahl_ausreißer}", fontsize=16)

        # DataFrames für jede Strategie vorbereiten
        df0 = Ausreißer(df.copy(), default_outlier_strategy="replace_0",
                        except_replace_0=[], except_replace_mean=[],
                        except_delete=[], ignore=[var])

        df1 = Ausreißer(df.copy(), default_outlier_strategy="replace_0",
                        except_replace_0=[], except_replace_mean=[],
                        except_delete=[], ignore=[])

        df2 = Ausreißer(df.copy(), default_outlier_strategy="replace_0",
                        except_replace_0=[], except_replace_mean=[var],
                        except_delete=[], ignore=[])

        df3 = Ausreißer(df.copy(), default_outlier_strategy="replace_0",
                        except_replace_0=[], except_replace_mean=[],
                        except_delete=[var], ignore=[])

        daten = [df0, df1, df2, df3]
        titel = ["Original mit Ausreißer", "mit 0 ersetzt", "mit Durchschnitt ersetzt", "Ausreißer gelöscht"]

        for i in range(4):
            ax = axes[i]
            if kündiger_var is None:
                sns.histplot(data=daten[i], x=var, bins=30, kde=False, ax=ax)
            else:
                sns.histplot(data=daten[i], x=var, hue=kündiger_var, bins=30, kde=False, ax=ax)
            ax.set_title(titel[i])
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        output_path = os.path.join("..", "Reports", "Ausreißeranalyse", f"Ausreißeranalyse_{var}.png")
        plt.savefig(output_path, dpi=300)
        
        plt.show()


def Verteilung_Übersicht(df):
    """
    Gibt für jede numerische Spalte die ursprüngliche Schiefe der Verteilung (skewness) der original Variablen
    sowie die Skewness nach der Transformationen zurück.
    
    Folgende Transformationen werden betrachtet:
    - log(x)
    - log1p(x)
    - sqrt(x)
    - boxcox(x)
    - -log1p(x)
    - log(max - x + 1)
    - power_transform (Yeo-Johnson)

    log(x) und boxcox(x)nicht auf negative Werte anwendbar
    
    Wenn eine Transformation nicht möglich ist, wird '-' angezeigt. Zum Beispiel bei den Binären Variablen.

    Input:
        df (pd.DataFrame): unsere aktuellen Daten
    
    Output:
        pd.DataFrame: Übersicht mit Skewness-Werten pro Transformation
    """
    results = []

    # Binäre Variablen überspringen
    for col in df.columns:
        if df[col].dtype == bool:
            results.append({
                "Spalte": col,
                "original": "-",
                "log(x)": "-",
                "log1p": "-",
                "sqrt": "-",
                "boxcox": "-",
                "power_transform": "-"

            })
            continue

        series = pd.to_numeric(df[col], errors='coerce')
        row = {"Spalte": col}

        # Original
        try:
            # Nur sinnvolle Werte verwenden
            #valid_series = series.dropna()
        
            # Bedingung: genügend eindeutige Werte und keine konstante Verteilung
            if series.nunique() <= 1:
                row["original"] = "nicht berechenbar"
            elif series.var() < 1e-8:
                row["original"] = "fast konstant"
            else:
                row["original"] = round(skew(series), 3)
        except:
            row["original"] = "-"

        # log(x) nur für x > 0
        try:
            if (series <= 0).any():
                row["log(x)"] = "nicht anwendbar, weil 0 oder negative Werte"
            else:
                transformed = np.log(series)
                if len(transformed) <= 1:
                    row["log(x)"] = "nicht berechenbar"
                elif np.var(transformed) < 1e-8:
                    row["log(x)"] = "fast konstant"
                else:
                    row["log(x)"] = round(skew(transformed), 3)
        except:
            row["log(x)"] = "-"
                    
        # log1p(x) nur für x > -1
        try:        
            if (series <= -1).any():
                row["log1p"] = "nicht anwendbar, weil Werte kleiner -1"
            else:
                transformed = np.log1p(series)
                row["log1p"] = round(skew(transformed), 3)
        except:
            row["log1p"] = "-"

        
        # sqrt(x) nur für x ≥ 0
        try:
            if (series < 0).any():
                row["sqrt"] = "nicht anwendbar, weil Werte kleiner 0"
            else:
                transformed = np.sqrt(series)
                row["sqrt"] = round(skew(transformed), 3)
        except:
            row["sqrt"] = "-"
        
        # boxcox(x) nur für x > 0
        try:
            if (series <= 0).any():
                row["boxcox"] = "nicht anwendbar, weil Werte 0 oder negativ"
            else:
                transformed, _ = boxcox(series) # zweiten Rückgabewert (lambda) irgnorieren mit ,_
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


def Transformieren(df, variable=None, transformation=None):
    """
    Wendet gewünschte Transformation auf eine übergebene Variable an.
    
    Input:
        df (pd.DataFrame): Ursprüngliches DataFrame
        variable (str): Name der Spalte im DataFrame, z.B. "Variable_1"
        transformation (str): Transformation zur Verbesserung der Schiefe (Skewness).
            Auswahl: "log", "log1p", "sqrt", "boxcox", "power_transform"
    
    Output:
        pd.DataFrame: DataFrame mit ggf. transformierter und umbenannter Spalte
        Print Statement: Skewness der Variable vor und nach der Transformation
    """
    
    series = df[variable]
    
    # Original Skewness
    if series.nunique() <= 1:
        og_skewness = "nicht berechenbar"
    elif series.var() < 1e-8:
        og_skewness = "fast konstant"
    else:
        og_skewness = round(skew(series), 3)
    
    transformed = None
    
    # log(x)
    if transformation == "log":
        if (series <= 0).any():
            new_skewness = "nicht anwendbar, weil 0 oder negative Werte"
        else:
            transformed = np.log(series)
    
    # log1p(x)
    elif transformation == "log1p":
        if (series <= -1).any():
            new_skewness = "nicht anwendbar, weil Werte kleiner -1"
        else:
            transformed = np.log1p(series)
    
    # sqrt(x)
    elif transformation == "sqrt":
        if (series < 0).any():
            new_skewness = "nicht anwendbar, weil Werte kleiner 0"
        else:
            transformed = np.sqrt(series)
    
    # boxcox(x)
    elif transformation == "boxcox":
        if (series <= 0).any():
            new_skewness = "nicht anwendbar, weil Werte 0 oder negativ"
        else:
            transformed, _ = boxcox(series) # zweiten Rückgabewert (lambda) irgnorieren mit ,_
    
    # power_transform (Yeo-Johnson)
    elif transformation == "power_transform":
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        transformed = pt.fit_transform(series.values.reshape(-1, 1)).flatten()
    
    # Berechne neue Skewness, wenn Transformation stattgefunden hat
    if transformed is not None:
        if len(transformed) <= 1:
            new_skewness = "nicht berechenbar"
        elif np.var(transformed) < 1e-8:
            new_skewness = "fast konstant"
        else:
            new_skewness = round(skew(transformed), 3)
        
        # Speichere transformierte Spalte im DataFrame
        new_colname = f"{variable} ({transformation})"
        df[new_colname] = transformed
    
    print(f"Skewness von '{variable}' ist vorher {og_skewness} und {transformation} ist {new_skewness}")
    
    return df

def balance(df, kündiger_var="target"):
    # Vorausgesetzter Code für die Erstellung von df_diff
    df_majority = df[df[kündiger_var] == False]
    df_minority = df[df[kündiger_var] == True]
    
    # Zufällige Stichprobe der Mehrheitklasse in der gleichen Größe wie die Minderheitklasse
    df_majority_downsampled = resample(df_majority, 
                                       replace=False,    # Ohne Zurücklegen
                                       n_samples=len(df_minority),  # Gleiche Anzahl wie die Minderheit
                                       random_state=42)  # Reproduzierbare Ergebnisse
    
    # Zusammenführen der beiden Gruppen aka Union
    df = pd.concat([df_majority_downsampled, df_minority])
    
    diagram(df, diagram="histplot", variable="target", kündiger_var=None, save_as=None)
    return df


def Skalierung(df, scale_stratagy = "alle", variable=None):
    """
    Wendet den Z-Score auf die Liste der übergebenen Variablen an. Per Default werden alle Variablen skaliert, 
    die keine bool Variablen sind.
    Empfohlen bei Logistic Regression aber nicht bei Decision Trees, Random Forest, XGBoost. Betsenfalls vorher
    Transformieren und dann Z-Score anwenden.
    
    Input:
        df (pd.DataFrame): Ursprüngliches DataFrame
        scale_stratagy (string): "alle" oder "ausgewählte"
        variable (list): Name der Spalte im DataFrame, z.B. ["Variable_1"]
        
    Output:
        pd.DataFrame: DataFrame mit ggf. skalierten Variablen
        Print Statement: Information über skalierte Variablen
        pd.DataFrame durch stats(df): Übersicht mit deskreptiven Statistiken und Missings zu jeder Variable
        csv durch stats(df): Übersicht mit deskreptiven Statistiken und Missings zu jeder Variable im Ordner Data/Exporte abgespeichert
    """
    
    if scale_stratagy=="alle":
        columns = df.select_dtypes(include=['number']).columns
    elif scale_stratagy=="ausgewählte": columns = variable
    
    for var in columns:
        df[var] = zscore(df[var])
        print(f"{var} wurde mit dem zscore angepasst")

    stats(df)


    return df


def scatterplot(df, variable=None, kündiger_var="target"):
    """
    Macht Scatterplots für jede Kombination der übergebenen Variablen und speichert alle gemeinsam als Grid.

    Input:
        df (pd.DataFrame): unsere aktuellen Daten
        variable (list): Liste mit Variablen 2 für Einzel-Scatterplot oder wenn None dann werden alle Variablen geplottet
        kündiger_var (str): Zielvariable für die Farbe (Hue)
    """

    if variable is not None:
        x, y = variable
        sns.scatterplot(data=df, x=x, y=y, hue=kündiger_var if kündiger_var in df.columns else None)
        plt.tight_layout()
        output_path = os.path.join("..", "Reports", "Scatterplot", f"Scatter_{x}_und_{y}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        return

    # Plots für alle Variablen
    variablen = df.select_dtypes(include='number').columns.tolist()
    kombis = list(itertools.combinations(variablen, 2))

    # Für jede Variable v ein eigenes Grid
    for v in variablen:
        relevante_kombis = [(x_var, y_var) for x_var, y_var in kombis if v in (x_var, y_var)]

        fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
        axes = axes.flatten()

        for idx, (x_var, y_var) in enumerate(relevante_kombis):
            if idx >= len(axes):
                break
            sns.scatterplot(
                data=df,
                x=x_var,
                y=y_var,
                hue=kündiger_var if kündiger_var in df.columns else None,
                ax=axes[idx]                        # <-- hier!
            )

        # Restliche leere Achsen entfernen
        for ax in axes[len(relevante_kombis):]:
            fig.delaxes(ax)

        plt.tight_layout()
        output_path = os.path.join("..", "Reports", "Scatterplot", f"Scatter_{v}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()



def diagram(df, diagram="boxplot", variable=None, kündiger_var="target", save_as="Plot.png", titel=None):
    """
    Macht Boxplots oder Histogramme für jede Variable im df oder die übergebene Variable in der Option variable, die als Liste übergeben wird.
    Wenn kündiger_var gleich unsere Kündigervariable ist, wird der Plot unterteielt in Kündiger/Nichtkündiger, sonst kündiger_var=None wird
    die gesamte Variable ohne Unterteilung betrachtet

    Input:
        df (pd.DataFrame): unsere aktuellen Daten
        diagram (string): Betsimmt den Diagramtyp, Auswahl zwischen "boxplot", "histplot"
        variable (list): Liste mit Variablen die als Boxplot ausgegeben werden soll, 
        z.B. ["Variable_1"] , per default None damit alle ausgegeben werden
        kündiger_var (string): per default unsere Kündigervariable, aber wenn None, dann wird einfachnur normales Boxplot ausgegeben
        save_as (string): Name der abgespeicherten PNG Datei, wenn None nichts gespeichert
    
    Output:
        Histogram/Boxplot Plots zu den übergebenen Variablen ggf. für Kündiger/Nichtkündiger
    """

    # Plots für eine Variable
    if variable is not None:
        plt.figure(figsize=(6, 4))
        if diagram == "boxplot":
            if kündiger_var is None:
                ax = sns.boxplot(y=df[variable])
            else:
                ax = sns.boxplot(data=df, x=kündiger_var, y=variable)
        elif diagram == "histplot":
            if kündiger_var is None:
                ax = sns.histplot(data=df, x=variable, bins=30, kde=False)
            else:
                ax = sns.histplot(data=df, x=variable, hue=kündiger_var, bins=30, kde=False)
        title_text = f"{variable}" if titel is None else f"{variable} – {titel}"
        ax.set_title(title_text)

        plt.tight_layout()
        if save_as:
            plt.savefig(save_as, dpi=300)
        plt.show()
        return

    
    # Plots für alle Variablen
    else:
        
        variablen = df.select_dtypes(include='number').columns.tolist()
        if kündiger_var in variablen:
            variablen.remove(kündiger_var)

        anzahl = len(variablen)
        spalten = 3  # Anzahl Spalten im Grid
        zeilen = math.ceil(anzahl / spalten)
    
        fig, axes = plt.subplots(zeilen, spalten, figsize=(spalten * 5, zeilen * 4))
        axes = axes.flatten()  # macht 2D-Achsenarray zu 1D-Liste
    
        for i, var in enumerate(variablen):
            ax = axes[i]
            if diagram == "boxplot":
                if kündiger_var is None:
                    sns.boxplot(y=df[var], ax=ax)
                else:
                    sns.boxplot(data=df, x=kündiger_var, y=var, ax=ax)
    
            elif diagram == "histplot":
                if kündiger_var is None:
                    sns.histplot(data=df, x=var, bins=30, kde=False, ax=ax)
                else:
                    sns.histplot(data=df, x=var, hue=kündiger_var, bins=30, kde=False, ax=ax)
                    
            # Titel pro Subplot
            if titel:
                ax.set_title(f"{var} – {titel}")
    
        # Leere Achsen ausblenden
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        if save_as!=None:
            plt.savefig(save_as, dpi=300)

  
    plt.show()


def null_analyse(df):
    # Anteil 0 pro Variable
    zero_all = (df==0).sum() / len(df) *100
    
    # Anzahl Kündiger
    kündiger_anzahl = (df["target"] == 1).sum()
    
    # Anzahl von Kündigern mit 0 als ausprägung geteielt durch GesamtzahlKündiger
    zero_kündiger = (df[df["target"] == 1] == 0).sum() / kündiger_anzahl * 100
    
    
    
    zerodf = pd.DataFrame({
        'Variable': df.columns,
        'Prozent von Nullwerten je Variable': zero_all.round(2).values,
        'Wie viel Prozent der Kündiger sind in der Variable 0?': zero_kündiger.round(2).values
        
    })
    
    zerodf = zerodf.sort_values(
        by='Prozent von Nullwerten je Variable',
        ascending=False
    ).reset_index(drop=True)

    zerodf["Kritische Variable"] = (
    (zerodf['Prozent von Nullwerten je Variable'] > 99)
    &
    (zerodf['Wie viel Prozent der Kündiger sind in der Variable 0?'] > 99)
    )

    # Vierfeldertafel erstellen für Kritische Variablen
    kritische_vars = zerodf.loc[zerodf["Kritische Variable"], 'Variable'].tolist()
    
    # Loop über die kritischen Variablen
    for var in kritische_vars:
        print(f"\nVierfeldertafel für {var}:\n")
        table = pd.crosstab(
            df[var] == 0,        
            df['target'],     
            rownames=[f"{var}"],
            colnames=['target'],
            margins=True # fügt Gesamtsummen hinzu
        )
        print(table)
    return zerodf

def drop_t1(df):
    t1_columns = [col for col in df.columns if ' T1' in col]
    
    df = df.drop(columns=t1_columns)
    return df
