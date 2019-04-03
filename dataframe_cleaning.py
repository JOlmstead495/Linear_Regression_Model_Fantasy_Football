import numpy as np
import pandas as pd
import math

def split_dataframe_by_position(df_combined):
    """
    Splits the combined dataframe based on players position.
    Args:
        df_combined (pandas.DataFrame): Combined reception and rushing
        statistics
    Returns:
        df_rb (pandas.DataFrame): DataFrame that contains running back stats.
        df_wr (pandas.DataFrame): DataFrame that contains wide receiver stats.
    """
    posRB = ((df_combined['Pos_Rec'].isnull()) &
             (df_combined['Yds_Rush'] > df_combined['Yds_Rec']))
    posWR = ((df_combined['Pos_Rec'].isnull()) &
             (df_combined['Yds_Rush'] <= df_combined['Yds_Rec']))
    df_combined.loc[posRB, 'Pos_Rec'] = 'RB'
    df_combined.loc[posWR, 'Pos_Rec'] = 'WR'
    df_combined.drop(columns=['Rk_Rec', 'Age_Rec', 'G_Rec', 'GS_Rec',
                              'Fmb_Rec', 'Rk_Rush', 'Pos_Rush'], inplace=True)
    df_final = df_combined.select_dtypes(exclude=['object']).copy()
    df_final['POS'] = df_combined['Pos_Rec']
    df_final['Player'] = df_combined['Player']
    df_final.dropna(thresh=5, inplace=True)
    df_final.fillna(0, inplace=True)
    df_rb = df_final[df_final['POS'].str[:2].str.lower() == 'rb']
    df_wr = df_final[df_final['POS'].str[:2].str.lower() == 'wr']
    df_rb = df_rb[df_rb[('Yds_Rush'] > 100) | (df_rb['Yds_Rec'] > 100)]
    df_wr = df_wr[(df_wr['Yds_Rush'] > 100) | (df_wr['Yds_Rec'] > 100)]
    df_rb = df_rb[['Age_Rush', 'Yds_Rec', 'Tgt', 'Y/G_Rush', 'Player']]
    df_wr = df_wr[['Yds_Rec', 'Y/G_Rec', 'Yds_Rush', 'Player']]
    return df_rb, df_wr


def set_pos_if_null(row):
    """
    Function used to set POS (Player Position) to the POS pulled from the
    rushing statistics dataframe if the POS is currently Null. This is
    needed because the POS column was originally pulled from the reception
    statistics, and if the player had no receptions, they would have a null
    POS.
    Args:
        row (pandas.Series): Each individual row in a DataFrame that has a
        POS and Pos_Rush column
    Returns:
        row['Pos_Rush'] (string): Value in row['Pos_Rush']
        row['POS'] (string): Value in row['POS']
    """
    if str(row['POS']).lower() == 'nan':
        return row['Pos_Rush']
    else:
        return row['POS']


def convert_series_to_float(df):
    """
    Function that will convert all numbers in dataframe to a float otherwise
    it will not convert the value if it is not a number
    Args:
        df (pandas.DataFrame): Dataframe to convert to numbers.
    Returns:
        None
    """
    columns = df.columns
    for i in columns:
        try:
            df[i] = df[i].map(lambda x: float(x))
        except Exception:
            pass
        else:
            continue
    return


def log_function(data):
    """
    Log a specific data point, if the data point is 0, return 0.
    Args:
        data (float): Number to get logged
    Returns:
        Logged Data (float)
        or
        0
    """
    try:
        return math.log(data)
    except Exception:
        return(0)


def log_function_shift(data):
    """
    Shift the data up by 1 and then take the log.
    Args:
        data (float): Number to get shifted and logged
    Returns:
        Logged Data (float)
    """
    data += 1
    return math.log(data)