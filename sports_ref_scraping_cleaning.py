import sys
import requests
from bs4 import BeautifulSoup
import os
import numpy as np
import pandas as pd
from selenium import webdriver
import re
import time
from dataframe_cleaning import convert_series_to_float


def sports_ref_stat_scrape(urlList):
    """
    This function will scrape Pro Football Reference seasonal rushing and
    receiving statistics.
    Args:
        urlList (list): Takes a list of a list of the rushing and receiving
        urls
    Returns:
        df_rush_total (pandas.DataFrame): Dataframe that contains rushing
        statistics for each player.
        df_rec_total (pandas.DataFrame): Dataframe that contains receiving
        statistics for each player.
    """
    sports_ref_table_regex = re.compile("sortable stats_table")
    yr = 18
    df_rush_total = pd.DataFrame()
    df_rec_total = pd.DataFrame()
    for url_year in urlList:
        count = 0
        for site in url_year:
            sports_ref_text = BeautifulSoup(requests.get(site).text, 'html')
            if count == 0:
                df_rush = pd.read_html(str(sports_ref_text.find_all(
                    class_=sports_ref_table_regex)))
            else:
                df_rec = pd.read_html(str(sports_ref_text.find_all(
                    class_=sports_ref_table_regex)))
            count += 1
        df_rush[0].columns = df_rush[0].columns.get_level_values(1)
        df_rush[0].pipe(remove_Non_Words, column='Player', year=yr)
        df_rec[0].pipe(remove_Non_Words, column='Player', year=yr)
        df_rush_total = df_rush_total.append(df_rush[0])
        df_rec_total = df_rec_total.append(df_rec[0])
        yr -= 1
    return df_rush_total, df_rec_total


def remove_Non_Words(df, column, year):
    """
    This function takes a DataFrame and will remove non alphabetic characters
    and will remove name titles (jr, ii, iii) and will add
    the year to the end of the name.
    Args:
        df (pandas.DataFrame): DataFrame to change.
        column (string): String name of the column to change.
        year (int or string): number to append to the end of the Player's name
    Returns:
        df (pandas.DataFrame): Updated DataFrame with the updated column name
        that got changed.
    """
    only_word = re.compile(r'(?:\S+\s+){0}(\w+)')
    df[column] = df[column].map(lambda x: (concat_list(re.findall(
        only_word, x)))+str(year))
    return df


def concat_list(player_name):
    """
    Helper function that concatenates and lower cases players names. It will
    not add on any player's titles (jr, ii, iii)
    Args:
        player_name (string): Player's First and Last Name
    Returns:
        return_string (string): Cleaned up player name
    """
    return_string = ''
    for i in player_name:
        if (player_name.lower() == "jr" or player_name.lower() == "ii" or
            player_name.lower() == "iii"):
            continue
        return_string = return_string+str(player_name).lower()
    return return_string


def sports_ref_cleaning(df_rush, df_rec):
    """
    This function will clean and merge Pro Football Reference seasonal rushing
    and receiving statistics.
    Args:
        df_rush (pandas.DataFrame): DataFrame that contains Rushing Statistics
        df_rec (pandas.DataFrame): DataFrame that contains Receiving Statistics
    Returns:
        df_sports_ref (pandas.DataFrame): A combined DataFrame of rushing and
        receiving statistics. The data has been cleaned.
    """
    df_rush = df_rush[df_rush['Rk'] != 'Rk']
    df_rec = df_rec[df_rec['Rk'] != 'Rk']
    perc_to_decimal(df_rec, 'Ctch%')
    df_sports_ref = df_rec.merge(df_rush, how='outer', on='Player',
                                 suffixes=['_Rec', '_Rush'])
    df_sports_ref.drop(columns=['Rk_Rec', 'Rk_Rush'], inplace=True)
    df_sports_ref = df_sports_ref.reset_index(drop=True)
    df_sports_ref.drop(columns=['Tm_Rec', 'Age_Rec', 'Pos_Rec', 'G_Rec',
                                'GS_Rec', 'Tm_Rush', ' Fmb_Rush'], axis=1,
                       inplace=True)
    df_sports_ref.drop_duplicates(subset='Player', keep='first', inplace=True)
    return df_sports_ref


def sports_ref_cleaning_2018(urlList, regex):
    """
    This function will call scrape_2018_Fantasy_Points_Sports_Ref to pull in
    PFF receiving and rushing stats. And will clean those statistics,
    returning a merged
    DataFrame.
    Args:
        urlList (list): A list of PFF urls strings.
        regex (re.Pattern): A Regex compiled Pattern
    Returns:
        (pandas.DataFrame): A combined DataFrame of rushing and
        receiving statistics. The data has been cleaned.
    """
    df_rush, df_rec = scrape_2018_Fantasy_Points_Sports_Ref(urlList, regex)
    df_rush = df_rush[df_rush['Rk'] != 'Rk']
    df_rec = df_rec[df_rec['Rk'] != 'Rk']
    df_rush = df_rush.reset_index()
    df_rush = df_rush.drop(['index'], axis=1)
    remove_Non_Words(df_rush, 'Player', 18)
    remove_Non_Words(df_rec, 'Player', 18)
    perc_to_decimal(df_rec, 'Ctch%')
    convert_series_to_float(df_rush)
    convert_series_to_float(df_rec)
    return df_rec.merge(df_rush, how='outer', on='Player', suffixes=['_Rec',
                        '_Rush'])


def scrape_2015_Fantasy_Points_Sports_Ref(url):
    """
    This function pulls in 2015 fantasy points total from Sports Ref,
    cleans the data, and then returns the data in a dataframe.
    Args:
        url (string): String that represents a url site.
    Returns:
        df_FF (pandas.DataFrame): DataFrame that contains cleaned
        data.
    """
    sports_ref_table_reg = re.compile("sortable stats_table")
    Play_Stat_Text = BeautifulSoup(requests.get(url).text, 'html')
    df_FF = pd.read_html(str(Play_Stat_Text.find_all(
        class_=sports_ref_table_reg)))
    df_FF[0].columns = df_FF[0].columns.get_level_values(1)
    df_FF = df_FF[0]
    df_FF = df_FF.loc[:, ['Player', 'PPR', 'FantPos']]
    df_FF.columns = ['PLAYER_NAME', 'PTS', 'POS']
    df_FF.drop_duplicates(subset='PLAYER_NAME', keep='first', inplace=True)
    df_FF.pipe(remove_Non_Words, column='PLAYER_NAME', year=16)
    df_FF.drop(df_FF.loc[df_FF['PLAYER_NAME'] == 'player16'].index,
               inplace=True)
    df_FF.reset_index(drop=True, inplace=True)
    return df_FF


def scrape_2018_Fantasy_Points_Sports_Ref(urlList, regex):
    """
    This function is a helper function that pulls in PFF statistics
    using BeautifulSoup.
    Args:
        urlList (list): A list of PFF urls strings.
        regex (re.Pattern): A Regex compiled Pattern
    Returns:
        df_rush (pandas.DataFrame): A DataFrame consisting of all rushing
        statistics for 2018.
        df_rec (pandas.DataFrame): A DataFrame consisting of all reception
        statistics
    """
    count = 0
    df_read = []
    for site in urlList:
        Play_Stat_Text = BeautifulSoup(requests.get(site).text, 'html')
        if count == 0:
            df_rush = pd.read_html(str(Play_Stat_Text.find_all(class_=regex)))
        else:
            df_rec = pd.read_html(str(Play_Stat_Text.find_all(class_=regex)))
        count += 1
    df_rush[0].columns = df_rush[0].columns.get_level_values(1)
    return df_rush[0], df_rec[0]
