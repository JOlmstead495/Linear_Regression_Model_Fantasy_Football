import sys
import requests
from bs4 import BeautifulSoup
import os
import numpy as np
import pandas as pd
from selenium import webdriver
import re
import time


def load_driver():
    '''
        This is a helper function that loads the Chromium Driver.
    Args:
        None
    Returns driver (selenium.webdriver): Driver used to run auto script.
    '''
    # path to the chromedriver executable
    chromedriver = "/Applications/chromedriver"
    os.environ["webdriver.chrome.driver"] = chromedriver
    driver = webdriver.Chrome(chromedriver)
    return driver


def ESPN_Next_Page(page_count, driver):
    '''
        This is a helper function that clicks ESPNs Next Button to go 
        to the next page.
    Args:
        page_count (int): Number of page the driver is on.
        driver (selenium.webdriver): Driver used to run auto script.
    Returns driver (selenium.webdriver): Driver used to run auto script.
    '''
    if page_count == 0:
        next_button = driver.find_element_by_xpath(
            "//div[@class='paginationNav']//a")
    else:
        next_button = driver.find_element_by_xpath(
            "//div[@class='paginationNav']//a[2]")
    driver.execute_script("window.scrollBy(0,200)", "")
    time.sleep(1)
    next_button.click()
    return driver


def load_ESPN_Data(url):
    '''
        Calls helper functions and returns a dataframe for a specific year,
        for WR and RB stats.
    Args:
        url (string): ESPN URL that has the table of WR and RB statistics.
    Returns df_table (list): List of DataFrames
    '''
    FF_Table_Class_Regex = re.compile("games-fullcol")
    driver = load_FF_driver()
    count = 0
    # Category 2 and 4 represents RB and WR stats.
    for catId in [2, 4]:
        tempurl = url
        tempurl += str(catId)
        driver.get(tempurl)
        driver.fullscreen_window()
        for page in np.arange(5):
            FFtext = requests.get(driver.current_url).text
            FF_Pret_Text = BeautifulSoup(FFtext, 'html')
            if count == 0:
                df_table = pd.read_html(str(FF_Pret_Text.find_all(
                    class_=FF_Table_Class_Regex)), skiprows=1)
            else:
                df_table.append(pd.read_html(str(FF_Pret_Text.find_all(
                    class_=FF_Table_Class_Regex)), skiprows=2))
            driver = ESPN_Next_Page(page, driver)
            count += 1
    return df_table


def ESPN_FF_Table(url_list):
    '''
        Calls load_ESPN_Data and runs through a list of URLs to pull out NFL
         Fantasy Stats.
    Args:
        url_list (list): List of ESPN URLs. Each URL represents a year of NFL
         Fantasy Stats.
    Returns full_df (pandas.DataFrame): DataFrame of ESPN stats
    '''
    year = 18
    full_df = pd.DataFrame()
    for url in url_list:
        df_comb = pd.DataFrame()
        df = load_FF_Data(url)
        for i in df:
            df_comb = df_comb.append(i)
        df_comb = clean_ESPN_Data(df_comb, yr=year)
        full_df = full_df.append(df_comb)
        year -= 1
    return full_df


def clean_ESPN_Data(df, yr):
    '''
        Pipeline to clean ESPN Data.
    Args:
        df (pandas.DataFrame): DataFrame that has ESPN statistics.
        yr (int): The year we are pulling statistics from. Used to append onto
        player name.
    Returns df (pandas.DataFrame): DataFrame of clean ESPN statistics
    '''
    df.columns = df.iloc[0, :]
    df = df.reset_index(drop=True)
    df = df.drop(0)
    df['PLAYER_NAME'] = df['PLAYER, TEAM POS'].apply(lambda x: x.split(',')[0])
    df['PLAYER, TEAM POS'] = df['PLAYER, TEAM POS'].apply(lambda x: x.split(',')[1])
    df['POS'] = df['PLAYER, TEAM POS'].apply(lambda x: x.split()[1])
    df['TEAM'] = df['PLAYER, TEAM POS'].apply(lambda x: x.split()[0])
    df.drop('PLAYER, TEAM POS', axis=1, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df.drop(df.columns[:-4], axis=1, inplace=True)
    df.drop(df.columns[3:], axis=1, inplace=True)
    df.pipe(remove_Non_Words, column='PLAYER_NAME', year=yr)
    df.drop_duplicates(subset='PLAYER_NAME', inplace=True)
    return df
