from selenium import webdriver
from selenium.webdriver.support.select import Select
import time
import pandas as pd

PATH = '/home/joel/Downloads/chromedriver_linux64/chromedriver' # Path of the driver
driver = webdriver.Chrome(PATH)
path_df = '/home/joel/Desktop/datasets/MEIK_limpio.csv' # Path of the dataset file
path_ndf = '/home/joel/Desktop/datasets/MEIK_limpio_comp2.csv' # Path of the new dataset file

df = pd.read_csv(path_df, index_col=False)

df['RIESGO BCRAT'] = 0
driver.get('https://bcrisktool.cancer.gov/calculator.html')
time.sleep(0.5)

for idx, registro in df.iterrows():
    try:
        driver.find_element_by_css_selector('#questionAndAnswers1 > div:nth-child(3) > div').click()
        time.sleep(0.7)
        driver.find_element_by_css_selector('#questionAndAnswers2 > div:nth-child(4) > div').click()
        time.sleep(0.7)

        select_element_age = driver.find_element_by_css_selector('#age')
        select_object_age = Select(select_element_age)
        select_object_age.select_by_value(str(int(registro['EDAD'])))

        time.sleep(0.7)

        select_element_race = driver.find_element_by_css_selector('#race')
        select_object_race = Select(select_element_race)

        select_object_race.select_by_value('Hispanic')

        time.sleep(0.7)

        select_element_srace = driver.find_element_by_css_selector('#sub_race')
        select_object_srace = Select(select_element_srace)

        select_object_srace.select_by_value('Foreign Hispanic')

        time.sleep(0.7)
        driver.find_element_by_css_selector('#patient-and-family-history-section > div:nth-child(1) > div:nth-child(3) > div').click()
        time.sleep(0.7)
        if registro['MENARCA'] <= 11:
            driver.find_element_by_css_selector('#patient-and-family-history-section > div:nth-child(4) > div:nth-child(2) > div').click()
        elif registro['MENARCA'] >= 12 and registro['MENARCA'] <= 13:
            driver.find_element_by_css_selector('#patient-and-family-history-section > div:nth-child(4) > div:nth-child(3) > div').click()
        else:
            driver.find_element_by_css_selector('#patient-and-family-history-section > div:nth-child(4) > div:nth-child(4) > div').click()

        time.sleep(0.7)

        select_element_ageb = driver.find_element_by_css_selector('#childbirth_age')
        select_object_ageb = Select(select_element_ageb)
        if registro['HIJO'] == 2:
            select_object_ageb.select_by_visible_text('No Births')
        else:
            select_object_ageb.select_by_visible_text('Unknown')

        time.sleep(0.7)
        if registro['AF'] == 3:
            driver.find_element_by_css_selector('#patient-and-family-history-section > div:nth-child(6) > div:nth-child(2) > div').click()
        else:
            driver.find_element_by_css_selector('#patient-and-family-history-section > div:nth-child(6) > div:nth-child(3) > div').click()

        time.sleep(0.7)
        driver.find_element_by_css_selector('#calculate').click()

        time.sleep(0.7)
        df.loc[idx, 'RIESGO BCRAT'] = driver.find_element_by_css_selector('#Risk1').text[:-1]


        driver.find_element_by_css_selector('#startOverButton').click()

        time.sleep(0.7)
        print((idx+1)/len(df))
    except Exception as e:
        print(e)
        df.to_csv(path_ndf,index=False)
        exit()

df.to_csv(path_ndf,index=False)
