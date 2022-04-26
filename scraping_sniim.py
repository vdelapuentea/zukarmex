# -*- coding: utf-8 -*-
"""Scraping_sniim.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lA4HUbfnaUfT5LXOkDhKj27J70phWxqw
"""

!pip install selenium
!apt-get update # to update ubuntu to correctly run apt install
!apt install chromium-chromedriver
!cp /usr/lib/chromium-browser/chromedriver /usr/bin

from selenium.webdriver.support.ui import Select
import pandas as pd
from bs4 import BeautifulSoup
#from selenium.webdriver.common.by import By
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import re
from dateutil.rrule import rrule, MONTHLY
from datetime import datetime

def scraping(mesInicial,anioInicial,mesFinal,anioFinal):
  sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
  chrome_options = webdriver.ChromeOptions()
  chrome_options.add_argument('--headless')
  chrome_options.add_argument('--no-sandbox')
  chrome_options.add_argument('--disable-dev-shm-usage')

  browser = webdriver.Chrome('chromedriver',chrome_options=chrome_options)
  def get_url(Mes,Anio):
    return "http://www.economia-sniim.gob.mx/AzucarMesPorDiaXregion.asp?Cons=D&prod=156&dqMesMes={Mes}&dqAnioMes={Anio}&Formato=Xls&submit=Ver+Resultados".format(Mes=Mes,Anio=Anio)

  def months(start_month, start_year, end_month, end_year):
    start = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    return [(d.month, d.year) for d in rrule(MONTHLY, dtstart=start, until=end)]
  df2=pd.DataFrame()
  a= months(mesInicial,anioInicial,mesFinal,anioFinal)
  
  for i in range(0,len(a),1):
    data = requests.get(get_url(a[i][0],a[i][1])).text
    anio=a[i][1]
    soup = BeautifulSoup(data, 'html.parser')
    table = soup.find_all('table')
    for row in table[17]:
        text = row.find_all('td')
    list_header = [] 
    header = table[17].find("tr") 
    for items in header: 
        try: 
            list_header.append(items.get_text()) 
        except: 
            continue
    data = [] 
    HTML_data = table[17].find_all("tr")[1:]
    for element in HTML_data: 
        sub_data = [] 
        for sub_element in element: 
            try: 
                sub_data.append(sub_element.get_text()) 
            except: 
                continue
        data.append(sub_data) 
    df = pd.DataFrame(data = data[3:])
    df.columns=df.iloc[0,:].values
    df=df.loc[1:,:]
    df=df.loc[df["Centros de Distribución"].str.contains('Centro') != True]
    df=df.loc[df["Centros de Distribución"].str.contains('Occidente') != True]
    df=df.loc[df["Centros de Distribución"].str.contains('Noroeste') != True]
    df=df.loc[df["Centros de Distribución"].str.contains('Noreste') != True]
    df=df.loc[df["Centros de Distribución"].str.contains('Golfo') != True]
    df=df.loc[df["Centros de Distribución"].str.contains('Sureste') != True]  	
    df=df.loc[df["Centros de Distribución"].str.contains('Fuente: Precios:') != True]
    df=df.loc[df["Centros de Distribución"] != ""] 
    df.drop(["PromMes 1/"], axis=1,inplace=True)
    df.set_index('Centros de Distribución',inplace=True)
    df = df.rename(columns=lambda x: re.sub('\xa0\xa0',' ',x))
    df=df.T
    df.reset_index(inplace=True)
    #df = df.rename(columns=lambda x: re.sub('\xa0',' ',x))
    df.rename(columns={'index':'fecha'},inplace=True)
    df = df.rename(columns=lambda x: re.sub('\xa0',' ',x))

    col=list(df.columns)
    col.remove("fecha")

    for i in range(len(col)):
      df1=df[["fecha", str(col[i]) ]] 
      df1["Centro_distribucion"]= col[i]
      df1["Año"]= anio
      df1.rename(columns={col[i]:'valor'},inplace=True)
      df1['mes'] = df1['fecha'].str[-3:]
      df1['dia'] = df1['fecha'].str[-6:-4]
      df1['nombre_dia'] = df1['fecha'].str[0:4]
      df2= pd.concat([df2,df1])

  return df2

df2=scraping(10,2000,4,2022)

df2.shape #64727

pd.value_counts(df2["Año"])

df2.to_csv("df2.csv")

df2.head()

