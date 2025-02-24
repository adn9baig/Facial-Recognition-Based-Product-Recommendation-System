from django.shortcuts import render
import pandas as pd


class Prepro:
    def pre(data):
        df = pd.read_csv(data,encoding='latin')
        df.ffill(inplace=True)
        df = df[['name','image','actual_price']]
        unknown_symbol = 'â¹'
        df['actual_price'] = df['actual_price'].apply(lambda x: x.replace(unknown_symbol, ''))
        df['actual_price'] = df['actual_price'].replace(',', '', regex=True).astype(float)
        return df