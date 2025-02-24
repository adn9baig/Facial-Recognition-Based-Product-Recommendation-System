from django.shortcuts import render,redirect
from sklearn.preprocessing import LabelEncoder
from .prepro import Prepro
import random
import pandas as pd
import numpy  as np
from .recomcode import Recommendation




class Tshirt(object):
    def tshits(req):
        tdata = 'DATASET/T-shirts and Polos.csv'
        df = Prepro.pre(tdata)
        print(df.columns)
        name =  df['name'][:15].to_list()
        price = df['actual_price'][:15].to_list()
        image = df['image'][:15].to_list()
        print(name)
        return render(req,'products.html',{'name':name,'price':price,'image':image,'msg1':'tshirt'})
    def tshitsrec(req):
        try:
            if req.method == 'POST':
                global name,price,image
                inp = req.POST['text']
                tdata = 'DATASET/T-shirts and Polos.csv'
                df = Prepro.pre(tdata)
                print(df.columns)        
                # Find unique names
                re  =  Recommendation()
                name,price,image = re.recom(df,inp)
                name = name
                price = price
                image = image
                data = name,price,image
                return render(req,'products.html',{'name':name,'price':price,'image':image,'msg1':'tshirt'})
        except:
            return redirect(req,Tshirt.tshitsrec)

class Shirt(object):
    def shirts(req):
        tdata = 'DATASET/shirts.csv'
        df = Prepro.pre(tdata)
        print(df.columns)
        name =  df['name'][:15].to_list()
        price = df['actual_price'][:15].to_list()
        image = df['image'][:15].to_list()
        print(name)
        return render(req,'products.html',{'name':name,'price':price,'image':image,'msg1':'shirt'})
    def shirtsrec(req):
        if req.method == 'POST':
            inp = req.POST['text']
            tdata = 'DATASET/shirts.csv'
            df = Prepro.pre(tdata)
            print(df.columns)        
            # Find unique names
            re  =  Recommendation()
            name,price,image = re.recom(df,inp)
            name = name
            price = price
            image = image
            data = name,price,image
            return render(req,'products.html',{'name':name,'price':price,'image':image,'msg1':'shirt'})
        
        
class Shoes(object):
    def shoe(req):
        tdata = 'DATASET/Shoes.csv'
        df = Prepro.pre(tdata)
        print(df.columns)
        name =  df['name'][:15].to_list()
        price = df['actual_price'][:15].to_list()
        image = df['image'][:15].to_list()
        print(name)
        return render(req,'products.html',{'name':name,'price':price,'image':image,'msg1':'Shoes'})
    def shoesrec(req):
        if req.method == 'POST':
            inp = req.POST['text']
            tdata = 'DATASET/Shoes.csv'
            df = Prepro.pre(tdata)
            print(df.columns)        
            # Find unique names
            re  =  Recommendation()
            name,price,image = re.recom(df,inp)
            name = name
            price = price
            image = image
            data = name,price,image
            return render(req,'products.html',{'name':name,'price':price,'image':image,'msg1':'Shoes'})
        
        
class Kid(object):
    def kidcl(req):
        tdata = 'DATASET/Kids Clothing.csv'
        df = Prepro.pre(tdata)
        print(df.columns)
        name =  df['name'][:15].to_list()
        price = df['actual_price'][:15].to_list()
        image = df['image'][:15].to_list()
        print(name)
        return render(req,'products.html',{'name':name,'price':price,'image':image,'msg1':'kid'})
    def kidclrec(req):
        if req.method == 'POST':
            inp = req.POST['text']
            tdata = 'DATASET/Kids Clothing.csv'
            df = Prepro.pre(tdata)
            print(df.columns)        
            # Find unique names
            re  =  Recommendation()
            name,price,image = re.recom(df,inp)
            name = name
            price = price
            image = image
            data = name,price,image
            return render(req,'products.html',{'name':name,'price':price,'image':image,'msg1':'kid'})
        
        
class Mens(object):
    def mencl(req):
        tdata = 'DATASET/Mens Fashion.csv'
        df = Prepro.pre(tdata)
        print(df.columns)
        name =  df['name'][:15].to_list()
        price = df['actual_price'][:15].to_list()
        image = df['image'][:15].to_list()
        print(name)
        return render(req,'products.html',{'name':name,'price':price,'image':image,'msg1':'mens'})
    def menrec(req):
        if req.method == 'POST':
            inp = req.POST['text']
            tdata = 'DATASET/Mens Fashion.csv'
            df = Prepro.pre(tdata)
            print(df.columns)        
            # Find unique names
            re  =  Recommendation()
            name,price,image = re.recom(df,inp)
            name = name
            price = price
            image = image
            data = name,price,image
            return render(req,'products.html',{'name':name,'price':price,'image':image,'msg1':'mens'})
        
        
class WoMens(object):
    def womencl(req):
        tdata = 'DATASET/Womens Fashion.csv'
        df = Prepro.pre(tdata)
        print(df.columns)
        name =  df['name'][:15].to_list()
        price = df['actual_price'][:15].to_list()
        image = df['image'][:15].to_list()
        print(name)
        return render(req,'products.html',{'name':name,'price':price,'image':image,'msg1':'womens'})
    def womenrec(req):
        if req.method == 'POST':
            inp = req.POST['text']
            tdata = 'DATASET/Womens Fashion.csv'
            df = Prepro.pre(tdata)
            print(df.columns)        
            # Find unique names
            re  =  Recommendation()
            name,price,image = re.recom(df,inp)
            name = name
            price = price
            image = image
            data = name,price,image
            return render(req,'products.html',{'name':name,'price':price,'image':image,'msg1':'womens'})
        
class Camera(object):
    def camncl(req):
        tdata = 'DATASET/Camera Accessories.csv'
        df = Prepro.pre(tdata)
        print(df.columns)
        name =  df['name'][:15].to_list()
        price = df['actual_price'][:15].to_list()
        image = df['image'][:15].to_list()
        print(name)
        return render(req,'products.html',{'name':name,'price':price,'image':image,'msg1':'cam'})
    def camnrec(req):
        if req.method == 'POST':
            inp = req.POST['text']
            tdata = 'DATASET/Camera Accessories.csv'
            df = Prepro.pre(tdata)
            print(df.columns)        
            # Find unique names
            re  =  Recommendation()
            name,price,image = re.recom(df,inp)
            name = name
            price = price
            image = image
            data = name,price,image
            return render(req,'products.html',{'name':name,'price':price,'image':image,'msg1':'cam'})
        
        
class Car(object):
    def carncl(req):
        tdata = 'DATASET/Car and Bike Care.csv'
        df = Prepro.pre(tdata)
        print(df.columns)
        name =  df['name'][:15].to_list()
        price = df['actual_price'][:15].to_list()
        image = df['image'][:15].to_list()
        print(name)
        return render(req,'products.html',{'name':name,'price':price,'image':image,'msg1':'car'})
    def carrec(req):
        if req.method == 'POST':
            inp = req.POST['text']
            tdata = 'DATASET/Car and Bike Care.csv'
            df = Prepro.pre(tdata)
            print(df.columns)        
            # Find unique names
            re  =  Recommendation()
            name,price,image = re.recom(df,inp)
            name = name
            price = price
            image = image
            data = name,price,image
            return render(req,'products.html',{'name':name,'price':price,'image':image,'msg1':'car'})
        
        
class Headset(object):
    def headcl(req):
        tdata = 'DATASET/Headphones.csv'
        df = Prepro.pre(tdata)
        print(df.columns)
        name =  df['name'][:15].to_list()
        price = df['actual_price'][:15].to_list()
        image = df['image'][:15].to_list()
        print(name)
        return render(req,'products.html',{'name':name,'price':price,'image':image,'msg1':'head'})
    def headrec(req):
        if req.method == 'POST':
            inp = req.POST['text']
            tdata = 'DATASET/Headphones.csv'
            df = Prepro.pre(tdata)
            print(df.columns)        
            # Find unique names
            re  =  Recommendation()
            name,price,image = re.recom(df,inp)
            name = name
            price = price
            image = image
            data = name,price,image
            return render(req,'products.html',{'name':name,'price':price,'image':image,'msg1':'head'})