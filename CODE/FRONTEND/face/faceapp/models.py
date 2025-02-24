from django.db import models

# Create your models here.

class UserRegistrationForm(models.Model):
    name = models.CharField(max_length=100)
    address = models.CharField(max_length=200)
    phone_number = models.CharField(max_length=15)
    
    
class Logindetail(models.Model):
    name12 = models.CharField(max_length=100)
    count = models.CharField(max_length=15)
    
    
class Product(models.Model):
    name1 = models.CharField(max_length=100)
    image1 = models.CharField(max_length=200)
    price1 = models.CharField(max_length=200)
    name_id = models.SmallIntegerField(max_length=200)
    
    
    
class Add_chart(models.Model):
    u_name = models.CharField(max_length=100)
    card_num = models.CharField(max_length=100)
    cvv = models.CharField(max_length=100)
    ba_name  = models.CharField(max_length=100)
    p_pin = models.CharField(max_length=100)
    m_num = models.CharField(max_length=100)
    de_date = models.CharField(max_length=100)
    name2 = models.CharField(max_length=100)
    image2 = models.CharField(max_length=200)
    price2 = models.CharField(max_length=200)
    b_id = models.SmallIntegerField(max_length=200,null=True)
    
class UserLogdet(models.Model):
    u_id = models.SmallIntegerField(max_length=100)