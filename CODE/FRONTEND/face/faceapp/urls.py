from django.urls import path

from .import views
from .admin import Admin
from .userproduct import Tshirt,Shirt,Shoes,Kid,Mens,WoMens,Camera,Car,Headset

urlpatterns = [
    path('',views.home,name='home'),
    path('login/',views.login,name='login'),
    path('register/',views.register,name='register'),
    path('uhome/',views.uhome,name='uhome'),
    path('refer/',views.refer,name='refer'),
    path('buy_pro/',views.buy_pro,name='buy_pro'),
    path('buy/',views.buy,name='buy'),
    path('add_cart/',views.add_cart,name='add_cart'),
    path('view_chart/',views.view_chart,name='view_chart'),
    path('cancel_order/',views.cancel_order,name='cancel_order'),
    path('admin_login/',Admin.admin_login,name='admin_login'),
    path('User_detils/',Admin.User_detils,name='User_detils'),
    path('tshits/',Tshirt.tshits,name='tshits'),
    path('tshitsrec/',Tshirt.tshitsrec,name='tshitsrec'),
    path('shirts/',Shirt.shirts,name='shirts'),
    path('shirtsrec/',Shirt.shirtsrec,name='shirtsrec'),
    path('shoe/',Shoes.shoe,name='shoe'),
    path('shoesrec/',Shoes.shoesrec,name='shoesrec'),
    path('kidcl/',Kid.kidcl,name='kidcl'),
    path('kidclrec/',Kid.kidclrec,name='kidclrec'),
    path('mencl/',Mens.mencl,name='mencl'),
    path('menrec/',Mens.menrec,name='menrec'),
    path('womencl/',WoMens.womencl,name='womencl'),
    path('womenrec/',WoMens.womenrec,name='womenrec'),
    path('camncl/',Camera.camncl,name='camncl'),
    path('camnrec/',Camera.camnrec,name='camnrec'),
    path('carncl/',Car.carncl,name='carncl'),
    path('carrec/',Car.carrec,name='carrec'),
    path('headcl/',Headset.headcl,name='headcl'),
    path('headrec/',Headset.headrec,name='headrec'),
    path('delinfo/',views.delinfo,name='delinfo'),
    path('uhome1/',views.uhome1,name='uhome1'),
    path('orderdet/',views.orderdet,name='orderdet'),
    path('changedate/',views.changedate,name='changedate'),
    path('adddate/',views.adddate,name='adddate'),
]
