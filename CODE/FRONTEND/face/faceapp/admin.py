from django.contrib import admin
from django.shortcuts import render
from .models import UserRegistrationForm
# Register your models here.

class Admin:
    def admin_login(req):
        if req.method == 'POST':
            ema = req.POST['aemail']
            pas = req.POST['pass']
            if ema == 'admin@gmail.com' and pas == 'admin':
                return render(req,'admin.html',{'msg':'sucess'})
        return render(req,'adminlog.html')
    
    def User_detils(req):
        user = UserRegistrationForm.objects.all()
        return render(req,'u_det.html',{'data':user})
