from django import forms

class UserRegistrationForm(forms.Form):
    name = forms.CharField(max_length=100)
    address = forms.CharField(max_length=200)
    phone_number = forms.CharField(max_length=15)
    face_data = forms.CharField(widget=forms.HiddenInput())  