from django.urls import path
from .import views

urlpatterns=[
    path('',views.index,name='index'),
    path('register/',views.register,name='register'),
    path('login/',views.login,name='login'),
    path('home/',views.home,name='home'),
    path('about/',views.about,name='about'),
    path('upload/',views.upload,name='upload')
]
