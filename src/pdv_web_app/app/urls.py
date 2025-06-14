from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('login', views.LoginView.as_view(), name='login'),
    path('logout', views.logout_view, name='logout'),
    path('register', views.register_view, name='register'),
    path('process_cnn', views.process_cnn, name='process_cnn'),
    path('performance_check', views.performance_check, name='performance_check'),
]