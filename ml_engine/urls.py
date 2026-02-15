from django.urls import path
from . import views
from .predict_api import api_predict

urlpatterns = [
    path('', views.index, name='ml_index'),
    path('add-material/', views.add_material, name='add_material'),
    path('recommend/', views.get_recommendation, name='get_recommendation'),
    path('api/predict/', api_predict, name='ml_api_predict'),
]
