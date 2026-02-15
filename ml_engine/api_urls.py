from django.urls import path

from ml_engine.predict_api import api_predict, api_predict_status


urlpatterns = [
    path("predict/", api_predict, name="api_predict"),
    path("predict/tasks/<str:task_id>/", api_predict_status, name="api_predict_status"),
]
