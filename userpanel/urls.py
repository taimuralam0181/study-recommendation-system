from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/', views.dashboard, name='student_dashboard'),
    path('performance/', views.performance_overview, name='performance_overview'),
    path('performance/semester/<int:semester_number>/', views.semester_detail, name='semester_detail'),
    path('performance/semester/<int:semester_number>/subject/<int:subject_id>/', views.enter_subject_marks, name='enter_subject_marks'),
    path('performance/semester/<int:semester_number>/subject/<int:subject_id>/plan/', views.recovery_plan_view, name='recovery_plan'),
    path('performance/semester/<int:semester_number>/plan/', views.semester_plan, name='semester_plan'),
]
