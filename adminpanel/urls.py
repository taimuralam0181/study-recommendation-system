from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/', views.teacher_dashboard, name='teacher_dashboard'),
    path('students/', views.student_list, name='student_list'),
    path('materials/', views.material_list, name='material_list'),
    path('materials/add/', views.add_material, name='teacher_add_material'),
    path('materials/<int:material_id>/edit/', views.edit_material, name='teacher_edit_material'),
    path('materials/<int:material_id>/delete/', views.delete_material, name='teacher_delete_material'),
    path('subjects/', views.subject_list, name='teacher_subjects'),
    path('subjects/add/', views.add_subject, name='teacher_add_subject'),
    path('subjects/<int:subject_id>/edit/', views.edit_subject, name='teacher_edit_subject'),
    path('subjects/<int:subject_id>/deactivate/', views.deactivate_subject, name='teacher_deactivate_subject'),
    path('subjects/<int:subject_id>/delete/', views.delete_subject, name='teacher_delete_subject'),
    path('upload-dataset/', views.upload_dataset, name='upload_dataset'),
    path('models/<int:model_id>/download/', views.download_model_version, name='download_model_version'),
    path('analytics/', views.analytics, name='teacher_analytics'),
    path('analytics/export/', views.analytics_export_csv, name='teacher_analytics_export'),
    path('recovery/', views.recovery_overview, name='teacher_recovery_overview'),
    path('recovery/<int:user_id>/semester/<int:semester_number>/', views.recovery_student_semester, name='teacher_recovery_student_semester'),
    path('recovery/<int:user_id>/semester/<int:semester_number>/export/', views.recovery_student_semester_export, name='teacher_recovery_export'),
]
