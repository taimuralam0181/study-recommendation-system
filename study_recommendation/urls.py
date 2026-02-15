from django.contrib import admin
from django.urls import path, include
from accounts import views as accounts_views
from django.conf import settings
from django.conf.urls.static import static

# Cyberpunk Dashboard imports
import sys
sys.path.insert(0, 'templates/dashboard')
from templates.dashboard import api as dashboard_api

urlpatterns = [
    path('admin/', admin.site.urls),

    # Auth
    path('', accounts_views.login_view, name='login'),
    path('register/', accounts_views.register_view, name='register'),
    path('logout/', accounts_views.logout_view, name='logout'),
    # Removed Django auth URLs to avoid conflicting login/logout routes.

    # Student Dashboard (Cyberpunk Theme)
    path('dashboard/', accounts_views.dashboard_view, name='dashboard'),
    path('dashboard/cyber/', accounts_views.cyberpunk_dashboard, name='cyberpunk_dashboard'),
    path('dashboard/api/', dashboard_api.api_dashboard, name='api_dashboard'),
    path('dashboard/api/predict/', dashboard_api.api_predict_grade, name='api_predict_grade'),
    path('dashboard/api/performance/', dashboard_api.api_performance_history, name='api_performance_history'),
    path('dashboard/api/materials/', dashboard_api.api_study_materials, name='api_study_materials'),

    # Teacher (adminpanel)
    path('teacher/', include('adminpanel.urls')),

    # ML ENGINE (🔥 THIS WAS MISSING)
    path('ml/', include('ml_engine.urls')),
    path('api/', include('ml_engine.api_urls')),
    path('student/', include('userpanel.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
