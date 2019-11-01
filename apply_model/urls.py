from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views
# from .views import ModelCreateView

urlpatterns = [
    path('', views.home,name='apply_model-home'),
    path('regression/', views.regression,name='apply_model-regression'),
    path('upload/', views.upload,name='apply_model-upload'),
    path('predict/', views.predict,name='apply_model-predict'),
    path('predict_upload/', views.predict_upload,name='apply_model-predict_upload'),    
    path('cv/', views.cv, name='apply_model-cv')
]
