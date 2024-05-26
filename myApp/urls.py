from django.contrib import admin
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views
from django.urls import path, include

urlpatterns = [
    path("",views.index,name="index"),


    # path('home/', views.home, name='home'),
    path('test/', views.test, name='test'),
    path('train/', views.registerWorker, name='train'),
    path('results/',views.results,name='results'),
    path('resultstest/', views.resultstest, name='resultstest'),
    path('trainrecog',views.UpdateWorkerFaceData,name='trainrecog'),
    path('trainWorkerFaces',views.trainWorkerFaces,name='trainforkerfaces'),
    path('identifyWorker/',views.identifyWorkerView,name='identifyWorker'),
    path('get_identfication_status/<str:task1_id>/', views.get_status_identfication, name='get_status_identfication'),

    path('video_feed/', views.video_feed, name='video_feed'),
    path('get_status/<str:task_id>/', views.get_status, name='get_status')



    # Other URL patterns for your project...
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
