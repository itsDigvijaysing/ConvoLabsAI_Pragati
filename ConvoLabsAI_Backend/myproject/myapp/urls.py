from django.urls import path
from . import views  # Ensure you're importing from the correct views.py file

urlpatterns = [
    path('process_audio/', views.process_audio_request, name='process_audio'),
]
