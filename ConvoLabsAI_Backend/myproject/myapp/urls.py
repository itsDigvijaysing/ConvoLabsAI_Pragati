# myapp/urls.py
from django.urls import path
from . import views # Assuming views.py is in the same directory (your app)

urlpatterns = [
    # This path should match the endpoint your React frontend is calling.
    # E.g., if React calls '/api/voice-input/', then it should be:
    path('api/voice-input/', views.process_voice_input_offline, name='process_voice_input_api'),
]