from django import forms
from .models import Transcription

class AudioUploadForm(forms.ModelForm):
    class Meta:
        model = Transcription
        fields = ["audio_file"]
