from .models import Video, workerProfile, worker
from django import forms



class Video_form(forms.ModelForm):
    class Meta:
        model=Video
        fields=['video']
        
class workerProfile_form(forms.ModelForm):
    class Meta:
        model=workerProfile
        fields=['id', 'photo']
        
class worker_registerForm(forms.ModelForm):
    class Meta:
        model=worker
        fields=['workerName', 'workerPhoneNo']