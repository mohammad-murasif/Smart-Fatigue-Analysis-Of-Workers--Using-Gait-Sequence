from .models import Video, worker
from django import forms



class Video_form(forms.ModelForm):
    class Meta:
        model=Video
        fields=['video']
        

        
class worker_registerForm(forms.ModelForm):
    class Meta:
        model=worker
        fields=['workerName', 'workerPhoneNo']