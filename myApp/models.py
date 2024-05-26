from django.db import models
from .validators import file_size
from django.core.validators import RegexValidator
from django.contrib.auth.models import User

class workerProfile(models.Model):
    photo = models.ImageField(upload_to='media/profile', default='media/profile/default.jpg', verbose_name='profile')
    workerId = models.ForeignKey('worker', verbose_name="worker Id", on_delete=models.CASCADE)

class worker(models.Model):
    workerId = models.AutoField(primary_key=True)
    workerName = models.CharField(verbose_name="Worker Name", max_length=100)
    # phone_regex = RegexValidator(regex=r'^\+?1?\d{9,10}$', message="Enter phone number without +91")
    workerPhoneNo = models.CharField(max_length=20, blank=False, unique=False)
    def __str__(self):
        return str(self.workerId)

class Video(models.Model):  
    id = models.AutoField(primary_key=True)
    video=models.FileField(upload_to="video/%y",validators=[file_size])
    
    def __str__(self):
        return str(self.id)
