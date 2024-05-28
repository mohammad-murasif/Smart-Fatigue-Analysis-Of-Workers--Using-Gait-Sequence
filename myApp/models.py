from django.db import models
from .validators import file_size
from django.core.validators import RegexValidator
from django.contrib.auth.models import User

class Supervisors(models.Model):
    supervisorId = models.AutoField(primary_key=True)
    name = models.CharField(verbose_name="Supervisor Name",max_length=100)
    phone_regex = RegexValidator(regex=r'^\+?1?\d{9,10}$', message="Enter phone number without +91")
    phone_num = models.CharField(validators=[phone_regex], max_length=112, blank=False, unique=True) # Validators should be a list
    email= models.EmailField(verbose_name="Email",max_length=255,blank=False,unique=True,null=False)    
    def __str__(self):
        return str(self.name)




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
