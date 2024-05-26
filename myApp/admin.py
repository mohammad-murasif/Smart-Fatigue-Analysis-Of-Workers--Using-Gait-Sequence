from django.contrib import admin

# Register your models here.
from .models import Video, workerProfile, worker
admin.site.register(Video)
admin.site.register(worker)
admin.site.register(workerProfile)