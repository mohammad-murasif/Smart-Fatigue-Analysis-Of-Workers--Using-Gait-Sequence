from django.contrib import admin
from .models import Video,  worker,Supervisors
admin.site.register(Supervisors)
admin.site.register(Video)
admin.site.register(worker)
