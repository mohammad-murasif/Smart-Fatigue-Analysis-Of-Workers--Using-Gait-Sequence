# Generated by Django 5.0.5 on 2024-05-08 21:19

import myApp.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('myApp', '0002_delete_video'),
    ]

    operations = [
        migrations.CreateModel(
            name='Video',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('caption', models.CharField(max_length=100)),
                ('video', models.FileField(upload_to='video/%y', validators=[myApp.validators.file_size])),
            ],
        ),
    ]