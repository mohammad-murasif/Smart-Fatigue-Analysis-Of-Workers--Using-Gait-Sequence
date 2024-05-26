import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
import json
from .models import *
from myProject.settings import BASE_DIR
from .util.identification import FaceRecognition
from .util.test import preProccessVideo
from django.shortcuts import  get_object_or_404
from django.http import JsonResponse
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
import json

from .util.pretreatment import *



class VideoConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        await self.stream_video_feed()

    async def disconnect(self, close_code):
        pass

    async def stream_video_feed(self):
        try:
            async for frame in videoFeed():
                await self.send(bytes_data=frame)
        except Exception as e:
            print("Error streaming video feed:", e)
