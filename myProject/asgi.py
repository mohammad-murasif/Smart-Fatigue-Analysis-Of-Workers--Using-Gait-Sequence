import os


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myProject.settings')

from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
django_asgi_app = get_asgi_application()

from channels.auth import AuthMiddlewareStack   
import myApp.routing


application = ProtocolTypeRouter({
    "http": django_asgi_app,
    'websocket': AuthMiddlewareStack(
        URLRouter(
            myApp.routing.websocket_urlpatterns
        )
    )
})