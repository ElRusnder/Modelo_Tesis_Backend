from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('prediction.urls')),  # Aquí incluimos las URLs de la app prediction
]
