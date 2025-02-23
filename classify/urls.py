from django.urls import path
from .views import classify_digit

urlpatterns = [
    path('', classify_digit, name='classify_digit'),
]