from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class MovieReview(models.Model):
    review = models.TextField()
    sentiment = models.IntegerField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    create_date = models.DateTimeField()
    modify_date= models.DateTimeField(null=True, blank=True)