import os
import django
import csv
os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
django.setup()

from movieReview.models import MovieReview
from django.contrib.auth.models import User
from django.utils import timezone

f = open('movieReview/movie_data.csv','r', encoding='utf-8')
info=[]

rdr = csv.reader(f)

for row in rdr:
    review,sentiment=row
    tuple =(review,sentiment)
    info.append(tuple)
f.close()

instances = []

user = User()
user.id= 1

for (review,sentiment) in info:
  
    instances.append(MovieReview(review=review, sentiment=sentiment,author=user,create_date=timezone.now()))

MovieReview.objects.bulk_create(instances)