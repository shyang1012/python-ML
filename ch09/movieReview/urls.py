from django.urls import path

from . import views

app_name = 'movieReview'

urlpatterns = [
    path('movieReview/', views.movie_review, name='movie_review'),
    path('movieReview/detail/<int:review_id>', views.review_detail, name='review_detail'),
    path('movieReview/create', views.review_create, name='review_create'),
    path('movieReview/create/feedback', views.review_feedback, name='review_feedback'),
]
