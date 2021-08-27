from django import forms
from movieReview.models import MovieReview

class MovieReviewForm(forms.ModelForm):
    class Meta:
        model=MovieReview
        fields= ['review']
        labels ={
            'review':'리뷰',
        }



class FeedbackForm(forms.ModelForm):
    class Meta:
        model = MovieReview
        fields = ['review','sentiment']
        labels ={
             'review':'리뷰',
             'sentiment':'평점',
        }