from django.contrib import messages
from django.contrib.auth.decorators import login_required

from django.utils import timezone

from django.core.paginator import Paginator
from django.db.models import Q, Count
from django.shortcuts import render, get_object_or_404, redirect, resolve_url

from .models import MovieReview
from .forms import FeedbackForm, MovieReviewForm
from .vectorizer import vect
import pickle
import os
import numpy as np


# Create your views here.

cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,
                 'pkl_objects',
                 'classifier.pkl'), 'rb'))

def classify(document):
    label = {0: 'negative', 1: 'positive'}
    X = vect.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])



def movie_review(request):
    """
    영화리뷰분류기 메인
    """
     # 입력 파라미터
    page = request.GET.get('page', '1')  # 페이지
    kw = request.GET.get('kw', '')  # 검색어

    review_list = MovieReview.objects.order_by('-create_date')
    
    if kw:
        review_list = review_list.filter(
            Q(review__icontains=kw)   #  리뷰검색
        ).distinct()

    paginator = Paginator(review_list, 10)
    page_obj = paginator.get_page(page)

    context = {'review_list': page_obj, 'page': page, 'kw': kw}  


    return render(request, 'movieReview/review_list.html', context)


def review_detail(request, review_id):
    """
    영화리뷰분류기 상세조회
    """
    review = get_object_or_404(MovieReview, pk=review_id)
    context = {'movieReview': review}
    return render(request, 'movieReview/review_detail.html', context)


@login_required(login_url='common:login')
def review_create(request):
    """
    review 리뷰등록
    """
    if request.method == 'POST':
        form = MovieReviewForm(request.POST)
        if form.is_valid():
            movieReview = form.save(commit=False)
            y, proba = classify(movieReview.review)

            prediction =''
            if y =='positive':
                 movieReview.sentiment = 1
                 prediction='긍정'
            else:
                 movieReview.sentiment = 0
                 prediction='부정'

            print("predict:",y)

            movieReview_obj = MovieReview(review=movieReview.review, sentiment=movieReview.sentiment)
            feedbackForm=FeedbackForm(instance=movieReview_obj)
            context = {'prediction':prediction,'probabilty':round(proba*100,2),'form':feedbackForm}
            return render(request,'movieReview/result.html',context)
    else:
        form = MovieReviewForm()
    context = {'form': form}
    return render(request, 'movieReview/review_form.html', context)


@login_required(login_url='common:login')
def review_feedback(request):
    form = FeedbackForm(request.POST)
    if request.method == 'POST':
        if form.is_valid():
            feedbackValue = request.POST.get('feedback')
            feedback = form.save(commit=False)
            feedback.author = request.user  # 추가한 속성 author 적용
            feedback.create_date = timezone.now()

            if feedbackValue =='N':
                feedback.sentiment = int(not(feedback.sentiment))
            train(feedback.review,feedback.sentiment)
            feedback.save()
            return render(request, 'movieReview/review_thanks.html')
    else:
        form=FeedbackForm()
    context = {'form': form}
    return render(request, 'movieReview/review_thanks.html',context)