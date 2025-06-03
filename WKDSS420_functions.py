# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:07:25 2023

@author: Hiren Patel
"""
#*****************************************************************************
# Functions for use in the notebooks

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

'''  
Module 2 Recommender Systems  
'''
# MODIFIED TO RETURN MOVIE TITLES
from collections import defaultdict
#movie_titles = pd.read_csv("Movie_Id_Titles",index_col='item_id')

def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    movie_titles = pd.read_csv("Movie_Id_Titles",index_col='item_id')
    
    for uid, iid, true_r, est, _ in predictions:
        movie_iid = movie_titles.loc[int(iid)]
        top_n[uid].append((movie_iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n
