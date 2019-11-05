import os
import joblib
import pandas as pd
import numpy as np
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .backend import BackendRegression
from django.core.files import File
from .post_options import PostOptions

from django.contrib import messages


# Create your views here.
def home(request):        
    return render(request,'apply_model/home.html')

def regression(request):
    return render(request,'apply_model/regression.html')

def upload(request):    

    if request.method=='POST':        
        uploaded_file=request.FILES['regression_dataset'] if 'regression_dataset' in request.FILES else False  
        
        if uploaded_file == False:
            messages.warning(request,'Please upload a .csv file first')                  
            return render(request,'apply_model/regression.html')
        else:
        
            uploaded_filename=uploaded_file.name
            uploaded_filename=uploaded_filename.split(".csv")[0]               
            
            method=request.POST.get('regression_method')
            approach=request.POST.get('regression_approach')
            
            post_request=PostOptions(approach)

            if method=="linear_regression":            
                hyperparameters=post_request.linear_regression(request)
                            
            elif method=="decision_tree":                        
                hyperparameters=post_request.decision_tree(request)            
                                                        
            elif method=="mlp":                        
                hyperparameters=post_request.mlp(request)            

            elif method=="random_forest":                    
                hyperparameters=post_request.random_forest(request)
                    
            elif method=="svr":                        
                hyperparameters=post_request.svr(request)
            
            approach_parameters=post_request.regression_approach(request)

            my_context={
                "method":request.POST.get('regression_method'),
                "approach":request.POST.get('regression_approach'),
                "dataset":uploaded_file,   
                "hyperparameters":hyperparameters,
                "approach_parameters":approach_parameters
            }                    
            if approach == "grid_search":
                my_context.update({"approach_parameters_kfold":{
                    "n_splits":int(request.POST.get("kfold_n_splits")),
                    "random_state":int(request.POST.get("kfold_seed")),
                    "shuffle":True
                }
                })
            
            model=BackendRegression(my_context)
            model.backend()
            model.path="static/apply_model/"+uploaded_filename+'_'+method+'_'+approach+".sav"
            
            global correct_filename,correct_file

            correct_filename = uploaded_filename+'_'+method+'_'+approach        
            correct_file = model.path
            
            # if os.path.exists("apply_model/static/apply_model/model.sav"):
            #     os.remove("apply_model/static/apply_model/model.sav")
            joblib.dump(model.model,"apply_model/"+model.path)        
            
            my_context.update({"model":model.model})        
            my_context.update({"model_path":model.path})
            my_context.update({"mae_train":model.metrics_training["mae"]})        
            my_context.update({"mse_train":model.metrics_training["mse"]})        
            my_context.update({"ex_variance_train":model.metrics_training["ex_variance"]})        
            my_context.update({"r2_train":model.metrics_training["r2"]})        

            if approach!="full_training":            
                my_context.update({"mae_test":model.metrics_testing["mae"]})        
                my_context.update({"mse_test":model.metrics_testing["mse"]})        
                my_context.update({"ex_variance_test":model.metrics_testing["ex_variance"]})        
                my_context.update({"r2_test":model.metrics_testing["r2"]})                                                  

            return render(request,'apply_model/upload.html',my_context)

def predict(request):            
    return render(request,'apply_model/predict.html')

def predict_upload(request):
    if request.method=='POST':
        uploaded_file=request.FILES['prediction_dataset'] 
        uploaded_model=request.FILES['prediction_model']
                                
        model=joblib.load(uploaded_model)  
        
        my_context={
            "model":model,
            "dataset":uploaded_file,               
        }         

        y_pred=BackendRegression(my_context)
        y_pred.predict()        

        my_context={
            "predictions":y_pred.predictions
        }

    return render(request,'apply_model/predict_upload.html',my_context)        

def cv(request):
    return render(request, 'apply_model/cv.html')