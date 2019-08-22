from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from .backend import BackendRegression


# Create your views here.
def home(request):    
    return render(request,'apply_model/home.html')

def regression(request):
    return render(request,'apply_model/regression.html')

def upload(request):
    
    if request.method=='POST':
        uploaded_file=request.FILES['regression_dataset']                                 
        
        method=request.POST.get('regression_method')
        approach=request.POST.get('regression_approach')

        if method=="linear_regression":
            hyperparameters={
                "fit_intercept":request.POST.get("linear_fit_intercept")                
            }
        elif method=="decision_tree":
            hyperparameters={
                "criterion":request.POST.get("decision_tree_criterion"),
                "splitter":request.POST.get("decision_tree_splitter"),
                "max_depth":request.POST.get("decision_tree_max_depth"),
                "min_samples_split":int(request.POST.get("decision_tree_min_samples_split")),
                "min_samples_leaf":int(request.POST.get("decision_tree_min_samples_leaf")),
                "max_leaf_nodes":request.POST.get("decision_tree_max_leaf_nodes")
            }
            if(hyperparameters["max_depth"]=="None"):
                hyperparameters["max_depth"]=None
            else:
                hyperparameters["max_depth"]=int(hyperparameters["max_depth"])
            if(hyperparameters["max_leaf_nodes"]=="None"):
                hyperparameters["max_leaf_nodes"]=None
            else:
                hyperparameters["max_leaf_nodes"]=int(hyperparameters["max_leaf_nodes"])
        elif method=="mlp":
            hyperparameters={
                "hidden_layer_sizes":request.POST.get("mlp_hidden_layer_sizes"),
                "activation":request.POST.get("mlp_activation"),
                "solver":request.POST.get("mlp_solver"),
                "alpha":float(request.POST.get("mlp_alpha")),
                "max_iter":int(request.POST.get("mlp_max_iter")),
                "tol":float(request.POST.get("mlp_tol"))
            }

            hidden_layers=hyperparameters["hidden_layer_sizes"]            
            hidden_layers=hidden_layers.split(",")            
            hidden_layers=tuple(hidden_layers)
            hidden_layers=tuple(int(x) if x.isdigit() else x for x in\
                 hidden_layers if x)
            
            hyperparameters["hidden_layer_sizes"]=hidden_layers
            
        elif method=="random_forest":
            hyperparameters={
                "n_estimators":int(request.POST.get("random_forest_n_estimators")),
                "criterion":request.POST.get("random_forest_criterion"),
                "max_depth":request.POST.get("random_forest_max_depth"),
                "min_samples_split":int(request.POST.get("random_forest_min_samples_split")),
                "min_samples_leaf":int(request.POST.get("random_forest_min_samples_leaf")),
                "max_features":request.POST.get("random_forest_max_features")
            }

            if(hyperparameters["max_depth"]=="None"):
                hyperparameters["max_depth"]=None
            else:
                hyperparameters["max_depth"]=int(hyperparameters["max_depth"])

            if(hyperparameters["max_features"].isdigit() is True):
                hyperparameters["max_features"]=int(hyperparameters["max_features"])
        elif method=="svr":
            hyperparameters={
                "kernel":request.POST.get("svr_kernel"),
                "degree":int(request.POST.get("svr_degree")),
                "gamma":request.POST.get("svr_gamma"),
                "tol":float(request.POST.get("svr_tol")),
                "C":float(request.POST.get("svr_C")),
                "epsilon":float(request.POST.get("svr_epsilon")),
                "max_iter":int(request.POST.get("svr_max_iter"))
            }
            
            if (hyperparameters["gamma"].isdigit() is True):
                hyperparameters["gamma"]=int(hyperparameters["gamma"])
            
        elif method=="pwr":
            hyperparameters={
                "method":request.POST.get("pwr_method"),
                "n_regions":int(request.POST.get("pwr_n_regions"))
            }
            print(hyperparameters)
        
        if approach=="full_training":
            approach_parameters={None:None}
        elif approach=="train_test":
            approach_parameters={
                "train_size":float(request.POST.get("train_test_train_size")),
                "random_state":int(request.POST.get("train_test_seed")),
                "shuffle":True
            }
        elif approach=="kfold":
            approach_parameters={
                "n_splits":int(request.POST.get("kfold_n_splits")),
                "random_state":int(request.POST.get("kfold_seed")),
                "shuffle":True
            }
        elif approach=="repeated_kfold":
            approach_parameters={
                "n_splits":int(request.POST.get("kfold_n_splits")),
                "random_state":int(request.POST.get("kfold_seed")),
                "n_repeats":int(request.POST.get("repeated_kfold_n_repeats"))                
            }

        my_context={
            "method":request.POST.get('regression_method'),
            "approach":request.POST.get('regression_approach'),
            "dataset":uploaded_file,   
            "hyperparameters":hyperparameters,
            "approach_parameters":approach_parameters
        }                    
                
        model=BackendRegression(my_context)
        model.backend()

        my_context.update({"model":model.model})        
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
        
def visuals(request):
    pass