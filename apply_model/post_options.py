class PostOptions():

    def __init__(self,approach):            
        self.approach=approach

    def linear_regression(self,request):
        hyperparameters={
            "fit_intercept":request.POST.get("linear_fit_intercept")
        }            
        if self.approach == "grid_search":
            hyperparameters["fit_intercept"]=[hyperparameters["fit_intercept"]]
            counter=2
            flag=True
            while flag:                                        
                if(request.POST.get("linear_fit_intercept_"+str(counter))\
                     is not None):
                    hyperparameters['fit_intercept'].append(\
                        request.POST.get("linear_fit_intercept_"+str(counter)))
                    counter+=1
                else:
                    flag = False        
        return hyperparameters

    def decision_tree(self,request):
        hyperparameters={
                "criterion":request.POST.get("decision_tree_criterion"),
                "splitter":request.POST.get("decision_tree_splitter"),
                "max_depth":request.POST.get("decision_tree_max_depth"),
                "min_samples_split":int(request.POST.get(\
                    "decision_tree_min_samples_split")),
                "min_samples_leaf":int(request.POST.get(\
                    "decision_tree_min_samples_leaf")),
                "max_leaf_nodes":request.POST.get(\
                    "decision_tree_max_leaf_nodes"),
                "max_features":request.POST.get("decision_tree_max_features")
            }

        if(hyperparameters["max_depth"]=="None"):
            hyperparameters["max_depth"]=None
        else:
            hyperparameters["max_depth"]=int(hyperparameters["max_depth"])
        if(hyperparameters["max_leaf_nodes"]=="None"):
            hyperparameters["max_leaf_nodes"]=None
        else:
            hyperparameters["max_leaf_nodes"]=int(hyperparameters["max_leaf_nodes"])            

        if(hyperparameters["max_features"].isdigit() is True):
            hyperparameters["max_features"]=int(hyperparameters["max_features"])
        
        if self.approach == "grid_search":
            hyperparameters["criterion"] = [hyperparameters["criterion"]]
            hyperparameters["splitter"] = [hyperparameters["splitter"]]                
            hyperparameters["max_depth"] = [hyperparameters["max_depth"]]
            hyperparameters["min_samples_split"] = [hyperparameters[\
                "min_samples_split"]]
            hyperparameters["min_samples_leaf"] = [hyperparameters[\
                "min_samples_leaf"]]
            hyperparameters["max_leaf_nodes"] = [hyperparameters[\
                "max_leaf_nodes"]]                
            hyperparameters["max_features"] = [hyperparameters[\
                "max_features"]]

            counter_criterion = 2
            counter_splitter = 2
            counter_max_depth = 2
            counter_min_samples_split = 2
            counter_min_samples_leaf = 2
            counter_max_leaf_nodes = 2
            counter_max_features = 2
            flag_criterion=True
            flag_splitter=True
            flag_max_depth=True
            flag_min_samples_split=True
            flag_min_samples_split=True
            flag_max_leaf_nodes=True
            flag_max_features=True
            flag=True
            while flag:
                # Best Criterion Hyperparameter
                if(request.POST.get("decision_tree_criterion_"+\
                    str(counter_criterion)) is not None):                        
                    hyperparameters['criterion'].append(request.POST.get(\
                        "decision_tree_criterion_"+str(counter_criterion)))
                    counter_criterion += 1
                else:
                    flag_criterion = False
                # Best split hyperparameter
                if(request.POST.get("decision_tree_splitter_"+\
                    str(counter_splitter)) is not None):                        
                    hyperparameters['splitter'].append(request.POST.get(\
                        "decision_tree_splitter_"+str(counter_splitter)))
                    counter_splitter += 1
                else:
                    flag_splitter = False
                # Max Depth hyperparameter
                if(request.POST.get("decision_tree_max_depth_"+\
                    str(counter_max_depth)) is not None):
                    if(request.POST.get("decision_tree_max_depth_"+\
                        str(counter_max_depth))=="None"):                            
                        hyperparameters['max_depth'].append(None)                        
                    else:
                        hyperparameters['max_depth'].append(int(\
                            request.POST.get("decision_tree_max_depth_"\
                                +str(counter_max_depth))))                        

                    counter_max_depth += 1
                else:
                    flag_max_depth = False
                
                # Min Samples Split hyperparameter
                if(request.POST.get("decision_tree_min_samples_split_"+\
                    str(counter_min_samples_split)) is not None):
                    hyperparameters['min_samples_split'].append(int(\
                        request.POST.get("decision_tree_min_samples_split_"+\
                            str(counter_min_samples_split))))

                    counter_min_samples_split += 1
                else:
                    flag_min_samples_split = False
                
                if flag_criterion == False and flag_splitter == False \
                    and flag_max_depth == False and flag_min_samples_split == False:
                    flag = False                    
                
                # Min Samples Leaf hyperparameter
                if(request.POST.get("decision_tree_min_samples_leaf_"+\
                    str(counter_min_samples_leaf)) is not None):
                    hyperparameters['min_samples_leaf'].append(int(\
                        request.POST.get("decision_tree_min_samples_leaf_"+\
                            str(counter_min_samples_leaf))))

                    counter_min_samples_leaf += 1
                else:
                    flag_min_samples_leaf = False

                # Max Leaf Nodes hyperparameter
                if(request.POST.get("decision_tree_max_leaf_nodes_"+\
                    str(counter_max_leaf_nodes)) is not None):
                    if(request.POST.get("decision_tree_max_depth_"+\
                        str(counter_max_leaf_nodes))=="None"):                            
                        hyperparameters['max_leaf_nodes'].append(None)                        
                    else:
                        hyperparameters['max_leaf_nodes'].append(int(\
                            request.POST.get("decision_tree_max_leaf_nodes_"+\
                                str(counter_max_leaf_nodes))))                        

                    counter_max_leaf_nodes += 1
                else:
                    flag_max_leaf_nodes = False

                # Max Features hyperparameter
                if(request.POST.get("decision_tree_max_features_"+\
                    str(counter_max_features)) is not None): 
                    if(request.POST.get("decision_tree_max_features_"+\
                        str(counter_max_features)).isdigit() is True): 
                        hyperparameters['max_features'].append(int(\
                            request.POST.get("decision_tree_max_features_"+\
                                str(counter_max_features))))                        
                    else:
                        hyperparameters['max_features'].append(request.POST.get(\
                            "decision_tree_max_features_"+\
                                str(counter_max_features)))

                    counter_max_features += 1
                else:
                    flag_max_features = False                    

                if flag_criterion == False and flag_splitter == False\
                        and flag_max_depth == False and flag_min_samples_split == False\
                            and flag_min_samples_leaf == False and flag_max_leaf_nodes==False\
                                and flag_max_features == False:
                    flag = False        
        
        return hyperparameters

    def mlp(self,request):
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

        if self.approach=="grid_search":
            hyperparameters['hidden_layer_sizes']=[hyperparameters['hidden_layer_sizes']]
            hyperparameters['activation']=[hyperparameters['activation']]
            hyperparameters['solver']=[hyperparameters['solver']]
            hyperparameters['alpha']=[hyperparameters['alpha']]
            hyperparameters['max_iter']=[hyperparameters['max_iter']]
            hyperparameters['tol']=[hyperparameters['tol']]
            
            counter_hidden_layer_sizes = 2
            counter_activation = 2
            counter_solver = 2
            counter_alpha = 2
            counter_max_iter = 2
            counter_tol = 2
            flag_hidden_layer_sizes = True
            flag_activation = True
            flag_solver = True
            flag_alpha = True
            flag_max_iter = True
            flag_tol = True
            flag=True

            while flag:

                # Hidden Layer Sizes
                if(request.POST.get("mlp_hidden_layer_sizes_"+\
                    str(counter_hidden_layer_sizes)) is not None):
                    hidden_layers = request.POST.get("mlp_hidden_layer_sizes_"+\
                        str(counter_hidden_layer_sizes))
                    hidden_layers=hidden_layers.split(",")
                    hidden_layers=tuple(hidden_layers)
                    hidden_layers=tuple(int(x) if x.isdigit() else x for x in\
                        hidden_layers if x)            
                    
                    hyperparameters['hidden_layer_sizes'].append(hidden_layers)
                
                # Activation Hyperparameter
                if(request.POST.get("mlp_activation_"+str(counter_activation))\
                     is not None):
                    hyperparameters['activation'].append(request.POST.get(\
                        "mlp_activation_"+str(counter_activation)))
                    counter_activation += 1
                else:
                    flag_activation = False

                # Solver Hyperparameter
                if(request.POST.get("mlp_solver_"+\
                    str(counter_solver)) is not None):
                    hyperparameters['solver'].append(\
                        request.POST.get("mlp_solver_"+\
                            str(counter_solver)))
                    counter_solver += 1
                else:
                    flag_solver = False

                # Alpha Hyperparameter
                if(request.POST.get("mlp_alpha_"+str(counter_alpha))\
                     is not None):
                    hyperparameters['alpha'].append(float(\
                        request.POST.get("mlp_alpha_"+\
                            str(counter_alpha))))
                    counter_alpha += 1
                else:
                    flag_alpha = False

                # Max Iter Hyperparameter
                if(request.POST.get("mlp_max_iter_"+str(counter_max_iter))\
                     is not None):
                    hyperparameters['max_iter'].append(int(\
                        request.POST.get("mlp_max_iter_"+\
                            str(counter_max_iter))))
                    counter_max_iter += 1
                else:
                    flag_max_iter = False
                
                # Tol Hyperparameter
                if(request.POST.get("mlp_tol_"+str(counter_tol))\
                     is not None):
                    hyperparameters['tol'].append(float(\
                        request.POST.get("mlp_tol_"+\
                            str(counter_tol))))
                    counter_tol += 1
                else:
                    flag_tol = False
                
                if flag_hidden_layer_sizes == False and flag_activation == False \
                    and flag_solver == False and flag_alpha == False and\
                         flag_max_iter == False and flag_tol == False:
                    flag = False        

        return hyperparameters
    
    def random_forest(self,request):
        
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
        
        if self.approach == "grid_search":
            hyperparameters["n_estimators"] = [hyperparameters["n_estimators"]]
            hyperparameters["criterion"] = [hyperparameters["criterion"]]                    
            hyperparameters["max_depth"] = [hyperparameters["max_depth"]]
            hyperparameters["min_samples_split"] = [hyperparameters["min_samples_split"]]
            hyperparameters["min_samples_leaf"] = [hyperparameters["min_samples_leaf"]]
            hyperparameters["max_features"] = [hyperparameters["max_features"]]

            counter_n_estimators = 2
            counter_criterion = 2                
            counter_max_depth = 2
            counter_min_samples_split = 2
            counter_min_samples_leaf = 2
            counter_max_features = 2
            flag_n_estimators=True
            flag_criterion=True                
            flag_max_depth=True
            flag_min_samples_split=True
            flag_min_samples_split=True
            flag_max_features=True
            flag=True
            while flag:
                # Best Criterion Hyperparameter
                if(request.POST.get("random_forest_ctiretion_"+\
                    str(counter_criterion)) is not None):                        
                    hyperparameters['criterion'].append(request.POST.get(\
                        "random_forest_ctiretion_"+str(counter_criterion)))
                    counter_criterion += 1
                else:
                    flag_criterion = False
                # Best n_estimators hyperparameter
                if(request.POST.get("random_forest_n_estimators_"+\
                    str(counter_n_estimators)) is not None):                        
                    hyperparameters['n_estimators'].append(int(\
                        request.POST.get("random_forest_n_estimators_"+\
                            str(counter_n_estimators))))
                    counter_n_estimators += 1
                else:
                    flag_n_estimators = False
                # Max Depth hyperparameter
                if(request.POST.get("random_forest_max_depth_"+\
                    str(counter_max_depth)) is not None):
                    if(request.POST.get("random_forest_max_depth_"+\
                        str(counter_max_depth))=="None"):                            
                        hyperparameters['max_depth'].append(None)                        
                    else:
                        hyperparameters['max_depth'].append(int(\
                            request.POST.get("random_forest_max_depth_"+\
                                str(counter_max_depth))))                        

                    counter_max_depth += 1
                else:
                    flag_max_depth = False
                
                # Min Samples Split hyperparameter
                if(request.POST.get("random_forest_min_samples_split_"+\
                    str(counter_min_samples_split)) is not None):
                    hyperparameters['min_samples_split'].append(int(\
                        request.POST.get("random_forest_min_samples_split_"+\
                            str(counter_min_samples_split))))

                    counter_min_samples_split += 1
                else:
                    flag_min_samples_split = False                                                        
                
                # Min Samples Leaf hyperparameter
                if(request.POST.get("random_forest_min_samples_leaf_"+\
                    str(counter_min_samples_leaf)) is not None):
                    hyperparameters['min_samples_leaf'].append(int(\
                        request.POST.get("random_forest_min_samples_leaf_"+\
                            str(counter_min_samples_leaf))))

                    counter_min_samples_leaf += 1
                else:
                    flag_min_samples_leaf = False

                # Max Features hyperparameter
                if(request.POST.get("random_forest_max_features_"+\
                    str(counter_max_features)) is not None): 
                    if(request.POST.get("random_forest_max_features_"+\
                        str(counter_max_features)).isdigit() is True): 
                        hyperparameters['max_features'].append(int(\
                            request.POST.get("random_forest_max_features_"+\
                                str(counter_max_features))))                        
                    else:
                        hyperparameters['max_features'].append(request.POST.get(\
                            "random_forest_max_features_"+\
                                str(counter_max_features)))

                    counter_max_features += 1
                else:
                    flag_max_features = False

                if flag_criterion == False and flag_n_estimators == False\
                        and flag_max_depth == False and flag_min_samples_split == False\
                            and flag_min_samples_leaf == False and flag_max_features==False:
                    flag = False          

        return hyperparameters
    
    def svr(self,request):

        hyperparameters={
            "kernel":request.POST.get("svr_kernel"),
            "degree":int(request.POST.get("svr_degree")),
            "gamma":request.POST.get("svr_gamma"),
            "tol":float(request.POST.get("svr_tol")),
            "C":float(request.POST.get("svr_c")),
            "epsilon":float(request.POST.get("svr_epsilon")),
            "max_iter":int(request.POST.get("svr_max_iter"))
        }

        if (hyperparameters["gamma"].isdigit() is True):
            hyperparameters["gamma"]=int(hyperparameters["gamma"])

        if self.approach == "grid_search":
            hyperparameters['kernel']=[hyperparameters['kernel']]
            hyperparameters['degree']=[hyperparameters['degree']]
            hyperparameters['gamma']=[hyperparameters['gamma']]
            hyperparameters['tol']=[hyperparameters['tol']]
            hyperparameters['C']=[hyperparameters['C']]
            hyperparameters['epsilon']=[hyperparameters['epsilon']]
            hyperparameters['max_iter']=[hyperparameters['max_iter']]

            counter_kernel = 2
            counter_degree = 2
            counter_gamma = 2
            counter_tol = 2
            counter_c = 2
            counter_epsilon = 2
            counter_max_iter = 2
            flag_kernel = True
            flag_degree = True
            flag_gamma = True
            flag_tol = True
            flag_c = True
            flag_epsilon = True
            flag_max_iter = True        
            flag = True        

            while flag:
                # Best Kenrel Hyperparameter
                if(request.POST.get("svr_kernel_"+str(counter_kernel))\
                     is not None):                        
                    hyperparameters['kernel'].append(request.POST.get(\
                        "svr_kernel_"+str(counter_kernel)))
                    counter_kernel += 1
                else:
                    flag_kernel = False                
                
                # Best Degree Hyperparameter
                if(request.POST.get("svr_degree_"+str(counter_degree))\
                     is not None):                        
                    hyperparameters['degree'].append(int(\
                        request.POST.get("svr_degree_"+str(counter_degree))))
                    counter_degree += 1
                else:
                    flag_degree = False                                    

                # Best Gamma Hyperparameter
                if(request.POST.get("svr_gamma_"+str(counter_gamma))\
                     is not None):   
                    if(request.POST.get("svr_gamma_"+\
                        str(counter_gamma)).isdigit() is True):
                        hyperparameters['gamma'].append(int(\
                            request.POST.get("svr_gamma_"+str(counter_gamma))))
                    else:
                        hyperparameters['gamma'].append(request.POST.get(\
                            "svr_gamma_"+str(counter_gamma)))
                    counter_gamma += 1
                else:
                    flag_gamma = False                                    

                # Best Tol Hyperparameter
                if(request.POST.get("svr_tol_"+str(counter_tol)) is not None):                        
                    hyperparameters['tol'].append(float(request.POST.get(\
                        "svr_tol_"+str(counter_tol))))
                    counter_tol += 1
                else:
                    flag_tol = False           

                # Best C Hyperparameter
                if(request.POST.get("svr_c_"+str(counter_c)) is not None):                        
                    hyperparameters['C'].append(float(request.POST.get(\
                        "svr_c_"+str(counter_c))))
                    counter_c += 1
                else:
                    flag_c = False

                # Best Epsilon Hyperparameter
                if(request.POST.get("svr_epsilon_"+str(counter_epsilon))\
                     is not None):                        
                    hyperparameters['epsilon'].append(float(request.POST.get(\
                        "svr_epsilon_"+str(counter_epsilon))))
                    counter_epsilon += 1
                else:
                    flag_epsilon = False

                # Best Max Iter Hyperparameter
                if(request.POST.get("svr_max_iter_"+str(counter_max_iter))\
                     is not None):                        
                    hyperparameters['max_iter'].append(int(request.POST.get(\
                        "svr_max_iter_"+str(counter_max_iter))))
                    counter_max_iter += 1
                else:
                    flag_max_iter = False

                if flag_kernel == False and flag_degree == False and flag_gamma == False \
                    and flag_tol == False and flag_c == False and flag_epsilon == False \
                        and flag_max_iter == False:
                    flag=False        

        return hyperparameters

    def regression_approach(self,request):

        if self.approach=="full_training":
            approach_parameters={None:None}
        elif self.approach=="train_test" or self.approach == "grid_search":
            approach_parameters={
                "train_size":float(request.POST.get("train_test_train_size")),
                "random_state":int(request.POST.get("train_test_seed")),
                "shuffle":True
            }
        elif self.approach=="kfold":
            approach_parameters={
                "n_splits":int(request.POST.get("kfold_n_splits")),
                "random_state":int(request.POST.get("kfold_seed")),
                "shuffle":True
            }
        elif self.approach=="repeated_kfold":
            approach_parameters={
                "n_splits":int(request.POST.get("kfold_n_splits")),
                "random_state":int(request.POST.get("kfold_seed")),
                "n_repeats":int(request.POST.get("repeated_kfold_n_repeats")),                
            }        
        
        return approach_parameters