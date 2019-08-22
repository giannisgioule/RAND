from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import gdxpds as gd
import numpy as np
import pandas as pd
import os

class PiecewiseRegression(BaseEstimator,RegressorMixin):
    
    '''
    A piecewise linear regressor

    This regressor fits piecewise linear expressions to data, based on optimisation models. Each region
    contains linear expressions based on the input variables.

    Parameters
    ----------

    method: str (default='PROA')    
        The optimisation method to be used

        - If 'OPLRA', then the selected method is the Optimal Piecewise Linear Regression Analysis
        - If 'PROA', then the selected method is the Piecewise Regression with Optimised Akaike Information Criterion.
        - If 'PRIA', then the selected method is the Piecewise Regression with Iterative Akaike Information Criterion.
        - If 'PROB', then the selected method is the Piecewise Regression with Optimised Bayesian Information Criterion.
        - If 'PRIB', then the selected method is the Piecewise Regression with Iterative Bayesian Information Criterion.

    n_regions: int (default=2)
        The maximum number of allowable regions. This parameter applies only to the 'PROA' and 'PROB' methods. It can take

    Attributes
    ----------

    model_: dict
        A dictionary that contains the regression coefficients for each region

    '''


    def __init__(self,method="PROA",n_regions=2):
        self.method=method
        self.n_regions=n_regions
    
    def fit(self,X,y):

        
        def check_regions(n_regions):
            
            assert 2<=n_regions<=8,\
                "n_regions out of range. Please select a number between 2 and 8"            
            assert isinstance(n_regions,int),"n_regions must be int"
            

        def check_method(method):
            assert (method=="PROA" or method=="PRIA" or method=="PRIB" \
                or method=="PROB" or method=="OPLRA"),\
                    "selected method is incorrect"
            
        
        check_regions(self.n_regions)
        check_method(self.method)

        class Regions():

            def __init__(self,coefficients,intercept):

                self.coefficients=coefficients
                self.intercept=intercept        
        
        # This function generates the .gdx file with all the input data
        def generate_gdx(X,method,n_regions):                

            if method=='OPLRA':
                method=1
            elif method=='PRIA':
                method=2
            elif method=='PROA':
                method=3
            elif method=='PRIB':
                method=4
            elif method=='PROB':
                method=5

            # The number of predictors in the set
            number_predictors=X.shape[1]-1 
            
            # The names of the attributes        
            attribute_names=X.columns[0:number_predictors] 
            
            # Create the set of samples
            s_val=list(range(1, X.shape[0]+1))
            s_val=list(map(str,s_val))
            s_val=["s"+x for x in s_val]

            # Create the list parameter for the input values
            s_val=s_val*number_predictors
            s_val=pd.DataFrame(s_val)

            values=X.iloc[:,0:number_predictors]
            values=values.unstack()
            values=pd.DataFrame(list(values))

            all_attr=pd.DataFrame(np.repeat(attribute_names[0],X.shape[0]))
            for i in range(1,number_predictors):
                single_attr=pd.DataFrame(np.repeat(attribute_names[i],\
                    X.shape[0]))
                all_attr=pd.concat([all_attr,single_attr],axis=0)

            all_attr.index=s_val.index
            final_values=pd.concat([s_val,all_attr],axis=1)
            final_values=pd.concat([final_values,values],axis=1)
            del(values)       
                
            # The output variable
            output=X.iloc[:,number_predictors] 

            # Change the row names of the set
            output.index=s_val[0:X.shape[0]].index 

            
            #---Define the .gdx file and assign values to all the elements----            
            gdx_file="input.gdx"             
            input_data=gd.gdx.GdxFile() 
            
            # First define the set of the samples
            input_data.append(gd.gdx.GdxSymbol("s",gd.gdx.GamsDataType.Set,\
                dims=1,description="set of samples in the set"))
            # Then define the set of the predictor variables
            input_data.append(gd.gdx.GdxSymbol("m",gd.gdx.GamsDataType.Set,\
                dims=1,description="The input variables of the X"))
            # Then define the input of the dataset
            input_data.append(gd.gdx.GdxSymbol("A",\
                gd.gdx.GamsDataType.Parameter,dims=2,\
                    description="The values of the samples"))
            input_data.append(gd.gdx.GdxSymbol("selected_method",\
                gd.gdx.GamsDataType.Parameter,dims=0,\
                    description="The selected method"))            
            input_data.append(gd.gdx.GdxSymbol("Y",\
                gd.gdx.GamsDataType.Parameter,dims=1,\
                    description="The output values"))            
            input_data.append(gd.gdx.GdxSymbol("selected_regions",\
                gd.gdx.GamsDataType.Parameter,dims=0,\
                    description="The number of maximum regions"))                        
            # Assign values
            input_data[0].dataframe=s_val[0:X.shape[0]] 
            input_data[1].dataframe=attribute_names 
            input_data[2].dataframe=final_values
            input_data[3].dataframe=pd.DataFrame([method])
            input_data[4].dataframe=pd.concat([s_val[0:X.shape[0]],output],\
                axis=1)
            input_data[5].dataframe=pd.DataFrame([n_regions])   
            #-----------------------------------------------------------------

            input_data.write(gdx_file)
            
            return(input_data)            

        # This function calls GAMS and returns the results of the optimisation
        def optimisation_stage(X,method,n_regions):
        
            input_data=generate_gdx(X,method,n_regions)  
            
            # Call GAMS for the optimisation
            # First, the file to identify the best partitioning feature
            os.system("gams identify_partitioning_variable.gms o=nul")
            
            # Save the partitioning results
            part_results=gd.to_dataframes('best_part_variable.gdx')        

            part_error=min(part_results['error_all_iterations']['Value'])

            part_variable=part_results['error_all_iterations'][part_results\
                ['error_all_iterations']['Value']==part_error]
            part_variable=part_variable['loop_part_variable']
            part_variable=part_variable.values.tolist()        

            # Append the existing .gdx file 
            input_data.append(gd.gdx.GdxSymbol("part_variable",\
                gd.gdx.GamsDataType.Set,dims=1,\
                    description="The partitioning variable"))
            input_data.append(gd.gdx.GdxSymbol("max_error",\
                gd.gdx.GamsDataType.Parameter,dims=0,\
                    description="The error of the partitioning"))
            
            input_data[6].dataframe=pd.DataFrame(part_variable)
            input_data[7].dataframe=pd.DataFrame([part_error])
            
            input_data.write('input.gdx')
            
            # Call GAMS to fit piecewise regression models
            os.system('gams optimal_piecewise_regression.gms o=nul')

            return(part_variable)            
                
        v_names=False
        if(isinstance(X,pd.DataFrame)):
            variable_names=X.columns
            v_names=True
                
        o_name=False
        if(isinstance(y,pd.DataFrame)):
            output_name=y.columns
            o_name=True
        
        X,y=check_X_y(X,y)
        X=pd.DataFrame(X)        
        y=pd.DataFrame(y)                
        if(v_names is True):
            X.columns=variable_names        
        else:
            X.columns=[("x"+str(i)) for i in range(len(X.columns))]
        if(o_name is True):
            y.columns=output_name
        
        # Convert to pandas for the gdxpds library
        X=pd.concat([X,y],axis=1)

        # Fit a model
        part_variable=optimisation_stage(X,self.method,self.n_regions)                

        regression_results=gd.to_dataframes("regression_results.gdx")
        regression_results['coefficients'].columns=\
            ['variable','region','Value']

        break_values=regression_results['break_point']\
            ['Value'].values.tolist()

        active_regions=len(break_values)+1

        # Create a dictionary with the regression coefficients and the intercepts
        # for every single region. The last pair of key-value in this dictionary
        # is a list containing the breaking point(s)
        piecewise_model={}

        for identify_region in range(active_regions):
            coefficients=regression_results['coefficients']\
                [regression_results['coefficients']['region']=='r'+str(identify_region+1)]\
                    ['Value'].values.tolist()
            intercept=regression_results['intercept']\
                [regression_results['intercept']['*']=='r'+str(identify_region+1)]\
                    ['Value'].values.tolist()
            
            piecewise_model.update({'r'+str(identify_region+1):\
                Regions(coefficients,intercept)})

        piecewise_model.update({part_variable[0]:break_values})

        os.remove('input.gdx')
        os.remove('best_part_variable.gdx')
        os.remove('regression_results.gdx')
        os.remove('cplex.opt')
        
        self.model_=piecewise_model
        self.is_fitted_=True
        
        if(v_names is True):        
            self.names_=variable_names
        else:
            self.names_=False
        return self
        
    def predict(self,X):                

        # This function assigns samples to the correct region
        def identify_region(break_points,X):
            region='r'+str(len(break_points)+1)
            for correct_region in range(len(break_points)):
                if X<=break_points[correct_region]:
                    region='r'+str(correct_region+1)
                    break        
            
            return region
        X=check_array(X)
        check_is_fitted(self,'is_fitted_')        
        
        # Define model and name 
        model=getattr(self,"model_")
        names=getattr(self,"names_")
        
        #========================================================================
        #                   Assign the correct variable names
        #========================================================================
        X=pd.DataFrame(X)
        if(np.all(names is not False)):
            X.columns=names        
        break_variable=list(model.keys())[len(list(model.keys()))-1]
        break_points=model[break_variable]

        if(np.all(names is False)):
            X.columns=[("x"+str(i)) for i in range(len(X.columns))]
        #========================================================================

        index_partition=X.columns.get_loc(break_variable)
        X=X.values.tolist()
        y_pred=[]
        for testing_samples in range(len(X)):
            assigned_region=identify_region(break_points,\
                X[testing_samples][index_partition])
            y_pred.append(sum([a*b for a,b in zip(X[testing_samples],\
                model[assigned_region].coefficients)])\
                    +model[assigned_region].intercept[0])

        return y_pred