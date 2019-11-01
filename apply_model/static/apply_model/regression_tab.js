/*-----------------------------------------------------------------------------
This file contains the necessary function that are required by the 
regression.html file. The following functions are responsible for chaning 
the UI based on different regression methods and approaches
-----------------------------------------------------------------------------*/


function changeHyperparameters() {
    
  /* This function changes the 'Hyperparameter selection' section of the 
  regression tab, according to which regression method the user selected */

  // Declaring all the necessary variables
  var linearBlock, decisionTreeBlock, mlpBlock, randomForestBlock, svrBlock,
   selected

  //-------------------------------------------------------------------------
  /* Defining the values of the variables. All of them are elements inside
  the regression.html file */

  // The hyperparameters of linear regression
  linearBlock = document.getElementById("linear_regression_parameters")
  // The hyperparameters of decision trees
  decisionTreeBlock = document.getElementById("decision_tree_parameters")
  // The hyperparameters of mlp (multi-layer perceptron)
  mlpBlock = document.getElementById("mlp_parameters")
  // The hyperparameters of random forest
  randomForestBlock = document.getElementById("random_forest_parameters")
  // The hyperparameters of svr (support-vector regression)
  svrBlock = document.getElementById("svr_parameters")
  // The user selected regression approach
  selected = document.getElementById("select_method")
  //-------------------------------------------------------------------------

  //-------------------------------------------------------------------------
  /* Activate the correct block so that the user can select the desired
  hyperparameter values */

  if(selected.value == "linear_regression") {
    linearBlock.style.display = "block";
  }else{
    linearBlock.style.display = "none";
  }

  if(selected.value == "decision_tree") {
    decisionTreeBlock.style.display = "block";
  }else{
    decisionTreeBlock.style.display = "none";
  }

  if(selected.value == "mlp") {
    mlpBlock.style.display = "block";
  }else{
    mlpBlock.style.display = "none";
  }

  if(selected.value == "random_forest") {
    randomForestBlock.style.display = "block";
  }else{
    randomForestBlock.style.display = "none";
  }

  if(selected.value == "svr") {
    svrBlock.style.display = "block";
  }else{
    svrBlock.style.display = "none";
  }
  //-------------------------------------------------------------------------
}

function approachParameters(){  

  // Declaring all the necessary variables          
  var kfoldParameters, repeatedKfoldParameters, trainTestParameters, approach
  var addOptions, selected, extraBlock
  //-------------------------------------------------------------------------
  /* Defining the values of the variables. All of them are elements inside
  the regression.html file */  
  kfoldParameters = document.getElementById("kfold_approach_parameters")
  repeatedKfoldParameters = document.getElementById("repeated_kfold_approach_parameters")
  trainTestParameters = document.getElementById("train_test_approach_parameters")   
  selected = document.getElementById("select_method")  
  //-------------------------------------------------------------------------
  addOptions = document.getElementsByClassName("button_add")
  removeOptions = document.getElementsByClassName("button_remove")
  extraBlock = document.getElementsByClassName("grid_search_block")

  approach = document.getElementById("select_approach")
  
  if(approach.value == "grid_search") {

    for(i = 0; i <addOptions.length; i++) {
      addOptions[i].style.display = "block";
      removeOptions[i].style.display = "block";
    }
  }else{
       
    for(i = 0; i <addOptions.length; i++) {
      addOptions[i].style.display = "none";
      removeOptions[i].style.display = "none";
    }
  }

  //-------------------------------------------------------------------------
  /* Activate the correct block so that the user can select the desired
  approach values */

  if(approach.value == "kfold") {  
    kfoldParameters.style.display = "";
    repeatedKfoldParameters.style.display = "none";
    trainTestParameters.style.display = "none";
    for(i = 0; i < extraBlock.length; i++) {
      extraBlock[i].style.display = "none";
    }
  }else if(approach.value == "repeated_kfold") { 
    kfoldParameters.style.display = "";
    repeatedKfoldParameters.style.display = "";
    trainTestParameters.style.display = "none";
    for(i = 0; i < extraBlock.length; i++) {
      extraBlock[i].style.display = "none";
    }
  }else if(approach.value == "train_test") {
    kfoldParameters.style.display = "none";
    repeatedKfoldParameters.style.display = "none";
    trainTestParameters.style.display = "";
    for(i = 0; i < extraBlock.length; i++) {
      extraBlock[i].style.display = "none";
    }
  }else if(approach.value == "grid_search") {
    kfoldParameters.style.display = "";
    repeatedKfoldParameters.style.display = "none";
    trainTestParameters.style.display = "";
    for(i = 0; i < extraBlock.length; i++) {
      extraBlock[i].style.display = "";
    }
  }else{    
    kfoldParameters.style.display = "none";
    repeatedKfoldParameters.style.display = "none";
    trainTestParameters.style.display = "none";
    for(i = 0; i < extraBlock.length; i++) {
      extraBlock[i].style.display = "none";
    }
  }
  //-------------------------------------------------------------------------

}


/* This part of the script is responsible for adding and removing elements
for Grid Search*/

//-----------------------------------------------------------------------------
// Linear Regression Parameters
var activityNumberLinearFitIntercept = 2;  
// Decision Tree Parameters
var activityNumberDecisionCriterion = 2;  
var activityNumberDecisionSplitter = 2;
var activityNumberDecisionMaxDepth = 2;
var activityNumberDecisionMinSamplesSplit = 2;
var activityNumberDecisionMinSamplesLeaf = 2;
var activityNumberDecisionMaxLeafNodes = 2;
var activityNumberDecisionMaxFeatures = 2;
// MLP Parameters
var activityNumberMlpHiddenLayerSizes = 2;
var activityNumberMlpActivation = 2;
var activityNumberMlpSolver = 2;
var activityNumberMlpAlpha = 2;
var activityNumberMlpMaxIter = 2;
var activityNumberMlpTol = 2;
// Random Forest Parameters
var activityNumberRandomForestEstimators = 2;
var activityNumberRandomForestCriterion = 2;
var activityNumberRandomForestMaxDepth = 2;
var activityNumberRandomForestMinSamplesSplit = 2;
var activityNumberRandomForestMinSamplesLeaf = 2;
var activityNumberRandomForestMaxFeatures = 2;
// Support Vector Machines Parameters
var activityNumberSvrKernel = 2;
var activityNumberSvrDegree = 2;
var activityNumberSvrGamma = 2;
var activityNumberSvrTol = 2;
var activityNumberSvrC = 2;
var activityNumberSvrEpsilon = 2;
var activityNumberSvrMaxIter = 2;

//-----------------------------------------------------------------------------
//                        ADD ELEMENTS TO GRID SEARCH
//-----------------------------------------------------------------------------
function addElement(clickedId) {    
  
  var selected

  selected = document.getElementById("select_method")
  
  // Linear regression
  if(selected.value == "linear_regression") {
    if(window.activityNumberLinearFitIntercept <= 2) {
      if(clickedId == "linear_fit_intercept_button_add") {
        var trackElementLinearFitIntercept = document.getElementById("linear_fit_intercept_grid_search")
        var toBeCloned = document.getElementById("linear_fit_intercept")
        var clone
      
        clone = toBeCloned.cloneNode(true);
        clone.id = "linear_fit_intercept_" + window.activityNumberLinearFitIntercept
        clone.name = "linear_fit_intercept_" + window.activityNumberLinearFitIntercept
      
        trackElementLinearFitIntercept.appendChild(clone)
        window.activityNumberLinearFitIntercept += 1
      }
    }
  // Decision Trees
  }else if(selected.value == "decision_tree") {
    if(window.activityNumberDecisionCriterion <= 3) {
      if(clickedId == "decision_tree_criterion_button_add") {
        var trackElementDecisionCriterion = document.getElementById("decision_tree_criterion_grid_search")
        var toBeCloned = document.getElementById("decision_tree_criterion")
        var clone
      
        clone = toBeCloned.cloneNode(true);
        clone.id = "decision_tree_criterion_" + window.activityNumberDecisionCriterion
        clone.name = "decision_tree_criterion_" + window.activityNumberDecisionCriterion
      
        trackElementDecisionCriterion.appendChild(clone)
        window.activityNumberDecisionCriterion += 1
      }
    }
    if(window.activityNumberDecisionSplitter <= 2) {
      if(clickedId == "decision_tree_splitter_button_add") {
        var trackElementDecisionSplitter = document.getElementById("decision_tree_splitter_grid_search")
        var toBeCloned = document.getElementById("decision_tree_splitter")
        var clone
      
        clone = toBeCloned.cloneNode(true);
        clone.id = "decision_tree_splitter_" + window.activityNumberDecisionSplitter
        clone.name = "decision_tree_splitter_" + window.activityNumberDecisionSplitter
      
        trackElementDecisionSplitter.appendChild(clone)
        window.activityNumberDecisionSplitter += 1
      }    
    }
    if(clickedId == "decision_tree_max_depth_button_add") {
      var trackElementDecisionMaxDepth = document.getElementById("decision_tree_max_depth_grid_search")
      var toBeCloned = document.getElementById("decision_tree_max_depth")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "decision_tree_max_depth_" + window.activityNumberDecisionMaxDepth
      clone.name = "decision_tree_max_depth_" + window.activityNumberDecisionMaxDepth
    
      trackElementDecisionMaxDepth.appendChild(clone)
      window.activityNumberDecisionMaxDepth += 1
    }
    if(clickedId == "decision_tree_min_samples_split_button_add") {
      var trackElementDecisionMinSamplesSplit = document.getElementById("decision_tree_min_samples_split_grid_search")
      var toBeCloned = document.getElementById("decision_tree_min_samples_split")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "decision_tree_min_samples_split_" + window.activityNumberDecisionMinSamplesSplit
      clone.name = "decision_tree_min_samples_split_" + window.activityNumberDecisionMinSamplesSplit
    
      trackElementDecisionMinSamplesSplit.appendChild(clone)
      window.activityNumberDecisionMinSamplesSplit += 1
    }
    if(clickedId == "decision_tree_min_samples_leaf_button_add") {
      var trackElementDecisionMinSamplesLeaf = document.getElementById("decision_tree_min_samples_leaf_grid_search")
      var toBeCloned = document.getElementById("decision_tree_min_samples_leaf")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "decision_tree_min_samples_leaf_" + window.activityNumberDecisionMinSamplesLeaf
      clone.name = "decision_tree_min_samples_leaf_" + window.activityNumberDecisionMinSamplesLeaf
    
      trackElementDecisionMinSamplesLeaf.appendChild(clone)
      window.activityNumberDecisionMinSamplesLeaf += 1
    }
    if(clickedId == "decision_tree_max_leaf_nodes_button_add") {
      var trackElementDecisionMaxLeafNodes = document.getElementById("decision_tree_max_leaf_nodes_grid_search")
      var toBeCloned = document.getElementById("decision_tree_max_leaf_nodes")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "decision_tree_max_leaf_nodes_" + window.activityNumberDecisionMaxLeafNodes
      clone.name = "decision_tree_max_leaf_nodes_" + window.activityNumberDecisionMaxLeafNodes
    
      trackElementDecisionMaxLeafNodes.appendChild(clone)
      window.activityNumberDecisionMaxLeafNodes += 1
    }
    if(clickedId == "decision_tree_max_features_button_add") {
      var trackElementDecisionMaxFeatures = document.getElementById("decision_tree_max_features_grid_search")
      var toBeCloned = document.getElementById("decision_tree_max_features")
      var clone

      clone = toBeCloned.cloneNode(true);
      clone.id = "decision_tree_max_features_" + window.activityNumberDecisionMaxFeatures
      clone.name = "decision_tree_max_features_" + window.activityNumberDecisionMaxFeatures

      trackElementDecisionMaxFeatures.appendChild(clone)
      window.activityNumberDecisionMaxFeatures += 1
    }
  // MLP
  }else if(selected.value == "mlp") {
    if(clickedId == "mlp_hidden_layer_sizes_button_add") {
      var trackElementMlpHiddenLayerSizes = document.getElementById("mlp_hidden_layer_sizes_grid_search")
      var toBeCloned = document.getElementById("mlp_hidden_layer_sizes")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "mlp_hidden_layer_sizes_" + window.activityNumberMlpHiddenLayerSizes
      clone.name = "mlp_hidden_layer_sizes_" + window.activityNumberMlpHiddenLayerSizes
    
      trackElementMlpHiddenLayerSizes.appendChild(clone)
      window.activityNumberMlpHiddenLayerSizes += 1
    }
    if(clickedId == "mlp_activation_button_add") {
      if(window.activityNumberMlpActivation <= 4) {
        var trackElementMlpActivation = document.getElementById("mlp_activation_grid_search")
        var toBeCloned = document.getElementById("mlp_activation")
        var clone
      
        clone = toBeCloned.cloneNode(true);
        clone.id = "mlp_activation_" + window.activityNumberMlpActivation
        clone.name = "mlp_activation_" + window.activityNumberMlpActivation
      
        trackElementMlpActivation.appendChild(clone)
        window.activityNumberMlpActivation += 1
      }
    }
    if(clickedId == "mlp_solver_button_add") {
      if(window.activityNumberMlpSolver <= 3) {
        var trackElementMlpSolver = document.getElementById("mlp_solver_grid_search")
        var toBeCloned = document.getElementById("mlp_solver")
        var clone
      
        clone = toBeCloned.cloneNode(true);
        clone.id = "mlp_solver_" + window.activityNumberMlpSolver
        clone.name = "mlp_solver_" + window.activityNumberMlpSolver
      
        trackElementMlpSolver.appendChild(clone)
        window.activityNumberMlpSolver += 1
      }
    }
    if(clickedId == "mlp_alpha_button_add") {
      var trackElementMlpAlpha = document.getElementById("mlp_alpha_grid_search")
      var toBeCloned = document.getElementById("mlp_alpha")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "mlp_alpha_" + window.activityNumberMlpAlpha
      clone.name = "mlp_alpha_" + window.activityNumberMlpAlpha
    
      trackElementMlpAlpha.appendChild(clone)
      window.activityNumberMlpAlpha += 1
    }
    if(clickedId == "mlp_max_iter_button_add") {
      var trackElementMlpMaxIter = document.getElementById("mlp_max_iter_grid_search")
      var toBeCloned = document.getElementById("mlp_max_iter")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "mlp_max_iter_" + window.activityNumberMlpMaxIter
      clone.name = "mlp_max_iter_" + window.activityNumberMlpMaxIter
    
      trackElementMlpMaxIter.appendChild(clone)
      window.activityNumberMlpMaxIter += 1
    }
    if(clickedId == "mlp_tol_button_add") {
      var trackElementMlpTol = document.getElementById("mlp_tol_grid_search")
      var toBeCloned = document.getElementById("mlp_tol")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "mlp_tol_" + window.activityNumberMlpTol
      clone.name = "mlp_tol_" + window.activityNumberMlpTol
    
      trackElementMlpTol.appendChild(clone)
      window.activityNumberMlpTol += 1
    }    
  // Random Forest
  }else if(selected.value == "random_forest") {
    if(clickedId == "random_forest_n_estimators_button_add") {
      var trackElementRandomForestEstimators = document.getElementById("random_forest_n_estimators_grid_search")
      var toBeCloned = document.getElementById("random_forest_n_estimators")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "random_forest_n_estimators_" + window.activityNumberRandomForestEstimators
      clone.name = "random_forest_n_estimators_" + window.activityNumberRandomForestEstimators
    
      trackElementRandomForestEstimators.appendChild(clone)
      window.activityNumberRandomForestEstimators += 1
    }
    if(clickedId == "random_forest_criterion_button_add") {
      var trackElementRandomForestCriterion = document.getElementById("random_forest_criterion_grid_search")
      var toBeCloned = document.getElementById("random_forest_criterion")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "random_forest_criterion_" + window.activityNumberRandomForestCriterion
      clone.name = "random_forest_criterion_" + window.activityNumberRandomForestCriterion
    
      trackElementRandomForestCriterion.appendChild(clone)
      window.activityNumberRandomForestCriterion += 1
    }
    if(clickedId == "random_forest_max_depth_button_add") {
      var trackElementRandomForestMaxDepth = document.getElementById("random_forest_max_depth_grid_search")
      var toBeCloned = document.getElementById("random_forest_max_depth")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "random_forest_max_depth_" + window.activityNumberRandomForestMaxDepth
      clone.name = "random_forest_max_depth_" + window.activityNumberRandomForestMaxDepth
    
      trackElementRandomForestMaxDepth.appendChild(clone)
      window.activityNumberRandomForestMaxDepth += 1
    }
    if(clickedId == "random_forest_min_samples_split_button_add") {
      var trackElementRandomForestMinSamplesSplit = document.getElementById("random_forest_min_samples_split_grid_search")
      var toBeCloned = document.getElementById("random_forest_min_samples_split")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "random_forest_min_samples_split_" + window.activityNumberRandomForestMinSamplesSplit
      clone.name = "random_forest_min_samples_split_" + window.activityNumberRandomForestMinSamplesSplit
    
      trackElementRandomForestMinSamplesSplit.appendChild(clone)
      window.activityNumberRandomForestMinSamplesSplit += 1
    }
    if(clickedId == "random_forest_min_samples_leaf_button_add") {
      var trackElementRandomForestMinSamplesLeaf = document.getElementById("random_forest_min_samples_leaf_grid_search")
      var toBeCloned = document.getElementById("random_forest_min_samples_leaf")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "random_forest_min_samples_leaf_" + window.activityNumberRandomForestMinSamplesLeaf
      clone.name = "random_forest_min_samples_leaf_" + window.activityNumberRandomForestMinSamplesLeaf
    
      trackElementRandomForestMinSamplesLeaf.appendChild(clone)
      window.activityNumberRandomForestMinSamplesLeaf += 1
    }
    if(clickedId == "random_forest_max_features_button_add") {
      var trackElementRandomForestMaxFeatures = document.getElementById("random_forest_max_features_grid_search")
      var toBeCloned = document.getElementById("random_forest_max_features")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "random_forest_max_features_" + window.activityNumberRandomForestMaxFeatures
      clone.name = "random_forest_max_features_" + window.activityNumberRandomForestMaxFeatures
    
      trackElementRandomForestMaxFeatures.appendChild(clone)
      window.activityNumberRandomForestMaxFeatures += 1
    }
    
  }else if(selected.value == "svr") {
    if(clickedId == "svr_kernel_button_add") {
      if(window.activityNumberSvrKernel <= 4){
        var trackElementSvrKernel = document.getElementById("svr_kernel_grid_search")
        var toBeCloned = document.getElementById("svr_kernel")
        var clone
      
        clone = toBeCloned.cloneNode(true);
        clone.id = "svr_kernel_" + window.activityNumberSvrKernel
        clone.name = "svr_kernel_" + window.activityNumberSvrKernel
      
        trackElementSvrKernel.appendChild(clone)
        window.activityNumberSvrKernel += 1
      }
    }
    if(clickedId == "svr_degree_button_add") {
      var trackElementSvrDegree = document.getElementById("svr_degree_grid_search")
      var toBeCloned = document.getElementById("svr_degree")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "svr_degree_" + window.activityNumberSvrDegree
      clone.name = "svr_degree_" + window.activityNumberSvrDegree
    
      trackElementSvrDegree.appendChild(clone)
      window.activityNumberSvrDegree += 1
    }
    if(clickedId == "svr_gamma_button_add") {
      var trackElementSvrGamma = document.getElementById("svr_gamma_grid_search")
      var toBeCloned = document.getElementById("svr_gamma")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "svr_gamma_" + window.activityNumberSvrGamma
      clone.name = "svr_gamma_" + window.activityNumberSvrGamma
    
      trackElementSvrGamma.appendChild(clone)
      window.activityNumberSvrGamma += 1
    }
    if(clickedId == "svr_tol_button_add") {
      var trackElementSvrTol = document.getElementById("svr_tol_grid_search")
      var toBeCloned = document.getElementById("svr_tol")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "svr_tol_" + window.activityNumberSvrTol
      clone.name = "svr_tol_" + window.activityNumberSvrTol
    
      trackElementSvrTol.appendChild(clone)
      window.activityNumberSvrTol += 1
    }
    if(clickedId == "svr_c_button_add") {
      var trackElementSvrC = document.getElementById("svr_c_grid_search")
      var toBeCloned = document.getElementById("svr_c")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "svr_c_" + window.activityNumberSvrC
      clone.name = "svr_c_" + window.activityNumberSvrC
    
      trackElementSvrC.appendChild(clone)
      window.activityNumberSvrC += 1
    }
    if(clickedId == "svr_epsilon_button_add") {
      var trackElementSvrEpsilon = document.getElementById("svr_epsilon_grid_search")
      var toBeCloned = document.getElementById("svr_epsilon")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "svr_epsilon_" + window.activityNumberSvrEpsilon
      clone.name = "svr_epsilon_" + window.activityNumberSvrEpsilon
    
      trackElementSvrEpsilon.appendChild(clone)
      window.activityNumberSvrEpsilon += 1
    }
    if(clickedId == "svr_max_iter_button_add") {
      var trackElementSvrMaxIter = document.getElementById("svr_max_iter_grid_search")
      var toBeCloned = document.getElementById("svr_max_iter")
      var clone
    
      clone = toBeCloned.cloneNode(true);
      clone.id = "svr_max_iter_" + window.activityNumberSvrMaxIter
      clone.name = "svr_max_iter_" + window.activityNumberSvrMaxIter
    
      trackElementSvrMaxIter.appendChild(clone)
      window.activityNumberSvrMaxIter += 1
    }      
  }
    
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//                      REMOVE ELEMENTS FROM GRID SEARCH
//-----------------------------------------------------------------------------
function removeElement(clickedId) {

  var selected
  var trackElement

  selected = document.getElementById("select_method")
 
  // Linear Regression
  if(selected.value == "linear_regression") {
    if(clickedId == "linear_fit_intercept_button_remove") {
      var elementNumber = window.activityNumberLinearFitIntercept - 1
    
      trackElement = document.getElementById("linear_fit_intercept_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberLinearFitIntercept -= 1
    }    
  // Decision Trees
  }else if(selected.value == "decision_tree") {
    if(clickedId == "decision_tree_criterion_button_remove") {
      var elementNumber = window.activityNumberDecisionCriterion - 1
    
      trackElement = document.getElementById("decision_tree_criterion_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberDecisionCriterion -= 1
    }
    if(clickedId == "decision_tree_splitter_button_remove") {
      var elementNumber = window.activityNumberDecisionSplitter - 1
    
      trackElement = document.getElementById("decision_tree_splitter_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberDecisionSplitter -= 1
    }    
    if(clickedId == "decision_tree_max_depth_button_remove") {
      var elementNumber = window.activityNumberDecisionMaxDepth - 1
    
      trackElement = document.getElementById("decision_tree_max_depth_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberDecisionMaxDepth -= 1
    }
    if(clickedId == "decision_tree_min_samples_split_button_remove") {
      var elementNumber = window.activityNumberDecisionMinSamplesSplit - 1
    
      trackElement = document.getElementById("decision_tree_min_samples_split_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberDecisionMinSamplesSplit -= 1
    }
    if(clickedId == "decision_tree_min_samples_leaf_button_remove") {
      var elementNumber = window.activityNumberDecisionMinSamplesLeaf - 1
    
      trackElement = document.getElementById("decision_tree_min_samples_leaf_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberDecisionMinSamplesLeaf -= 1
    }
    if(clickedId == "decision_tree_max_leaf_nodes_button_remove") {
      var elementNumber = window.activityNumberDecisionMaxLeafNodes - 1
    
      trackElement = document.getElementById("decision_tree_max_leaf_nodes_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberDecisionMaxLeafNodes -= 1
    }
    if(clickedId == "decision_tree_max_features_button_remove") {
      var elementNumber = window.activityNumberDecisionMaxFeatures - 1
    
      trackElement = document.getElementById("decision_tree_max_features_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberDecisionMaxFeatures -= 1
    }
  // MLP
  }else if(selected.value == "mlp") {
    if(clickedId == "mlp_hidden_layer_sizes_button_remove") {
      var elementNumber = window.activityNumberMlpHiddenLayerSizes - 1
    
      trackElement = document.getElementById("mlp_hidden_layer_sizes_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberMlpHiddenLayerSizes -= 1
    }
    if(clickedId == "mlp_activation_button_remove") {
      var elementNumber = window.activityNumberMlpActivation - 1
    
      trackElement = document.getElementById("mlp_activation_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberMlpActivation -= 1
    }
    if(clickedId == "mlp_solver_button_remove") {
      var elementNumber = window.activityNumberMlpSolver - 1
    
      trackElement = document.getElementById("mlp_solver_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberMlpSolver -= 1
    }
    if(clickedId == "mlp_alpha_button_remove") {
      var elementNumber = window.activityNumberMlpAlpha - 1
    
      trackElement = document.getElementById("mlp_alpha_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberMlpAlpha -= 1
    }
    if(clickedId == "mlp_max_iter_button_remove") {
      var elementNumber = window.activityNumberMlpMaxIter - 1
    
      trackElement = document.getElementById("mlp_max_iter_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberMlpMaxIter -= 1
    }
    if(clickedId == "mlp_tol_button_remove") {
      var elementNumber = window.activityNumberMlpTol - 1
    
      trackElement = document.getElementById("mlp_tol_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberMlpTol -= 1
    }    
  // Random Forest
  }else if(selected.value == "random_forest") {
    
    if(clickedId == "random_forest_n_estimators_button_remove") {
      var elementNumber = window.activityNumberRandomForestEstimators - 1
    
      trackElement = document.getElementById("random_forest_n_estimators_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberRandomForestEstimators -= 1
    }
    if(clickedId == "random_forest_criterion_button_remove") {
      var elementNumber = window.activityNumberRandomForestCriterion - 1
    
      trackElement = document.getElementById("random_forest_criterion_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberRandomForestCriterion -= 1
    }
    if(clickedId == "random_forest_max_depth_button_remove") {
      var elementNumber = window.activityNumberRandomForestMaxDepth - 1
    
      trackElement = document.getElementById("random_forest_max_depth_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberRandomForestMaxDepth -= 1
    }
    if(clickedId == "random_forest_min_samples_split_button_remove") {
      var elementNumber = window.activityNumberRandomForestMinSamplesSplit - 1
    
      trackElement = document.getElementById("random_forest_min_samples_split_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberRandomForestMinSamplesSplit -= 1
    }
    if(clickedId == "random_forest_min_samples_leaf_button_remove") {
      var elementNumber = window.activityNumberRandomForestMinSamplesLeaf - 1
    
      trackElement = document.getElementById("random_forest_min_samples_leaf_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberRandomForestMinSamplesLeaf -= 1
    }
    if(clickedId == "random_forest_max_features_button_remove") {
      var elementNumber = window.activityNumberRandomForestMaxFeatures - 1
    
      trackElement = document.getElementById("random_forest_max_features_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberRandomForestMaxFeatures -= 1
    }
            
  }else if(selected.value == "svr") {
    if(clickedId == "svr_kernel_button_remove") {
      var elementNumber = window.activityNumberSvrKernel - 1
    
      trackElement = document.getElementById("svr_kernel_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberSvrKernel -= 1
    }
    if(clickedId == "svr_degree_button_remove") {
      var elementNumber = window.activityNumberSvrDegree - 1
    
      trackElement = document.getElementById("svr_degree_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberSvrDegree -= 1
    }
    if(clickedId == "svr_gamma_button_remove") {
      var elementNumber = window.activityNumberSvrGamma - 1
    
      trackElement = document.getElementById("svr_gamma_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberSvrGamma -= 1
    }
    if(clickedId == "svr_tol_button_remove") {
      var elementNumber = window.activityNumberSvrTol - 1
    
      trackElement = document.getElementById("svr_tol_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberSvrTol -= 1
    }
    if(clickedId == "svr_c_button_remove") {
      var elementNumber = window.activityNumberSvrC - 1
    
      trackElement = document.getElementById("svr_c_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberSvrC -= 1
    }
    if(clickedId == "svr_epsilon_button_remove") {
      var elementNumber = window.activityNumberSvrEpsilon - 1
    
      trackElement = document.getElementById("svr_epsilon_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberSvrEpsilon -= 1
    }
    if(clickedId == "svr_max_iter_button_remove") {
      var elementNumber = window.activityNumberSvrMaxIter - 1
    
      trackElement = document.getElementById("svr_max_iter_" + elementNumber)
      trackElement.parentNode.removeChild(trackElement)
      window.activityNumberSvrMaxIter -= 1
    }    
  }

}
//-----------------------------------------------------------------------------