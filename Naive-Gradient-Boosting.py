import numpy as np
import pandas as pd
from math import e

class Node:
    '''
    This class defines a node which creates a tree structure by recursively calling itself 
    whilst checking a number of ending parameters such as depth and min_leaf. It uses an exact greedy method
    to exhaustively scan every possible split point. The gain metric of choice is conservation of varience.
    This is a Naive solution and does not comapre to Frieman's 2001 Gradient Boosting Machines
    
    Input
    ------------------------------------------------------------------------------------------
    X: Pandas dataframe
    y: Pandas Series
    idxs: indices of values used to keep track of splits points in tree
    predictions: are the predictions of a gradient boosting algorthim thus far used to calculate the leaf values
    min_leaf: minimum number of samples needed to be classified as a node
    depth: sets the maximum depth allowed
    classification: a flag that indicates if the problem is regression or binary classification
    
    Output
    ---------------------------------------------------------------------------------------------
    Regression tree that can either be used for classification or regression
    '''

    def __init__(self, x, y, idxs, classification, predictions, min_leaf=5, depth = 10):
        self.x, self.y = x, y
        self.idxs = idxs 
        self.depth = depth
        self.min_leaf = min_leaf
        self.row_count = len(idxs)
        self.col_count = x.shape[1]
        self.classification = classification
        self.predictions = predictions
        
        if classification:
            self.val = self.compute_leaf_value(y[idxs], predictions[idxs])
        else:
            self.val = np.mean(y[idxs])
          
        self.score = float('inf')
        self.find_varsplit()
            
    def find_varsplit(self):
        '''
        Scans through every column and calcuates the best split point.
        The node is then split at this point and two new nodes are created.
        Depth is only parameter to change as we have added a new layer to tre structure.
        If no split is better than the score initalised at the begining then no splits further splits are made
        
        '''
        for c in range(self.col_count): self.find_better_split(c)
        if self.is_leaf: return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = Node(self.x, self.y, self.idxs[lhs], self.classification, self.predictions, self.min_leaf, depth = self.depth-1)
        self.rhs = Node(self.x, self.y, self.idxs[rhs], self.classification, self.predictions, self.min_leaf, depth = self.depth-1)
        
    def find_better_split(self, var_idx):
        '''
        For a given feature calculates the gain at each split.
        Globally updates the best score if a better split point is found
        '''
        x = self.x.values[self.idxs, var_idx]

        for r in range(self.row_count):
            lhs = x <= x[r]
            rhs = x > x[r]
            if rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf: continue

            curr_score = self.find_score(lhs, rhs)
            if curr_score < self.score: 
                self.var_idx = var_idx
                self.score = curr_score
                self.split = x[r]
                
    def find_score(self, lhs, rhs):
        '''
        Calculates or metric for evaluating split use standard deviation chooses splits where standard 
        deviation is low as it means these points are similar and can be grouped together.
        '''
        y = self.y[self.idxs]
        lhs_std = y[lhs].std()
        rhs_std = y[rhs].std()
        return lhs_std * lhs.sum() + rhs_std * rhs.sum()

    def compute_leaf_value(self, leaf_values, predictions):
        '''
        if we are constructing a GBM classifier this is the optimal leaf node 
        value.
        '''
        p = self.sigmoid(predictions)
        denominator = p*(1- p)
        return(np.sum(leaf_values)/np.sum(denominator))
    
    @staticmethod  
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @property
    def split_col(self):
        return self.x.values[self.idxs,self.var_idx]
                
    @property
    def is_leaf(self):
        return self.score == float('inf') or self.depth <= 0                 

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        node = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return node.predict_row(xi)

class DecisionTreeRegressor:
    '''
    Wrapper class that provides a scikit learn interface to the recursive regression tree above
    
    Input
    ------------------------------------------------------------------------------------------
    X: Pandas dataframe
    y: Pandas Series
    idxs: indices of values used to keep track of splits points in tree
    predictions: are the predictions of a gradient boosting algorthim thus far used to calculate the leaf values
    min_leaf: minimum number of samples needed to be classified as a node
    depth: sets the maximum depth allowed
    classification: a flag that indicates if the problem is regression or binary classification
    
    '''

    def fit(self, X, y, classification ,min_leaf = 5, depth = 5, predictions = []):
        self.dtree = Node(x = X, y = y, idxs = np.array(np.arange(len(y))), predictions = predictions, min_leaf = min_leaf, depth = depth, classification = classification)
        return self
    
    def predict(self, X):
        return self.dtree.predict(X.values)
        
class GradientBoostingClassification:
    '''
    Applies the methododlgy of gradeint boosting for binary classification only.
    Uses the Binary logistic loss function to calculate the negative derivate.
    
    Input
    ------------------------------------------------------------------------------------------
    X: Pandas dataframe
    y: Pandas Series
    min_leaf: minimum number of samples needed to be classified as a node
    depth: sets the maximum depth allowed
    Boosting_Rounds: number of boosting rounds or iterations
    
    Output
    ---------------------------------------------------------------------------------------------
    Gradient boosting machine that can be used for binary classification.
    
    '''
    def __init__(self):
        self.estimators = []
        
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
        
    def negativeDerivitiveLogloss(self, y, log_odds):
        p = self.sigmoid(log_odds)
        return(y - p)
        
    @staticmethod
    def log_odds(column):
        '''
        Calculates the initial log odds prediction of the taget variable
        '''
        if isinstance(column, pd.Series):
            binary_yes = np.count_nonzero(column.values == 1)
            binary_no  = np.count_nonzero(column.values == 0)
        elif isinstance(column, list):
            column = np.array(column)
            binary_yes = np.count_nonzero(column == 1)
            binary_no  = np.count_nonzero(column == 0) 
        else:
            binary_yes = np.count_nonzero(column == 1)
            binary_no  = np.count_nonzero(column == 0)
            
        value = np.log(binary_yes/binary_no)
        return(np.full((len(column), 1), value).flatten())
    
    def fit(self, X, y, depth = 5, min_leaf = 5, learning_rate = 0.1, boosting_rounds = 5):
        
        # use the log odds value of the target variable as our inital prediction
        self.learning_rate = learning_rate
        self.base_pred = self.log_odds(y)
        
        for booster in range(boosting_rounds):
            # Calculate the initial Pseudo Residuals using Base Prediction.
            pseudo_residuals = self.negativeDerivitiveLogloss(y, self.base_pred)
            # Approximate the residuals
            boosting_tree = DecisionTreeRegressor().fit(X = X, y = pseudo_residuals, depth = 5, min_leaf = 5, classification = True , predictions = self.base_pred)
            self.base_pred += self.learning_rate * boosting_tree.predict(X)
            # store predictors for later
            self.estimators.append(boosting_tree)
   
    def predict(self, X):
        
        pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X)
            
        return self.log_odds(y) + pred
        
        
class GradientBoostingRegressor:
    '''
    Applies the methododlgy of gradeint boosting for regression.
    Uses the mean squared loss function to calculate the negative derivate.
    
    Input
    ------------------------------------------------------------------------------------------
    X: Pandas dataframe
    y: Pandas Series
    min_leaf: minimum number of samples needed to be classified as a node
    depth: sets the maximum depth allowed
    Boosting_Rounds: number of boosting rounds or iterations
    
    Output
    ---------------------------------------------------------------------------------------------
    Gradient boosting machine that can be used for regression.
    
    '''
    def __init__(self):
        self.estimators = []
      
    @staticmethod
    def negativeMeanSquaredErrorDerivitive(y, y_pred):
        return(2*(y-y_pred))
        
    def fit(self, X, y, depth = 5, min_leaf = 5, learning_rate = 0.1, boosting_rounds = 5):
        
        # Start with the mean y value as our initial prediciton
        self.learning_rate = learning_rate
        self.base_pred = np.full((X.shape[0], 1), np.mean(y)).flatten()
        
        for booster in range(boosting_rounds):
            # Calculate the initial Pseudo Residuals using Base Prediction.
            pseudo_residuals = self.negativeMeanSquaredErrorDerivitive(y, self.base_pred)
            # Approximate the residuals
            boosting_tree = DecisionTreeRegressor().fit(X = X, y = pseudo_residuals, depth = 5, min_leaf = 5, classification = False , predictions = None)
            self.base_pred += self.learning_rate * boosting_tree.predict(X)
            # store predictors for later
            self.estimators.append(boosting_tree)
   
    def predict(self, X):
        
        pred = np.zeros(X.shape[0])
        for estimator in self.estimators:
            pred += self.learning_rate * estimator.predict(X)
            
        return np.full((X.shape[0], 1), np.mean(y)).flatten() + pred
