import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from statistics import mean
import warnings

## Importing utility scripts 
import os
import sys
import inspect
src_file_path = os.path.dirname(os.path.abspath("__file__"))
mymodule_dir = os.path.join( src_file_path, '..', '..', 'utility' )
sys.path.append( mymodule_dir )
from pairwise_comparaison import *

## Importing data  
import os
import sys
import inspect
src_file_path = os.path.dirname(os.path.abspath("__file__"))
mydata_dir = os.path.join( src_file_path, '..', 'import_data' )
sys.path.append( mydata_dir )
from All_counts_data_processed_for_classification  import *


## Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

## Importing the data
from All_counts_data_processed_for_classification  import *


##Prevent printing warnings
warnings.filterwarnings("ignore")

loss = 1  # missclassication error and not squared  
K = 10   # Number of folds for the CV 
m = 3    # How many time we randomise the training set

X = preprocessing.scale(X) # Scale the imput data 

#vector of accuracy difference
r = []
kf = model_selection.KFold(n_splits=K)




LR = LogisticRegression(solver='newton-cg' ,
                        penalty='l2',
                        C= 10e-3)

RF = RandomForestClassifier(max_features= 'auto' ,
                                       n_estimators=17 ,
                                       min_samples_split =7 ,
                                       max_depth =2 ,
                                       min_samples_leaf =12 , 
                                       bootstrap=False )

SVM = SVC(kernel='linear' ,
           gamma=10000 ,
           C =0.0001,
           probability=True)


MLP = MLPClassifier(hidden_layer_sizes=5,
                              solver = 'sgd',
                              activation= "relu",
                              learning_rate="adaptive",
                              alpha=0.1  )


XGB = xgboost.XGBClassifier(n_estimators= 700 ,
                            colsample_bytree= 0.7864840831382067 ,
                            max_depth= 15,
                            subsample=0.6 ,
                            learning_rate= 0.1)


models_1 = [LR ,RF ,SVM ,MLP]
models_2 = [ [RF ,SVM ,MLP, XGB],[SVM ,MLP, XGB] ,[MLP, XGB] , [XGB ]]

models_1_ = ["LR" ,"RF" ,"SVM" ,"MLP"]
models_2_ =  [ ["RF" ,"SVM" ,"MLP", "XGB"],["SVM" ,"MLP", "XGB"] ,["MLP", "XGB"] , ["XGB"] ]


#Here we compare the LR to the baseline
#Model A is the LR and model B is the baseline

with open("../../../../ML_results/all_gene_count/classification_performance/model_pairwise_comparaison.txt" , 'w') as f:
    for ele in  ["With a confidence level",0.05  ,'\n' ] :
        f.write(str(ele))


for i in range (len(models_1)) :
    for j in range (len(models_2[i]) ):
        for dm in range(m):
            print('Randomization fold: {0}/{1}'.format(dm+1,m))
            y_true = []
            yhat = []
            z=0
            for train_index, test_index in kf.split(X):
                print('Crossvalidation fold: {0}/{1}'.format(z+1,K))  
                
                X_train = X[train_index,:]
                y_train = y[train_index]
                X_test = X[test_index,:]
                y_test = y[test_index]
                
            
                mA = models_1[i];
                mA.fit(X_train, y_train)
                yhatA = mA.predict(X_test)
                    
                mB = models_2[i][j];
                mB.fit(X_train, y_train)
                yhatB = mB.predict(X_test)
                
                yhatA = yhatA.reshape(len(yhatA) , 1)
                yhatB =yhatB.reshape(len(yhatB) , 1)
                
                y_true.append(y_test)
                yhat.append( np.concatenate([yhatA, yhatB], axis=1) )
                
                r.append( np.mean( np.abs( yhatA-y_test ) ** loss - np.abs( yhatB-y_test) ** loss ) )
                z = z+1
                
        # Initialize parameters and run test appropriate for setup II
        alpha = 0.05
        rho = 1/ K
        p_setupII, CI_setupII = correlated_ttest(r, rho, alpha=alpha)
        
        with open("../../../../ML_results/all_gene_count/classification_performance/model_pairwise_comparaison.txt" , 'a') as f:
            for ele in  [" Test :", models_1_[i] , "loss is signficantly different from the ", models_2_[i][j] , "loss , p-value : ",  p_setupII , '\n' ,
                         " Test :", models_1_[i] , "loss is signficantly different from the ", models_2_[i][j] , " loss , CI : ",  CI_setupII  , '\n'] :
                f.write(str(ele))


