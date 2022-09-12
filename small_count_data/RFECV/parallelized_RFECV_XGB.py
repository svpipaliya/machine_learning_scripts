import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler

# RFECV
from sklearn.feature_selection import RFECV
import yellowbrick

# CV 
from sklearn.model_selection import StratifiedKFold


## Classifiers
import xgboost
from sklearn import preprocessing

 # Recursive Feature Elimination With Cross-Validation (RFECV)
from RFECV_yellow_brick import rfecv
 
import os
import sys
import inspect
src_file_path = os.path.dirname(os.path.abspath("__file__"))
mymodule_dir = os.path.join( src_file_path, '..', 'import_data' )
sys.path.append( mymodule_dir )
from Data_processed_for_classification  import *



RS =20 # RANDOM STATE so that the results are reproducible

X = preprocessing.scale(X) # Scale the imput data 

# Optimal eXtreme Gradient Boost model
trainedmodel = xgboost.XGBClassifier(n_estimators= 900 ,
              colsample_bytree= 0.75,
              max_depth= 7,
              subsample=0.7 ,
              learning_rate= 0.05)

cv = StratifiedKFold(2)
#Measuring the optimal number of feature to find the optimal accuracy / cost tradeoff of the classifiers 
plt.figure()
visualizer = rfecv(estimator=trainedmodel ,
                   X = X ,
                   y= y ,
                   cv=cv, 
                   step=1 ,
                   n_jobs=-1 ,
                   scoring='accuracy' ,
                   random_state=RS, 
                   show=("../../../../ML_results/small_gene_count/feature_importance/parallelized_XGB_optimal_nbr_features.png"))

visualizer.fit(X, y)        # Fit the data to the visualizer
visualizer.show("../../../../ML_results/small_gene_count/feature_importance/parallelized_XGB_optimal_nbr_features.png") 
plt.close()

importance = visualizer.estimator_.feature_importances_
sorted_idx = importance.argsort()
optimal_nbr = visualizer.n_features_

plt.figure()
plt.barh(attributeNames[sorted_idx][-optimal_nbr:] ,importance [sorted_idx][-optimal_nbr:] )
plt.xlabel(" Most important features")
plt.ylabel(" Feature importance")
plt.savefig("../../../../ML_results/small_gene_count/feature_importance/parallelized_most_important_features_XGB.png")
plt.close()
with open("../../../../ML_results/small_gene_count/feature_importance/parallelized_XGB_most_important_genes.txt" , 'w') as f:
    for gene in  attributeNames[sorted_idx][-optimal_nbr:]:
        f.write(gene)
        f.write('\n')    