"""
    Major model traning source code
    !!!CAUTION!!! Because of jackknife, the code is extremely slow, plz run in server! 
    A server with 32 cores may take 10 days to finish (SVM is the battelneck)
"""
__author__ = "Xiangeng Wang"  

from chemocommons import * # many useful functions, "reinvented" wheels, wrote by me!
import scipy.io as scio # load ".mat" file in 
from skmultilearn.cluster import NetworkXLabelGraphClusterer # clusterer
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder # as it writes
from skmultilearn.ensemble import LabelSpacePartitioningClassifier # so?
from skmultilearn.problem_transform import ClassifierChain, LabelPowerset # sorry, we only used LP
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier # Okay?
from sklearn.model_selection import LeaveOneOut # jackknife, "socalled"
from sklearn.metrics import jaccard_similarity_score # for some calculation


data_dict = scio.loadmat("ATC_42_3883.mat")
X = data_dict['atc_fea'] # X is standardized [0, 1] no need for scale
Y = data_dict['atcClass']
Y[Y==-1] = 0 # why in matlab is -1,1; not 0,1?

loocv = LeaveOneOut() # jackknife

scoring_funcs = {"hamming loss": hamming_func, 
                 "aiming": aiming_func, 
                 "coverage": coverage_func, 
                 "accuracy": accuracy_func, 
                 "absolute true": absolute_true_func, 
                 "absolute false":absolute_false_func
                 } # Keep recorded
parameters = {
    'classifier': [LabelPowerset()],
    'classifier__classifier': [ExtraTreesClassifier()],
    'classifier__classifier__n_estimators': [100, 500],
    'clusterer' : [
        NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'louvain'),
        NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'lpa')
    ]
}


ext = GridSearchCV(LabelSpacePartitioningClassifier(), param_grid=parameters, n_jobs=-1, cv=loocv, 
                    scoring=scoring_funcs, verbose=0, refit="absolute true")
ext.fit(X.T, Y.T)
print(ext.best_score_)


parameters = {
    'classifier': [LabelPowerset()],
    'classifier__classifier': [RandomForestClassifier()],
    'classifier__classifier__n_estimators': [500,1000],
    'clusterer' : [
        NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'louvain'),
        NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'lpa')
    ]
}

rf = GridSearchCV(LabelSpacePartitioningClassifier(), param_grid=parameters, n_jobs=-1, cv=loocv, 
                    scoring=scoring_funcs, verbose=0, refit="absolute true")
rf.fit(X.T, Y.T)
print(rf.best_score_)



parameters = {
    'classifier': [LabelPowerset()],
    'classifier__classifier': [SVC(probability=True)],
    'classifier__classifier__C': [0.01, 0.1, 1, 10, 100],
    'clusterer' : [
        NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'louvain'),
        NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'lpa')
    ]
}



svm = GridSearchCV(LabelSpacePartitioningClassifier(), param_grid=parameters, n_jobs=-1, cv=loocv, 
                    scoring=scoring_funcs, verbose=0, refit="absolute true")
svm.fit(X.T, Y.T)
print(svm.best_score_)


parameters = {
    'classifier': [LabelPowerset(), ClassifierChain()],
    'classifier__classifier': [XGBClassifier()],
    'classifier__classifier__n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'clusterer' : [
        NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'louvain'),
        NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'lpa')
    ]
}
xgb = GridSearchCV(LabelSpacePartitioningClassifier(), param_grid=parameters, n_jobs=-1, cv=loocv, 
                    scoring=scoring_funcs, verbose=0, refit="absolute true")
xgb.fit(X.T, Y.T)
print(xgb.best_score_)

parameters = {
    'classifier': [LabelPowerset(), ClassifierChain()],
    'classifier__classifier': [MLPClassifier()],
    'classifier__classifier__hidden_layer_sizes': [50, 100, 200, 500, 1000],
    'clusterer' : [
        NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'louvain'),
        NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'lpa')
    ]
}
mlp = GridSearchCV(LabelSpacePartitioningClassifier(), param_grid=parameters, n_jobs=-1, cv=loocv, 
                    scoring=scoring_funcs, verbose=0, refit="absolute true")
mlp.fit(X.T, Y.T)
print(mlp.best_score_)

mytuple = (
    ext,
    rf,
    svm,
    xgb,
    mlp,
)

to_save = dump(mytuple, filename="ensemble.joblib")
