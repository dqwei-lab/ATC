from chemocommons import * 
import scipy.io as scio
from skmultilearn.cluster import NetworkXLabelGraphClusterer
from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
from skmultilearn.ensemble import LabelSpacePartitioningClassifier
from skmultilearn.problem_transform import ClassifierChain, LabelPowerset
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import LeaveOneOut, cross_validate, RepeatedKFold
from sklearn.metrics import jaccard_similarity_score
from skmultilearn.utils import measure_per_label

data_dict = scio.loadmat("ATC_42_3883.mat")
X = data_dict['atc_fea'] # X is standardized [0, 1] no need for scale
Y = data_dict['atcClass']
Y[Y==-1] = 0

X = X.T
Y = Y.T

def measure_per_label(measure, y_true, y_predicted):
    """
        This code is inspired by skmultilearn, but our y_true and y_predicted are all dense numpy.ndarray
    """
    return [
        measure(
            y_true[:, i],
            y_predicted[:, i]
        )
        for i in range(y_true.shape[1])
    ]


XGB, MLP = load("ensemble_xgb_mlp.joblib") # this is part of final training

final_model = XGB.best_estimator_
label_acc = []
label_sp = []
label_rc = []
label_f1 = []
label_auc = []



for i in range(10): #10*10-cv
    print(i, "th repeat:")
    kfold = KFold(10, random_state=i, shuffle=True)
    for k, (train, test) in enumerate(kfold.split(X, Y)):
        print(k, "th fold.")
        final_model.fit(X[train], Y[train])
        y_pred = np.array(final_model.predict(X[test]).todense())
        y_proba = np.array(final_model.predict_proba(X[test]).todense())
        label_acc.append(measure_per_label(metrics.accuracy_score, Y[test], y_pred))
        label_sp.append(measure_per_label(metrics.precision_score, Y[test], y_pred))
        label_rc.append(measure_per_label(metrics.recall_score, Y[test], y_pred))
        label_f1.append(measure_per_label(metrics.f1_score, Y[test], y_pred))
        label_auc.append(measure_per_label(metrics.roc_auc_score, Y[test], y_proba))

label_acc = np.array(label_acc)
label_sp = np.array(label_sp)
label_rc = np.array(label_rc)
label_f1 = np.array(label_f1)
label_auc = np.array(label_auc)

to_sav = dump((label_acc, label_sp, label_rc, label_f1, label_auc), file="report_array.joblib")

print(label_acc.mean(axis=0), label_sp.mean(axis=0), label_rc.mean(axis=0),label_f1.mean(axis=0), label_auc.mean(axis=0))
