"""
    Common functions for all projects
"""
__author__ = "Xiangeng Wang"  


import os
from joblib import load, dump



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', 10)


"""
   Support functionality of sklearn
"""
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, roc_curve, auc
import sklearn.metrics as metrics 
"""
   Sklearn models
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMModel
"""
   Stacking models
"""
from mlxtend.classifier import StackingCVClassifier
"""
    chemoinformatics functionality
"""
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
import molvs as mv # for standardization 

"""
    Constants 
"""
RANDOM_SEED = 1994
MACCS_LENGTH = 166

def y_scramble(model, X_train, y_train, X_test, y_test, time=50):
    mcc_ = []
    f1_ = []
    for i in range(0, time):
        np.random.seed(i)
        y_train_ = np.random.permutation(y_train)
        model_ = model
        model_.fit(X_train, y_train_)
        mcc_.append(metrics.matthews_corrcoef(y_test, model_.predict(X_test)))
        f1_.append(metrics.f1_score(y_test, model_.predict(X_test)))  
        print("PROCESSED: {}".format(i))
    
    model_ = model
    model_.fit(X_train, y_train)
    mcc_.append(metrics.matthews_corrcoef(y_test, model_.predict(X_test)))
    f1_.append(metrics.f1_score(y_test, model_.predict(X_test)))  
    hue_ = ["Scrambled"] * time
    hue_.append("Unscrambled")
    return mcc_, f1_, hue_


def parse_sdf(filename, save=True):
    basename = filename.split(".")[0]
    collector = []
    sdprovider = Chem.SDMolSupplier(filename)
    for i,mol in enumerate(sdprovider):
        try:
            moldict = {}
            moldict['SMILES'] = Chem.MolToSmiles(mol)
            for propname in mol.GetPropNames():
                moldict[propname] = mol.GetProp(propname)
            collector.append(moldict)
        except:
            print("Molecule %s failed"%i)
    data = pd.DataFrame(collector)
    if save:
        data.to_csv(basename + '.csv')
    return data


def standardize_my_smiles(smiles):
    st = mv.Standardizer() #MolVS standardizer
    try:
        mols = st.charge_parent(Chem.MolFromSmiles(smiles))
        return Chem.MolToSmiles(mols)
    except:
        print("%s failed conversion"%smiles)
        return None

def save_sdf_from_list(filename, smiles_list):
    """
        save the SMILES into a SDF format
    """
    w = Chem.SDWriter(filename)
    mol_list = map(Chem.MolFromSmiles, smiles_list)
    for mol in mol_list:
        w.write(mol)
    w.close()
    
def get_mol2vec_from_list(smiles_list, pretrained_model):
    """
        get the mol2vec from a SMILES list
    """
    save_sdf_from_list("temp.sdf", smiles_list)
    from mol2vec import features
    features.featurize("temp.sdf", "mol_vec.csv", pretrained_model, r=1, uncommon="UNK")
    vec = pd.read_csv("mol_vec.csv", index_col=2)
    vec.drop(axis=1, labels=["Unnamed: 0", "ID"], inplace=True)
    os.remove("temp.sdf")
    os.remove("mol_vec.csv")
    return vec

def get_ecfp_from_list(smiles_list, radius=3, length=2048):
    """
        get the ecfp from SMILES list
    """
    now_list = []
    mol_obj = map(Chem.MolFromSmiles, smiles_list)
    arr = np.zeros((1,), dtype='int')
    sample_fp6 = []
    for (i, mol) in enumerate(mol_obj):
        if i % 1000 == 0:
            print("PROCESSED: {}".format(i))
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, length)
        except Exception as e:
            print("we have {} as wrong SMILES".format(i))
        else:
            DataStructs.ConvertToNumpyArray(fp, arr)
            sample_fp6.append(arr.tolist())
            now_list.append(smiles_list[i])

    sample_fp = pd.DataFrame(sample_fp6, dtype='int', index=now_list)
    sample_fp.rename(columns=lambda x: "ECFP{}_".format(2 * radius) + str(x), inplace=True)
    return sample_fp

def get_maccs_from_list(smiles_list):
    """
        get the maccs from SMILES list
    """
    mol_obj = map(Chem.MolFromSmiles, smiles_list)
    arr = np.zeros((1,), dtype='int')
    sample_maccs = []
    for (i, mol) in enumerate(mol_obj):
        if i % 1000 == 0:
            print("PROCESSED: {}".format(i))
        try:
            fp = MACCSkeys.GenMACCSKeys(mol)
        except Exception as e:
            print("we have {} as wrong SMILES".format(i))
        else:
            DataStructs.ConvertToNumpyArray(fp, arr)
            sample_maccs.append(arr.tolist())

    sample_maccs = pd.DataFrame(sample_maccs, dtype='int', index=smiles_list)
    sample_maccs.rename(columns=lambda x: "MACCS_" + str(x), inplace=True)
    return sample_maccs


def get_scores(model, X, y):
    prediction = model.predict(X)
    print("Hamming loss: %.4f" % metrics.hamming_loss(y, prediction))
    print("Accuracy score: %.4f" % metrics.accuracy_score(y, prediction))

"""
    scorings for the multi-label models
"""
    
def hamming_score(y_true, y_pred):
    """
        make sure the Ys must be dense numpy.ndarray
    """
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred.todense())
    return metrics.hamming_loss(y_pred, y_true)

def aiming(y_true, y_pred):
    """
        make sure the Ys must be dense numpy.ndarray
    """
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred.todense())
    count_pred = y_pred.sum(axis=1)
    intersection = np.bitwise_and(y_pred, y_true).sum(axis=1)
    count_pred[count_pred == 0] = 1 # add a pseudonumber to avoid zero division
    return (intersection/count_pred).mean(axis=0)

def coverage(y_true, y_pred):
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred.todense())
    count_true = y_true.sum(axis=1)
    intersection = np.bitwise_and(y_pred, y_true).sum(axis=1)
    return (intersection/count_true).mean(axis=0)

def accuracy_multilabel(y_true, y_pred):
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred.todense())
    intersection = np.bitwise_and(y_pred, y_true).sum(axis=1)
    union = np.bitwise_or(y_pred, y_true).sum(axis=1)
    return (intersection/union).mean(axis=0)
    
def absolute_true(y_true, y_pred):
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred.todense())
    row_equal = []
    for i in range(y_pred.shape[0]):
        row_equal.append(np.array_equal(y_pred[i,:], y_true[i,:]))
    return sum(row_equal)/y_pred.shape[0]

def absolute_false(y_true, y_pred):
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred.todense())
    return (np.bitwise_xor(y_true, y_pred).sum(axis=1) == y_pred.shape[1]).mean()
    

hamming_func = metrics.make_scorer(hamming_score, greater_is_better=False)
aiming_func = metrics.make_scorer(aiming, greater_is_better=True)
coverage_func = metrics.make_scorer(coverage, greater_is_better=True)
accuracy_func = metrics.make_scorer(accuracy_multilabel, greater_is_better=True)
absolute_true_func = metrics.make_scorer(absolute_true, greater_is_better=True)
absolute_false_func = metrics.make_scorer(absolute_false, greater_is_better=True)

def top1(model, X_test, Y_test):
    """
        get the top1 score for the multilable model
    """
    if isinstance(Y_test, pd.DataFrame):
        Y_test = Y_test.values
    sum_top1 = 0
    prob = model.predict_proba(X_test).todense()
    for i in range(Y_test.shape[0]):
        idx = int(prob.argmax(axis=1)[i])
        if(Y_test[i, idx] == 1):
            sum_top1 += 1
    
    top1_ = sum_top1/Y_test.shape[0]
    return top1_

def top3(model, X_test, Y_test):
    sum_top3 = 0
    prob = model.predict_proba(X_test).todense()
    if isinstance(Y_test, pd.DataFrame):
        Y_test = Y_test.values
    for i in range(Y_test.shape[0]):
        of_the = np.array((prob[0:,]).argsort()[:,-3:])
        if Y_test[i, of_the[i, :]].sum() >= 1:
            sum_top3 += 1
            
    top3_ = sum_top3/Y_test.shape[0]
    return top3_

def top2(model, X_test, Y_test):
    sum_top2 = 0
    prob = model.predict_proba(X_test).todense()
    if isinstance(Y_test, pd.DataFrame):
        Y_test = Y_test.values
    for i in range(Y_test.shape[0]):
        of_the = np.array((prob[0:,]).argsort()[:,-2:])
        if Y_test[i, of_the[i, :]].sum() >= 1:
            sum_top2 += 1
            
    top2_ = sum_top2/Y_test.shape[0]
    return top2_

def train_once(X_train, Y_train, X_val, Y_val, X_test, Y_test, model, parameters):
    """
        train the model for once on predefined trian, val, test
    """
    clf = GridSearch(model, parameters)
    clf.fit(X_train, Y_train, X_val, Y_val, scoring=hamming_func)
    loss = metrics.hamming_loss(clf.predict(X_test), Y_test)
    top1_ = top1(clf, X_test, Y_test)
    top2_ = top2(clf, X_test, Y_test)
    top3_ = top3(clf, X_test, Y_test)        
    model_dict[i] = (clf, clf.best_score_, loss, top1_, top2_, top3_)
    print(i, clf.best_params_, loss, top1_, top2_, top3_)



def train_seeds(X, Y, model, parameters, n_cv=5, test_size=0.15, use_cv=False):
    """
        train the model for random seeds; save model with crucial metrics 
    """
    MLkNN_dict = load("MLkNN_dict.joblib") # this dict with crucial random seeds
    model_dict = {}
    if not use_cv:
        for i in list(MLkNN_dict.keys()):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=i) 
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            clf = GridSearchCV(model, parameters, scoring=hamming_func, cv=n_cv, n_jobs=-1)
            clf.fit(X_train_scaled, Y_train.values)
            loss = metrics.hamming_loss(clf.predict(X_test_scaled), Y_test)
            top1_ = top1(clf, X_test_scaled, Y_test)
            top2_ = top2(clf, X_test_scaled, Y_test)
            top3_ = top3(clf, X_test_scaled, Y_test)        
            model_dict[i] = (clf, clf.best_score_, loss, top1_, top2_, top3_)
            print(i, clf.best_params_, loss, top1_, top2_, top3_)
    else:
        for i in list(MLkNN_dict.keys()):
            kfold = KFold(n_cv, random_state=i, shuffle=True)
            loss = 0
            top1_ = 0
            top2_ = 0
            top3_ = 0
        
            for k, (train, test) in enumerate(kfold.split(X, Y)):

                scaler = MinMaxScaler()
                X_train_scaled = scaler.fit_transform(X.values[train])
                X_test_scaled = scaler.transform(X.values[test])
            
                clf = GridSearchCV(model, parameters, scoring=hamming_func, cv=n_cv, n_jobs=-1)
                clf.fit(X_train_scaled, Y.values[train])
                loss += metrics.hamming_loss(clf.predict(X_test_scaled), Y.values[test])
                top1_ += top1(clf, X_test_scaled, Y.values[test])
                top2_ += top2(clf, X_test_scaled, Y.values[test])
                top3_ += top3(clf, X_test_scaled, Y.values[test])
            
            
            top1_ /= n_cv
            top2_ /= n_cv
            top3_ /= n_cv
            loss /= n_cv
        
            model_dict[i] = (clf, clf.best_score_, loss, top1_, top2_, top3_)
            print(i, clf.best_params_, loss, top1_, top2_, top3_)
    return model_dict

def train_seeds_hm(X, Y, model, parameters, n_cv=5, test_size=0.15, use_cv=False):
    """
        train the model for random seeds; save model with crucial metrics 
    """
    MLkNN_dict = load("MLkNN_dict.joblib") # this dict with crucial random seeds
    model_dict = {}
    if not use_cv:
        for i in list(MLkNN_dict.keys()):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=i) 
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            clf = GridSearchCV(model, parameters, scoring=hamming_func, cv=n_cv, n_jobs=-1)
            clf.fit(X_train_scaled, Y_train.values)
            loss = metrics.hamming_loss(clf.predict(X_test_scaled), Y_test)
            #top1_ = top1(clf, X_test_scaled, Y_test)
            #top2_ = top2(clf, X_test_scaled, Y_test)
            #top3_ = top3(clf, X_test_scaled, Y_test)        
            model_dict[i] = (clf, clf.best_score_, loss,) #top1_, top2_, top3_)
            print(i, clf.best_params_, loss,) #top1_, top2_, top3_)
    else:
        for i in list(MLkNN_dict.keys()):
            kfold = KFold(n_cv, random_state=i, shuffle=True)
            loss = 0
            top1_ = 0
            top2_ = 0
            top3_ = 0
        
            for k, (train, test) in enumerate(kfold.split(X, Y)):

                scaler = MinMaxScaler()
                X_train_scaled = scaler.fit_transform(X.values[train])
                X_test_scaled = scaler.transform(X.values[test])
            
                clf = GridSearchCV(model, parameters, scoring=hamming_func, cv=n_cv, n_jobs=-1)
                clf.fit(X_train_scaled, Y.values[train])
                loss += metrics.hamming_loss(clf.predict(X_test_scaled), Y.values[test])
                #top1_ += top1(clf, X_test_scaled, Y.values[test])
                #top2_ += top2(clf, X_test_scaled, Y.values[test])
                #top3_ += top3(clf, X_test_scaled, Y.values[test])
            
            
            #top1_ /= n_cv
            #top2_ /= n_cv
            #top3_ /= n_cv
            loss /= n_cv
        
            model_dict[i] = (clf, clf.best_score_, loss,) #top1_, top2_, top3_)
            print(i, clf.best_params_, loss,) #top1_, top2_, top3_)
    return model_dict



def get_seeds_on_top1(model_dict):
    top1_list = []
    seed_list = []
    for i in model_dict:
        top1_list.append(model_dict[i][3])
        seed_list.append(i)
    top1_array = np.array(top1_list)
    seed_array = np.array(seed_list)
    my_seeds = seed_array[top1_array.argsort()[-10:]]
    print("Mean top1: {0:.4f}".format(top1_array[top1_array.argsort()[-10:]].mean()))
    train_ = []
    top2_ = []
    top3_ = []
    hm_ = []
    for i in my_seeds:
        train_.append(model_dict[i][1])
        top2_.append(model_dict[i][4])
        top3_.append(model_dict[i][5])
        hm_.append(model_dict[i][2])
    top2_array = np.array(top2_)
    top3_array = np.array(top3_)
    hm_array = np.array(hm_)
    train_ = np.array(train_)
    print("Mean top2: {0:.4f}".format(top2_array.mean())) 
    print("Mean top3: {0:.4f}".format(top3_array.mean())) 
    print("Mean humming loss: {0:.4f}".format(hm_array.mean()))
    return top1_array[top1_array.argsort()[-10:]], top2_array, top3_array, hm_array, train_, my_seeds

def get_seeds_on_hm(model_dict):
    hm_list = []
    seed_list = []
    for i in model_dict:
        hm_list.append(model_dict[i][2])
        seed_list.append(i)
    hm_array = np.array(hm_list)
    seed_array = np.array(seed_list)
    my_seeds = seed_array[hm_array.argsort()[:10]]
    print("Mean hm: {0:.4f}".format(hm_array[hm_array.argsort()[:10]].mean()))
    # top2_ = []
    # top3_ = []
    hm_ = []
    train_ = []
    for i in my_seeds:
        # top2_.append(model_dict[i][4])
        # top3_.append(model_dict[i][5])
        hm_.append(model_dict[i][2])
        train_.append(model_dict[i][1])
    # top2_array = np.array(top2_)
    # top3_array = np.array(top3_)
    train_ = np.array(train_)
    hm_array = np.array(hm_)
    # print("Mean top2: {0:.4f}".format(top2_array.mean())) 
    # print("Mean top3: {0:.4f}".format(top3_array.mean())) 
    print("Mean humming loss: {0:.4f}".format(hm_array.mean()))
    return hm_array, train_, seed_array


def get_result_on_seeds(model_dict, chosen_seeds):
    top1_list = []
    top2_list = []
    top3_list = []
    hm_list = []
    for i in model_dict:
        if i in chosen_seeds:
            hm_list.append(model_dict[i][2])
            top1_list.append(model_dict[i][3])
            top2_list.append(model_dict[i][4])
            top3_list.append(model_dict[i][5])
    top1_array = np.array(top1_list)
    top2_array = np.array(top2_list)
    top3_array = np.array(top3_list)
    hm_array = np.array(hm_list)
    print("Mean top1: {0:.4f}".format(top1_array.mean()))
    print("Mean top2: {0:.4f}".format(top2_array.mean())) 
    print("Mean top3: {0:.4f}".format(top3_array.mean())) 
    print("Mean humming loss: {0:.4f}".format(hm_array.mean()))
    return top1_array, top2_array, top3_array, hm_array

def get_X_Y():
    feature_physiochemical = pd.read_csv("feature_physiochemical.csv", index_col=0)
    feature_ECFP = pd.read_csv("feature_ecfp.csv", index_col=0)
    feature_MACCS = pd.read_csv("feature_maccs.csv", index_col=0)
    feature_ECFP.index = feature_physiochemical.index
    feature_MACCS.index = feature_physiochemical.index
    loc_y = feature_physiochemical.columns.get_loc('Class1A2')
    Y = feature_physiochemical.iloc[:, loc_y:]
    feature_ECFP.drop(columns=Y.columns, inplace=True)
    feature_MACCS.drop(columns=Y.columns, inplace=True)
    feature_physiochemical.drop(columns=Y.columns, inplace=True)
    X = pd.concat([feature_physiochemical, feature_ECFP, feature_MACCS], axis=1)
    
    return X, Y



"""
    visualization usage
"""
def plot_roc(model, X_test, y_test):
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def plot_chemical_diversity(X, ):
    pass



"""
    report usage
"""
def report_classification(model, X_test, y_test):
    """
        we got: precision, recall, f1, kappa, mcc
    """
    result = metrics.precision_recall_fscore_support(y_test, model.predict(X_test))
    precision = result[0][1]
    recall = result[1][1]
    f1 = result[2][1]
    kappa = metrics.cohen_kappa_score(y_test, model.predict(X_test))
    mcc = metics.matthews_corrcoef(y_test, model.predict(X_test))
    return "{},{},{},{},{},{}\n".format(type(model).__name, precision, recall, f1, kappa, mcc)