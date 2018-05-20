import math
from sklearn import svm
#This file needs to be imported into basic_svm.py for execution

#------------------Covarience Optimizations-------------------
def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)
        
def optimize(df):
    for _ in df:
        corr = pearson_def(df[_],df['y'])
        if (corr > -0.05 and corr < 0.05):
            df.drop([_],1,inplace=True)
            print ("Deleted: " + str(corr))
        else:
            print (corr)
    return df

#---------------------SVM Enhancements---------------------------


def modify(clf,x): #Use x as option to select change
    return {
        
        # KERNEL optimised to linear
        
        '1': return (svm.SVC(C=1.0, kernel=’linear’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)),
        
        # KERNEL optimised to Sigmoidal
        
        '2': return (svm.SVC(C=1.0, kernel=’sigmoid’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)),
        
        # KERNEL changed to poly
        
        '3': return (svm.SVC(C=1.0, kernel=’poly’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)),
        
        #probabalistic model
        
        '4': return (svm.SVC(C=1.0, kernel=’linear’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True, probability=True, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)),
        
        #removed SHRINKING heurestic
        
        '5': return (svm.SVC(C=1.0, kernel=’linear’, degree=3, gamma=’auto’, coef0=0.0, shrinking=False, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)),
        
        #decision function shape
        
        '6': return (svm.SVC(C=1.0, kernel=’linear’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovo’, random_state=None)),
        
        #tolerance change
        
        '7': return (svm.SVC(C=1.0, kernel=’linear’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True, probability=False, tol=0.01, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)),
        
        
    }.get(x, return (svm.SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)))
#Default option; no changes

#----------------------------Extended dataset-----------------------------------

def dataextend():
    return pd.read_csv('bax.csv')

#--------------------------------------------------------------------------------
    
