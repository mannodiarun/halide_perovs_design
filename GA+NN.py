import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pandas import read_csv
import sklearn
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

##  ML Model Specific Packages  ##
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, ExpSineSquared
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
import tensorflow.keras as keras
#import keras
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy as np     
import csv 
import copy 
import random 
#import mlpy
import pandas
import matplotlib.pyplot as plt 
#from mlpy import KernelRidge                                                                                                                                  
from sklearn.preprocessing import normalize

#from geneticalgorithm import geneticalgorithm as ga
from geneticalgorithm2 import geneticalgorithm2 as ga # for creating and running optimization model
from geneticalgorithm2 import Crossover, Mutations, Selection # classes for specific mutation and crossover behavior
from geneticalgorithm2 import Population_initializer # for creating better start population
from geneticalgorithm2 import np_lru_cache # for cache function (if u want)
from geneticalgorithm2 import plot_pop_scores # for plotting population scores, if u want
from geneticalgorithm2 import Callbacks # simple callbacks
from geneticalgorithm2 import Actions, ActionConditions, MiddleCallbacks # middle callbacks



##  Choose Dataset Type  ##
Data = 'PBE'
#Data = 'HSE'

##  Choose Descriptor Type  ##
Desc = 'Composition Only'
#Desc = 'Elemental Only'
#Desc = 'Composition and Elemental'

##  Properties = Lattice Constant, Band Gap, Decomposition Energy, PV Figure of Merit; all trained together, nothing to choose for now  ##

##  ML Model is fixed to be Neural Network (NN) Regression here  ##


###  Read Data  ###
if Data == 'PBE':
    ifile  = open('PBE_data.csv', "rt")
    reader = csv.reader(ifile)
    csvdata=[]
    for row in reader:
        csvdata.append(row)   
    ifile.close()
    numrow=len(csvdata)
    numcol=len(csvdata[0]) 
    csvdata = np.array(csvdata).reshape(numrow,numcol)
    Index = csvdata[:,0]
    Formula = csvdata[:,0]
    Mixing = csvdata[:,1]
    PBE_latt = csvdata[:,2]
    PBE_gap  = csvdata[:,3]
    PBE_form_en = csvdata[:,4]
    PBE_decomp_en = csvdata[:,5]
    PBE_eps  = csvdata[:,6]
    PBE_fom  = csvdata[:,7]
    Comp_desc = csvdata[:,8:22]
    Elem_desc = csvdata[:,22:]
    if Desc == 'Composition Only':
        X = csvdata[0:514,8:22]
        Formula = csvdata[0:514,0]
        PBE_latt = csvdata[0:514,2]
        PBE_gap  = csvdata[0:514,3]
        PBE_form_en = csvdata[0:514,4]
        PBE_decomp_en = csvdata[0:514,5]
        PBE_eps  = csvdata[0:514,6]
        PBE_fom  = csvdata[0:514,7]
    if Desc == 'Elemental Only':
        X = copy.deepcopy(Elem_desc)
    if Desc == 'Composition and Elemental':
        X = csvdata[0:514,8:22]
        Formula = csvdata[0:514,0]
        PBE_latt = csvdata[0:514,2]
        PBE_gap  = csvdata[0:514,3]
        PBE_form_en = csvdata[0:514,4]
        PBE_decomp_en = csvdata[0:514,5]
        PBE_eps  = csvdata[0:514,6]
        PBE_fom  = csvdata[0:514,7]

if Data == 'HSE':
    ifile  = open('HSE_data.csv', "rt")
    reader = csv.reader(ifile)
    csvdata=[]
    for row in reader:
        csvdata.append(row)
    ifile.close()
    numrow=len(csvdata)
    numcol=len(csvdata[0])
    csvdata = np.array(csvdata).reshape(numrow,numcol)
    Index = csvdata[:,0]
    Formula = csvdata[:,0]
    Mixing = csvdata[:,1]
    HSE_latt = csvdata[:,2]
    HSE_gap  = csvdata[:,3]
    HSE_form_en = csvdata[:,4]
    HSE_decomp_en = csvdata[:,5]
    Comp_desc = csvdata[:,6:20]
    Elem_desc = csvdata[:,20:]
    if Desc == 'Composition Only':
        X = copy.deepcopy(Comp_desc)
    if Desc == 'Elemental Only':
        X = copy.deepcopy(Elem_desc)
    if Desc == 'Composition and Elemental':
        X = csvdata[:,6:]

n = Formula.size
m = int(X.size/n)
##  If we want to use the float or normalized versison of X  ##
#X_fl = np.array(X, dtype="float32")
#X_norm = normalize(X_fl, norm='l2', axis=0)
#X = copy.deepcopy(X_norm)



###  Training-Test Split  ###
t = 0.20
if Data == 'PBE':
    X_train, X_test, Prop_train_gap, Prop_test_gap, Prop_train_decomp, Prop_test_decomp, Prop_train_fom, Prop_test_fom = train_test_split(X, PBE_gap, PBE_decomp_en, PBE_fom, test_size=t)
if Data == 'HSE':
    X_train, X_test, Prop_train_gap, Prop_test_gap, Prop_train_decomp, Prop_test_decomp = train_test_split(X, HSE_gap, HSE_decomp_en, test_size=t)
n_tr = Prop_train_gap.size
n_te = Prop_test_gap.size
X_train_fl = np.array(X_train, dtype="float32")
X_test_fl = np.array(X_test, dtype="float32")



###  NN Optimizers and Model Definition  ###
pipelines = []
parameters = [[0.0 for a in range(6)] for b in range(729)]
dp = [0.00, 0.10, 0.20]
n1 = [50, 75, 100]
n2 = [50, 75, 100]
lr = [0.001, 0.01, 0.1]
ep = [200, 400, 600]
bs = [50, 100, 200]
count = 0
for a in range(0,3):
    for b in range(0,3):
        for c in range(0,3):
            for d in range(0,3):
                for e in range(0,3):
                    for f in range(0,3):
                        parameters[count][0] = lr[a]
                        parameters[count][1] = n1[b]
                        parameters[count][2] = dp[c]
                        parameters[count][3] = n2[d]
                        parameters[count][4] = ep[e]
                        parameters[count][5] = bs[f]
                        count = count+1                        
                        keras.optimizers.Adam(learning_rate=lr[a], beta_1=0.9, beta_2=0.999, amsgrad=False)
                        # define base model
                        def baseline_model():
                            model = Sequential()
                            model.add(Dense(m, input_dim=m, kernel_initializer='normal', activation='relu'))
                            model.add(Dense(n1[b], kernel_initializer='normal', activation='relu'))
                            model.add(Dropout(dp[c], input_shape=(m,)))
                            model.add(Dense(n2[d], kernel_initializer='normal', activation='relu'))
                            model.add(Dense(1, kernel_initializer='normal'))
                            model.compile(loss='mean_squared_error', optimizer='Adam')
                            return model
                        # evaluate model with standardized dataset
                        estimators = []
#                        estimators.append(('standardize', StandardScaler()))
                        estimators.append(('scaler', StandardScaler()))
                        estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=ep[e], batch_size=bs[f], verbose=0)))
                        pipelines.append ( Pipeline(estimators) )
times = 1
#times = len(pipelines)


###  Train Model For Band Gap  ###
train_errors = [0.0]*times
test_errors = [0.0]*times
nn_errors = list()
n_fold = 5
Prop_train_temp = np.array(Prop_train_gap, dtype="float32")
Prop_test_temp = np.array(Prop_test_gap, dtype="float32")
for i in range(0,times):
    pipeline = pipelines[np.random.randint(0,729)]
#    pipeline = pipelines[i]
    kf = KFold(n_splits = n_fold)
    mse_test_cv = 0.00
    mse_train_cv = 0.00
    for train, test in kf.split(X_train):
        X_train_cv, X_test_cv, Prop_train_cv, Prop_test_cv = X_train[train], X_train[test], Prop_train_temp[train], Prop_train_temp[test]
        X_train_cv_fl = np.array(X_train_cv, dtype="float32")
        Prop_train_cv_fl = np.array(Prop_train_cv, dtype="float32")
        X_test_cv_fl = np.array(X_test_cv, dtype="float32")
        Prop_test_cv_fl = np.array(Prop_test_cv, dtype="float32")
        pipeline.fit(X_train_cv_fl, Prop_train_cv_fl)
        Prop_pred_train_cv = pipeline.predict(X_train_cv_fl)
        Prop_pred_test_cv  = pipeline.predict(X_test_cv_fl)
        Pred_train_cv_fl = np.array(Prop_pred_train_cv, dtype="float32")
        Pred_test_cv_fl = np.array(Prop_pred_test_cv, dtype="float32")
        mse_test_cv = mse_test_cv  + sklearn.metrics.mean_squared_error(Prop_test_cv_fl, Pred_test_cv_fl)
        mse_train_cv = mse_train_cv + sklearn.metrics.mean_squared_error(Prop_train_cv_fl, Pred_train_cv_fl)
    mse_test = mse_test_cv / n_fold
    mse_train = mse_train_cv / n_fold
    train_errors[i] = mse_train
    test_errors[i] = mse_test
    nn_errors.append(pipeline)
i_opt = np.argmin(test_errors)
pipeline_opt = nn_errors[i_opt]
train_errors_gap = copy.deepcopy(train_errors)
test_errors_gap  = copy.deepcopy(test_errors)
pipeline_opt.fit(X_train_fl, Prop_train_temp)
Pred_train = pipeline_opt.predict(X_train)
Pred_test  = pipeline_opt.predict(X_test)
Prop_train_gap_fl = copy.deepcopy(Prop_train_temp)
Pred_train_gap_fl = np.array(Pred_train, dtype="float32")
Prop_test_gap_fl = copy.deepcopy(Prop_test_temp)
Pred_test_gap_fl = np.array(Pred_test, dtype="float32")
## Outside Predictions ##
#Pred_out = pipeline_opt.predict(X_out_fl)
#Pred_out_gap = np.array(Pred_out, dtype="float32")
pipeline_gap = pipeline_opt



###  Train Model For Decomposition Energy ##
train_errors = [0.0]*times
test_errors = [0.0]*times
nn_errors = list()
n_fold = 5
Prop_train_temp = np.array(Prop_train_decomp, dtype="float32")
Prop_test_temp = np.array(Prop_test_decomp, dtype="float32")
for i in range(0,times):
    pipeline = pipelines[np.random.randint(0,729)]
#    pipeline = pipelines[i]
    kf = KFold(n_splits = n_fold)
    mse_test_cv = 0.00
    mse_train_cv = 0.00
    for train, test in kf.split(X_train):
        X_train_cv, X_test_cv, Prop_train_cv, Prop_test_cv = X_train[train], X_train[test], Prop_train_temp[train], Prop_train_temp[test]
        X_train_cv_fl = np.array(X_train_cv, dtype="float32")
        Prop_train_cv_fl = np.array(Prop_train_cv, dtype="float32")
        X_test_cv_fl = np.array(X_test_cv, dtype="float32")
        Prop_test_cv_fl = np.array(Prop_test_cv, dtype="float32")
        pipeline.fit(X_train_cv_fl, Prop_train_cv_fl)
        Prop_pred_train_cv = pipeline.predict(X_train_cv_fl)
        Prop_pred_test_cv  = pipeline.predict(X_test_cv_fl)
        Pred_train_cv_fl = np.array(Prop_pred_train_cv, dtype="float32")
        Pred_test_cv_fl = np.array(Prop_pred_test_cv, dtype="float32")
        mse_test_cv = mse_test_cv  + sklearn.metrics.mean_squared_error(Prop_test_cv_fl, Pred_test_cv_fl)
        mse_train_cv = mse_train_cv + sklearn.metrics.mean_squared_error(Prop_train_cv_fl, Pred_train_cv_fl)
    mse_test = mse_test_cv / n_fold
    mse_train = mse_train_cv / n_fold
    train_errors[i] = mse_train
    test_errors[i] = mse_test
    nn_errors.append(pipeline)
i_opt = np.argmin(test_errors)
pipeline_opt = nn_errors[i_opt]
train_errors_decomp = copy.deepcopy(train_errors)
test_errors_decomp  = copy.deepcopy(test_errors)
pipeline_opt.fit(X_train_fl, Prop_train_temp)
Pred_train = pipeline_opt.predict(X_train)
Pred_test  = pipeline_opt.predict(X_test)
Prop_train_decomp_fl = copy.deepcopy(Prop_train_temp)
Pred_train_decomp_fl = np.array(Pred_train, dtype="float32")
Prop_test_decomp_fl = copy.deepcopy(Prop_test_temp)
Pred_test_decomp_fl = np.array(Pred_test, dtype="float32")
## Outside Predictions ##
#Pred_out = pipeline_opt.predict(X_out_fl)
#Pred_out_decomp = np.array(Pred_out, dtype="float32")
pipeline_decomp = pipeline_opt



###  Train Model For Photovoltaic Figure of Merit (FOM)  ###
if Data == 'PBE':
    X = csvdata[0:514,8:22]
    PBE_fom  = csvdata[0:514,7]
    X_train_temp, X_test_temp, Prop_train_fom, Prop_test_fom = train_test_split(X, PBE_fom, test_size=t)
    X_train_temp_fl = np.array(X_train_temp, dtype="float32")
    X_test_temp_fl = np.array(X_test_temp, dtype="float32")
    train_errors = [0.0]*times
    test_errors = [0.0]*times
    nn_errors = list()
    n_fold = 5
    Prop_train_temp = np.array(Prop_train_fom, dtype="float32")
    Prop_test_temp = np.array(Prop_test_fom, dtype="float32")
    for i in range(0,times):
        pipeline = pipelines[np.random.randint(0,729)]
#        pipeline = pipelines[i]
        kf = KFold(n_splits = n_fold)
        mse_test_cv = 0.00
        mse_train_cv = 0.00
        for train, test in kf.split(X_train_temp):
            X_train_cv, X_test_cv, Prop_train_cv, Prop_test_cv = X_train_temp[train], X_train_temp[test], Prop_train_temp[train], Prop_train_temp[test]
            X_train_cv_fl = np.array(X_train_cv, dtype="float32")
            Prop_train_cv_fl = np.array(Prop_train_cv, dtype="float32")
            X_test_cv_fl = np.array(X_test_cv, dtype="float32")
            Prop_test_cv_fl = np.array(Prop_test_cv, dtype="float32")
            pipeline.fit(X_train_cv_fl, Prop_train_cv_fl)
            Prop_pred_train_cv = pipeline.predict(X_train_cv_fl)
            Prop_pred_test_cv  = pipeline.predict(X_test_cv_fl)
            Pred_train_cv_fl = np.array(Prop_pred_train_cv, dtype="float32")
            Pred_test_cv_fl = np.array(Prop_pred_test_cv, dtype="float32")
            mse_test_cv = mse_test_cv  + sklearn.metrics.mean_squared_error(Prop_test_cv_fl, Pred_test_cv_fl)
            mse_train_cv = mse_train_cv + sklearn.metrics.mean_squared_error(Prop_train_cv_fl, Pred_train_cv_fl)
        mse_test = mse_test_cv / n_fold
        mse_train = mse_train_cv / n_fold
        train_errors[i] = mse_train
        test_errors[i] = mse_test
        nn_errors.append(pipeline)
    i_opt = np.argmin(test_errors)
    pipeline_opt = nn_errors[i_opt]
    train_errors_fom = copy.deepcopy(train_errors)
    test_errors_fom  = copy.deepcopy(test_errors)
    pipeline_opt.fit(X_train_temp_fl, Prop_train_temp)
    Pred_train = pipeline_opt.predict(X_train_temp)
    Pred_test  = pipeline_opt.predict(X_test_temp)
    Prop_train_fom_fl = copy.deepcopy(Prop_train_temp)
    Pred_train_fom_fl = np.array(Pred_train, dtype="float32")
    Prop_test_fom_fl = copy.deepcopy(Prop_test_temp)
    Pred_test_fom_fl = np.array(Pred_test, dtype="float32")
## Outside Predictions ##
#Pred_out = pipeline_opt.predict(X_out_fl)
#Pred_out_fom = np.array(Pred_out, dtype="float32")
    pipeline_fom = pipeline_opt


##  Calculate RMSE Values  ##

mse_test_prop = sklearn.metrics.mean_squared_error(Prop_test_decomp_fl, Pred_test_decomp_fl)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_decomp_fl, Pred_train_decomp_fl)
rmse_test_decomp = np.sqrt(mse_test_prop)
rmse_train_decomp = np.sqrt(mse_train_prop)
print('rmse_test_decomp = ', np.sqrt(mse_test_prop))
print('rmse_train_decomp = ', np.sqrt(mse_train_prop))
print('      ')

mse_test_prop = sklearn.metrics.mean_squared_error(Prop_test_gap_fl, Pred_test_gap_fl)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_gap_fl, Pred_train_gap_fl)
rmse_test_gap = np.sqrt(mse_test_prop)
rmse_train_gap = np.sqrt(mse_train_prop)
print('rmse_test_gap = ', np.sqrt(mse_test_prop))
print('rmse_train_gap = ', np.sqrt(mse_train_prop))
print('      ')

if Data == 'PBE':
    mse_test_prop = sklearn.metrics.mean_squared_error(Prop_test_fom_fl, Pred_test_fom_fl)
    mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_fom_fl, Pred_train_fom_fl)
    rmse_test_fom = np.sqrt(mse_test_prop)
    rmse_train_fom = np.sqrt(mse_train_prop)
    print('rmse_test_fom = ', np.sqrt(mse_test_prop))
    print('rmse_train_fom = ', np.sqrt(mse_train_prop))
    print('      ')





###  GA portion begins here  ###


x1 = [0.0]*65
x2 = [0.0]*65
for i in range(0,65):
    x1[i] = i/64
    x2[i] = 3*i/64
varbound = np.array([[0,64], [0,64], [0,64], [0,64], [0,64], [0,64], [0,64], [0,64], [0,64], [0,64], [0,64], [0,64], [0,64], [0,64] ] )

def f(XX):
    X = [0.0]*14
    for i in range(0,11):
        X[i] = x1[int(XX[i])]
    for i in range(11,14):
        X[i] = x2[int(XX[i])]
    pen = 0
    if np.abs(np.sum(X[0:5]) - 1.0) > 0.02:
        pen = pen + 1000*np.abs(1-np.sum(X[0:5]))
    if np.abs(np.sum(X[5:11]) - 1.0) > 0.02:
        pen = pen + 1000*np.abs(1-np.sum(X[5:11]))
    if np.abs(np.sum(X[11:14]) - 3.0) > 0.02:
        pen = pen + 1000*np.abs(3-np.sum(X[11:14]))
    pred_decomp = float(pipeline_decomp.predict( np.reshape(X, (1, -1)) ))
    pred_gap = float(pipeline_gap.predict( np.reshape(X, (1, -1)) ))
    if Data == 'PBE':
        pred_fom = -1*float(pipeline_fom.predict( np.reshape(X, (1, -1)) ))
        if pred_gap < 1.0:
            pen = pen + 1000*np.abs(pred_gap - 1)
        if pred_gap > 2.5:
            pen = pen + 1000*np.abs(pred_gap - 2.5)
        if pred_decomp > -0.5:
            pen = pen + 1000*np.abs(pred_decomp + 0.5)
    if Data == 'HSE':
        pred_fom = 0.0
        if pred_gap < 1.0:
            pen = pen + 1000*np.abs(pred_gap - 1)
        if pred_gap > 3.0:
            pen = pen + 1000*np.abs(pred_gap - 3)
        if pred_decomp > -0.5:
            pen = pen + 1000*np.abs(pred_decomp + 0.5)
    return pred_fom + pen

algorithm_param = {'max_num_iteration': 100,
                   'population_size':100,
                   'mutation_probability':0.1,
                   'elit_ratio': 0.01,  
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'mutation_type': 'uniform_by_center',
                   'selection_type': 'roulette',
                   'max_iteration_without_improv':None}

model = ga(function=f, dimension=14, variable_type='int', variable_boundaries=varbound, variable_type_mixed = None, function_timeout = 10, algorithm_parameters=algorithm_param)

model.run(
    no_plot = False, 
    disable_progress_bar = False,
    disable_printing = False,

    set_function = None, 
    apply_function_to_parents = False, 
    start_generation = {'variables':None, 'scores': None},
    studEA = False,
    mutation_indexes = None,

    init_creator = None,
    init_oppositors = None,
    duplicates_oppositor = None,
    remove_duplicates_generation_step = None,
    revolution_oppositor = None,
    revolution_after_stagnation_step = None,
    revolution_part = 0.3,

    population_initializer = Population_initializer(select_best_of = 10, local_optimization_step = 'never', local_optimizer = None),

    stop_when_reached = None,
    callbacks = [],
    middle_callbacks = [],
#    middle_callbacks = [MiddleCallbacks.UniversalCallback(action, ActionConditions.EachGen(generation_step = 1))],
    time_limit_secs = None, 
#    save_last_generation_as = None,
    seed = None,
#    save_last_generation_as = 'last_gen'
    )


X_best = [0.0]*14
for i in range(0,11):
    X_best[i] = float( x1[int(model.best_variable[i])] )
for i in range(11,14):
    X_best[i] = float( x2[int(model.best_variable[i])] )

print ('best_decomp_energy = ', float ( pipeline_decomp.predict( np.reshape(X_best[:], (1,-1)) ) ))
print ('best_band_gap = ', float ( pipeline_gap.predict( np.reshape(X_best[:], (1,-1)) ) ))
if Data == 'PBE':
    print ('best_PV_FOM = ', float ( pipeline_fom.predict( np.reshape(X_best[:], (1,-1)) ) ))

#print ('best_decomp_energy = ', float(pipeline_decomp.predict( np.reshape(model.best_variable, (1, -1)) )) )
#print ('best_PV_FOM = ', float(pipeline_fom.predict( np.reshape(model.best_variable, (1, -1)) )) )
#print ('best_band_gap = ', float(pipeline_gap.predict( np.reshape(model.best_variable, (1, -1)) )) )

xx = model.output_dict.get('last_generation')
zz = xx.get('variables')[0]/64
for i in range(11,14):
    zz[i] = zz[i]*3
formula_best_perovs = [' ']
atoms = ['K', 'Rb', 'Cs', 'MA', 'FA', 'Ca', 'Sr', 'Ba', 'Ge', 'Sn', 'Pb', 'I', 'Br', 'Cl']
for i in range(0,14):
    if float(zz[i]) > 0.004:
        ii = '%.2f' % float(zz[i])
        formula_best_perovs[0] = formula_best_perovs[0] + atoms[i] + '_' + str(ii) + ' '

print ('best_perovs_alloy = ', formula_best_perovs)



text_file = open("results_GA+NN.txt", "w")
text_file.write("rmse_test_decomp: %s" % np.sqrt( sklearn.metrics.mean_squared_error(Prop_test_decomp_fl, Pred_test_decomp_fl) ) + '\n')
text_file.write("rmse_train_decomp: %s" % np.sqrt( sklearn.metrics.mean_squared_error(Prop_train_decomp_fl, Pred_train_decomp_fl) ) + '\n')
text_file.write("rmse_test_gap: %s" % np.sqrt( sklearn.metrics.mean_squared_error(Prop_test_gap_fl, Pred_test_gap_fl) ) + '\n')
text_file.write("rmse_train_gap: %s" % np.sqrt( sklearn.metrics.mean_squared_error(Prop_train_gap_fl, Pred_train_gap_fl) ) + '\n')
if Data == 'PBE':
    text_file.write("rmse_test_fom: %s" % np.sqrt( sklearn.metrics.mean_squared_error(Prop_test_fom_fl, Pred_test_fom_fl) ) + '\n')
    text_file.write("rmse_train_fom: %s" % np.sqrt( sklearn.metrics.mean_squared_error(Prop_train_fom_fl, Pred_train_fom_fl) ) + '\n')
text_file.write(":: %s" % '  ' + '\n')
text_file.write("best_decomp_energy: %s" % float(pipeline_decomp.predict( np.reshape(X_best[:], (1, -1)) )) + ' eV' + '\n')
text_file.write("best_band_gap: %s" % float(pipeline_gap.predict( np.reshape(X_best[:], (1, -1)) )) + ' eV' + '\n')
if Data == 'PBE':
    text_file.write("best_PV_FOM: %s" % float(pipeline_fom.predict( np.reshape(X_best[:], (1, -1)) )) + '\n')
text_file.write("best_perovs_alloy: %s" % formula_best_perovs + '\n')
text_file.close()

np.savetxt('scores_GA+NN.txt', model.report)




# plot and save optimization process plot
model.plot_results(save_as = 'plot_scores_process_GA+NN.png')

# plot scores of last population
model.plot_generation_scores(title = 'Population scores after ending of searching', save_as= 'plot_scores_end_GA+NN.png')


