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

##  ML Model is fixed to be Random Forest Regression (RFR) here  ##


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
        PBE_latt = csvdata[0:514,2]
        PBE_gap  = csvdata[0:514,3]
        PBE_form_en = csvdata[0:514,4]
        PBE_decomp_en = csvdata[0:514,5]
        PBE_eps  = csvdata[0:514,6]
        PBE_fom  = csvdata[0:514,7]
    if Desc == 'Elemental Only':
        X = copy.deepcopy(Elem_desc)
    if Desc == 'Composition and Elemental':
        X = csvdata[:,8:]

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
    X_train, X_test, Prop_gap_train, Prop_gap_test, Prop_decomp_train, Prop_decomp_test, Prop_fom_train, Prop_fom_test = train_test_split(X, PBE_gap, PBE_decomp_en, PBE_fom, test_size=t)
if Data == 'HSE':
    X_train, X_test, Prop_gap_train, Prop_gap_test, Prop_decomp_train, Prop_decomp_test = train_test_split(X, HSE_gap, HSE_decomp_en, test_size=t)
n_tr = Prop_gap_train.size
n_te = Prop_gap_test.size
X_train_fl = np.array(X_train, dtype="float32")
Prop_decomp_train_fl = np.array(Prop_decomp_train, dtype="float32")
Prop_gap_train_fl = np.array(Prop_gap_train, dtype="float32")
X_test_fl = np.array(X_test, dtype="float32")
Prop_decomp_test_fl = np.array(Prop_decomp_test, dtype="float32")
Prop_gap_test_fl = np.array(Prop_gap_test, dtype="float32")
if Data == 'PBE':
    Prop_fom_train_fl = np.array(Prop_fom_train, dtype="float32")
    Prop_fom_test_fl = np.array(Prop_fom_test, dtype="float32")


###  Define Random Forest Hyperparameter Space  ###
#param_grid = {
#"n_estimators": [100, 200, 500],
#"max_features": [6, 10, m],
#"min_samples_leaf": [5,10,20],
#"max_depth": [5,10,15],
#"min_samples_split": [2, 5, 10]
#}
param_grid = { "n_estimators": [100, 200], "max_depth": [10,15], "min_samples_split": [5, 10] }


#  Train Model For Decomposition Energy ##
rfreg_decomp = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=5)
rfreg_decomp.fit(X_train_fl, Prop_decomp_train_fl)
Pred_train = rfreg_decomp.predict(X_train_fl)
Pred_test = rfreg_decomp.predict(X_test_fl)
Pred_out = rfreg_decomp.predict(X_out_fl)
Pred_decomp_train_fl = np.array(Pred_train, dtype="float32")
Pred_decomp_test_fl = np.array(Pred_test, dtype="float32")
Pred_decomp_out_fl = np.array(Pred_out, dtype="float32")
print('rmse_test_decomp = ', np.sqrt( sklearn.metrics.mean_squared_error(Prop_decomp_test_fl, Pred_decomp_test_fl) ))
print('rmse_train_decomp = ', np.sqrt( sklearn.metrics.mean_squared_error(Prop_decomp_train_fl, Pred_decomp_train_fl) ))

#  Train Model For Band Gap ##
rfreg_gap = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=5)
rfreg_gap.fit(X_train_fl, Prop_gap_train_fl)
Pred_train = rfreg_gap.predict(X_train_fl)
Pred_test = rfreg_gap.predict(X_test_fl)
Pred_out = rfreg_gap.predict(X_out_fl)
Pred_gap_train_fl = np.array(Pred_train, dtype="float32")
Pred_gap_test_fl = np.array(Pred_test, dtype="float32")
Pred_gap_out_fl = np.array(Pred_out, dtype="float32")
print('rmse_test_gap = ', np.sqrt( sklearn.metrics.mean_squared_error(Prop_gap_test_fl, Pred_gap_test_fl) ))
print('rmse_train_gap = ', np.sqrt( sklearn.metrics.mean_squared_error(Prop_gap_train_fl, Pred_gap_train_fl) ))

#  Train Model For PV FOM ##
if Data == 'PBE':
    rfreg_fom = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=5)
    rfreg_fom.fit(X_train_fl, Prop_fom_train_fl)
    Pred_train = rfreg_fom.predict(X_train_fl)
    Pred_test = rfreg_fom.predict(X_test_fl)
    Pred_out = rfreg_fom.predict(X_out_fl)
    Pred_fom_train_fl = np.array(Pred_train, dtype="float32")
    Pred_fom_test_fl = np.array(Pred_test, dtype="float32")
    Pred_fom_out_fl = np.array(Pred_out, dtype="float32")
    print('rmse_test_fom = ', np.sqrt( sklearn.metrics.mean_squared_error(Prop_fom_test_fl, Pred_fom_test_fl) ))
    print('rmse_train_fom = ', np.sqrt( sklearn.metrics.mean_squared_error(Prop_fom_train_fl, Pred_fom_train_fl) ))





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
    pred_decomp = float(rfreg_decomp.predict( np.reshape(X, (1, -1)) ))
    pred_gap = float(rfreg_gap.predict( np.reshape(X, (1, -1)) ))
    if Data == 'PBE':
        pred_fom = -1*float(rfreg_fom.predict( np.reshape(X, (1, -1)) ))
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

print ('best_decomp_energy = ', float ( rfreg_decomp.predict( np.reshape(X_best[:], (1,-1)) ) ))
print ('best_band_gap = ', float ( rfreg_gap.predict( np.reshape(X_best[:], (1,-1)) ) ))
if Data == 'PBE':
    print ('best_PV_FOM = ', float ( rfreg_fom.predict( np.reshape(X_best[:], (1,-1)) ) ))

#print ('best_decomp_energy = ', float(rfreg_decomp.predict( np.reshape(model.best_variable, (1, -1)) )) )
#print ('best_PV_FOM = ', float(rfreg_fom.predict( np.reshape(model.best_variable, (1, -1)) )) )
#print ('best_band_gap = ', float(rfreg_gap.predict( np.reshape(model.best_variable, (1, -1)) )) )

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



text_file = open("results_GA+RFR.txt", "w")
text_file.write("rmse_test_decomp: %s" % np.sqrt( sklearn.metrics.mean_squared_error(Prop_decomp_test_fl, Pred_decomp_test_fl) ) + '\n')
text_file.write("rmse_train_decomp: %s" % np.sqrt( sklearn.metrics.mean_squared_error(Prop_decomp_train_fl, Pred_decomp_train_fl) ) + '\n')
text_file.write("rmse_test_gap: %s" % np.sqrt( sklearn.metrics.mean_squared_error(Prop_gap_test_fl, Pred_gap_test_fl) ) + '\n')
text_file.write("rmse_train_gap: %s" % np.sqrt( sklearn.metrics.mean_squared_error(Prop_gap_train_fl, Pred_gap_train_fl) ) + '\n')
if Data == 'PBE':
    text_file.write("rmse_test_fom: %s" % np.sqrt( sklearn.metrics.mean_squared_error(Prop_fom_test_fl, Pred_fom_test_fl) ) + '\n')
    text_file.write("rmse_train_fom: %s" % np.sqrt( sklearn.metrics.mean_squared_error(Prop_fom_train_fl, Pred_fom_train_fl) ) + '\n')
text_file.write(":: %s" % '  ' + '\n')
text_file.write("best_decomp_energy: %s" % float(rfreg_decomp.predict( np.reshape(X_best[:], (1, -1)) )) + ' eV' + '\n')
text_file.write("best_band_gap: %s" % float(rfreg_gap.predict( np.reshape(X_best[:], (1, -1)) )) + ' eV' + '\n')
if Data == 'PBE':
    text_file.write("best_PV_FOM: %s" % float(rfreg_fom.predict( np.reshape(X_best[:], (1, -1)) )) + '\n')
text_file.write("best_perovs_alloy: %s" % formula_best_perovs + '\n')
text_file.close()

np.savetxt('scores_GA+RFR.txt', model.report)

# plot and save optimization process plot
model.plot_results(save_as = 'plot_scores_process_GA+RFR.png')

# plot scores of last population
model.plot_generation_scores(title = 'Population scores after ending of searching', save_as= 'plot_scores_end_GA+RFR.png')


