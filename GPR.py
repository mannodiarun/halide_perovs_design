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


##  Choose Dataset Type  ##
Data = 'PBE'
#Data = 'HSE'

##  Choose Descriptor Type  ##
Desc = 'Composition Only'
#Desc = 'Elemental Only'
#Desc = 'Composition and Elemental'

##  Properties = Lattice Constant, Band Gap, Decomposition Energy, PV Figure of Merit; all trained together, nothing to choose for now  ##

##  ML Model is fixed to be Gaussian Process Regression (GPR) here  ##


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
        X = csvdata[0:514,8:22]
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


##  Input outside / new data here as a .csv file  ##
#ifile  = open('Outside.csv', "rt")
#reader = csv.reader(ifile)
#csvdata=[]
#for row in reader:
#    csvdata.append(row)
#ifile.close()
#numrow=len(csvdata)
#numcol=len(csvdata[0])
#csvdata = np.array(csvdata).reshape(numrow,numcol)
#formulas_out = csvdata[:,0]
#Comp_out = csvdata[:,1:15]
#Elem_out = csvdata[:,15:]
#n_out = formulas_out.size
#X_out = copy.deepcopy(Comp_out)
#X_out = copy.deepcopy(Elem_out)
#X_out = csvdata[:,1:]
#X_out_fl = np.array(X_out, dtype="float32")
#X_out_norm = normalize(X_out_fl, norm='l2', axis=0)



###  Training-Test Split  ###

t = 0.20
if Data == 'PBE':
    X_train, X_test, Prop_train_latt, Prop_test_latt, Prop_train_gap, Prop_test_gap, Prop_train_form, Prop_test_form, Prop_train_decomp, Prop_test_decomp, Prop_train_eps, Prop_test_eps, Prop_train_fom, Prop_test_fom = train_test_split(X, PBE_latt, PBE_gap, PBE_form_en, PBE_decomp_en, PBE_eps, PBE_fom, test_size=t)
if Data == 'HSE':
    X_train, X_test, Prop_train_latt, Prop_test_latt, Prop_train_gap, Prop_test_gap, Prop_train_form, Prop_test_form, Prop_train_decomp, Prop_test_decomp = train_test_split(X, HSE_latt, HSE_gap, HSE_form_en, HSE_decomp_en, test_size=t)

n_tr = int(Prop_train_latt.size)
n_te = int(Prop_test_latt.size)

X_train_fl = np.array(X_train, dtype="float32")
X_test_fl = np.array(X_test, dtype="float32")


###  GPR Definition  ###
ker_dp = C(1.0, (1e-3, 1e3)) * DotProduct(2)
#ker_rbf = C(1.0, (1e-5, 1e5)) * RBF(10, (1e-5, 1e5))
ker_rq = C(1.0, (1e-5, 1e5)) * RationalQuadratic(alpha=0.1, length_scale=10, length_scale_bounds=(1e-5, 1e5))
ker_expsine = C(1.0, (1e-5, 1e5)) * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1))
ker_matern = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e4), nu=1.5)
ker_rbf = C(1.0) * RBF(10)
param_grid = {
#"kernel": [ker_matern, ker_rbf, ker_rq, ker_dp, ker_expsine],
"kernel": [ker_rbf],
#"alpha": [1e0, 1e-1, 1e-2, 1e-3],
"alpha": [1e-2],
#"optimizer": ['fmin_l_bfgs_b'],
#"n_restarts_optimizer": [50, 100, 200]
"n_restarts_optimizer": [50]
}

###  Train Model For Lattice Constant  ###
Prop_train_temp = np.array(Prop_train_latt, dtype="float32")
Prop_test_temp = np.array(Prop_test_latt, dtype="float32")
gpr_opt = GridSearchCV(GaussianProcessRegressor(normalize_y=False), param_grid=param_grid, cv=5)
gpr_opt.fit(X_train_fl, Prop_train_temp)
Pred_train = gpr_opt.predict(X_train_fl)
Pred_test  = gpr_opt.predict(X_test_fl)
Prop_train_latt_fl = copy.deepcopy(Prop_train_temp)
Pred_train_latt_fl = np.array(Pred_train, dtype="float32")
Prop_test_latt_fl = copy.deepcopy(Prop_test_temp)
Pred_test_latt_fl = np.array(Pred_test, dtype="float32")
## Outside Predictions ##
#Pred_out = gpr_opt.predict(X_out_fl)
#Pred_out_latt = np.array(Pred_out, dtype="float32")


###  Train Model For Band Gap  ###
Prop_train_temp = np.array(Prop_train_gap, dtype="float32")
Prop_test_temp = np.array(Prop_test_gap, dtype="float32")
gpr_opt = GridSearchCV(GaussianProcessRegressor(normalize_y=False), param_grid=param_grid, cv=5)
gpr_opt.fit(X_train_fl, Prop_train_temp)
Pred_train = gpr_opt.predict(X_train_fl)
Pred_test  = gpr_opt.predict(X_test_fl)
Prop_train_gap_fl = copy.deepcopy(Prop_train_temp)
Pred_train_gap_fl = np.array(Pred_train, dtype="float32")
Prop_test_gap_fl = copy.deepcopy(Prop_test_temp)
Pred_test_gap_fl = np.array(Pred_test, dtype="float32")
## Outside Predictions ##
#Pred_out = gpr_opt.predict(X_out_fl)
#Pred_out_gap = np.array(Pred_out, dtype="float32")


###  Train Model For Decomposition Energy ##
Prop_train_temp = np.array(Prop_train_decomp, dtype="float32")
Prop_test_temp = np.array(Prop_test_decomp, dtype="float32")
gpr_opt = GridSearchCV(GaussianProcessRegressor(normalize_y=False), param_grid=param_grid, cv=5)
gpr_opt.fit(X_train_fl, Prop_train_temp)
Pred_train = gpr_opt.predict(X_train_fl)
Pred_test  = gpr_opt.predict(X_test_fl)
Prop_train_decomp_fl = copy.deepcopy(Prop_train_temp)
Pred_train_decomp_fl = np.array(Pred_train, dtype="float32")
Prop_test_decomp_fl = copy.deepcopy(Prop_test_temp)
Pred_test_decomp_fl = np.array(Pred_test, dtype="float32")
## Outside Predictions ##
#Pred_out = gpr_opt.predict(X_out_fl)
#Pred_out_decomp = np.array(Pred_out, dtype="float32")


###  Train Model For Photovoltaic Figure of Merit (FOM)  ###
if Data == 'PBE':
    X = csvdata[0:514,8:22]
    PBE_fom  = csvdata[0:514,7]
    X_train_temp, X_test_temp, Prop_train_fom, Prop_test_fom = train_test_split(X, PBE_fom, test_size=t)
    X_train_temp_fl = np.array(X_train_temp, dtype="float32")
    X_test_temp_fl = np.array(X_test_temp, dtype="float32")
    Prop_train_temp = np.array(Prop_train_fom[:], dtype="float32")
    Prop_test_temp = np.array(Prop_test_fom[:], dtype="float32")
    gpr_opt = GridSearchCV(GaussianProcessRegressor(normalize_y=False), param_grid=param_grid, cv=5)
    gpr_opt.fit(X_train_temp_fl, Prop_train_temp)
    Pred_train = gpr_opt.predict(X_train_temp_fl)
    Pred_test  = gpr_opt.predict(X_test_temp_fl)
    Prop_train_fom_fl = copy.deepcopy(Prop_train_temp)
    Pred_train_fom_fl = np.array(Pred_train, dtype="float32")
    Prop_test_fom_fl = copy.deepcopy(Prop_test_temp)
    Pred_test_fom_fl = np.array(Pred_test, dtype="float32")
    ## Outside Predictions ##
    #Pred_out = gpr_opt.predict(X_out_fl)
    #Pred_out_fom = np.array(Pred_out, dtype="float32")


#Pred_out = [[0.0 for a in range(4)] for b in range(n_out)]
#for i in range(0,n_out):
#    Pred_out[i][0] = Pred_out_latt[i]
#    Pred_out[i][1] = Pred_out_gap[i]
#    Pred_out[i][2] = Pred_out_decomp[i]
#    Pred_out[i][3] = Pred_out_fom[i]
#np.savetxt('Pred_out.txt', Pred_out)


##  Calculate RMSE Values  ##

mse_test_prop = sklearn.metrics.mean_squared_error(Prop_test_latt_fl, Pred_test_latt_fl)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_latt_fl, Pred_train_latt_fl)
rmse_test_latt = np.sqrt(mse_test_prop)
rmse_train_latt = np.sqrt(mse_train_prop)
print('rmse_test_latt = ', np.sqrt(mse_test_prop))
print('rmse_train_latt = ', np.sqrt(mse_train_prop))
print('      ')

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



#  ML Parity Plots ##

if Data == 'PBE':
    fig, ( [ax1, ax2], [ax3, ax4] ) = plt.subplots( nrows=2, ncols=2, figsize=(8,8) )
    fig.text(0.5, 0.03, 'DFT Calculation', ha='center', fontsize=32)
    fig.text(0.02, 0.5, 'ML Prediction', va='center', rotation='vertical', fontsize=32)
    plt.subplots_adjust(left=0.14, bottom=0.14, right=0.97, top=0.92, wspace=0.30, hspace=0.40)
    plt.rc('font', family='Arial narrow')

    Prop_train_temp = copy.deepcopy(Prop_train_latt_fl)
    Pred_train_temp = copy.deepcopy(Pred_train_latt_fl)
    Prop_test_temp  = copy.deepcopy(Prop_test_latt_fl)
    Pred_test_temp  = copy.deepcopy(Pred_test_latt_fl)
    a = [-175,0,125]
    b = [-175,0,125]
    ax1.plot(b, a, c='k', ls='-')
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.scatter(Prop_train_temp[:], Pred_train_temp[:], c='orangered', marker='s', s=60, edgecolors='dimgrey', alpha=0.9, label='Training')
    ax1.scatter(Prop_test_temp[:], Pred_test_temp[:], c='lawngreen', marker='s', s=60, edgecolors='dimgrey', alpha=0.9, label='Test')
    te = '%.2f' % rmse_test_latt
    tr = '%.2f' % rmse_train_latt
    ax1.text(5.96, 5.48, 'Test_rmse = ' + te + ' $\AA$', c='navy', fontsize=16)
    ax1.text(5.93, 5.28, 'Train_rmse = ' + tr + ' $\AA$', c='navy', fontsize=16)
    ax1.set_ylim([5.1, 7.1])
    ax1.set_xlim([5.1, 7.1])
    ax1.set_xticks([5.5, 6.0, 6.5, 7.0])
    ax1.set_yticks([5.5, 6.0, 6.5, 7.0])
    ax1.set_title('Lattice Constant ($\AA$)', c='k', fontsize=20, pad=12)
    ax1.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':12})

    Prop_train_temp = copy.deepcopy(Prop_train_decomp_fl)
    Pred_train_temp = copy.deepcopy(Pred_train_decomp_fl)
    Prop_test_temp  = copy.deepcopy(Prop_test_decomp_fl)
    Pred_test_temp  = copy.deepcopy(Pred_test_decomp_fl)
    ax2.plot(b, a, c='k', ls='-')
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    ax2.scatter(Prop_train_temp[:], Pred_train_temp[:], c='orangered', marker='s', s=60, edgecolors='dimgrey', alpha=0.9, label='Training')
    ax2.scatter(Prop_test_temp[:], Pred_test_temp[:], c='lawngreen', marker='s', s=60, edgecolors='dimgrey', alpha=0.9, label='Test')
    te = '%.2f' % rmse_test_decomp
    tr = '%.2f' % rmse_train_decomp
    ax2.text(0.58, -0.65, 'Test_rmse = ' + te + ' eV', c='navy', fontsize=16)
    ax2.text(0.45, -1.19, 'Train_rmse = ' + tr + ' eV', c='navy', fontsize=16)
    ax2.set_ylim([-1.7, 3.8])
    ax2.set_xlim([-1.7, 3.8])
    ax2.set_xticks([-1.0, 0.0, 1.0, 2.0, 3.0])
    ax2.set_yticks([-1.0, 0.0, 1.0, 2.0, 3.0])
    ax2.set_title('Decomposition Energy (eV)', c='k', fontsize=20, pad=12)
    #ax2.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':12})

    Prop_train_temp = copy.deepcopy(Prop_train_gap_fl)
    Pred_train_temp = copy.deepcopy(Pred_train_gap_fl)
    Prop_test_temp  = copy.deepcopy(Prop_test_gap_fl)
    Pred_test_temp  = copy.deepcopy(Pred_test_gap_fl)
    ax3.plot(b, a, c='k', ls='-')
    ax3.xaxis.set_tick_params(labelsize=20)
    ax3.yaxis.set_tick_params(labelsize=20)
    ax3.scatter(Prop_train_temp[:], Pred_train_temp[:], c='orangered', marker='s', s=60, edgecolors='dimgrey', alpha=0.9, label='Training')
    ax3.scatter(Prop_test_temp[:], Pred_test_temp[:], c='lawngreen', marker='s', s=60, edgecolors='dimgrey', alpha=0.9, label='Test')
    te = '%.2f' % rmse_test_gap
    tr = '%.2f' % rmse_train_gap
    ax3.text(2.50, 1.10, 'Test_rmse = ' + te + ' eV', c='navy', fontsize=16)
    ax3.text(2.36, 0.52, 'Train_rmse = ' + tr + ' eV', c='navy', fontsize=16)
    ax3.set_ylim([0.0, 6.0])
    ax3.set_xlim([0.0, 6.0])
    ax3.set_xticks([1, 2, 3, 4, 5])
    ax3.set_yticks([1, 2, 3, 4, 5])
    ax3.set_title('Band Gap (eV)', c='k', fontsize=20, pad=12)
    #ax3.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':12})

    Prop_train_temp = copy.deepcopy(Prop_train_fom_fl)
    Pred_train_temp = copy.deepcopy(Pred_train_fom_fl)
    Prop_test_temp  = copy.deepcopy(Prop_test_fom_fl)
    Pred_test_temp  = copy.deepcopy(Pred_test_fom_fl)
    ax4.plot(b, a, c='k', ls='-')
    ax4.xaxis.set_tick_params(labelsize=20)
    ax4.yaxis.set_tick_params(labelsize=20)
    ax4.scatter(Prop_train_temp[:], Pred_train_temp[:], c='orangered', marker='s', s=60, edgecolors='dimgrey', alpha=0.9, label='Training')
    ax4.scatter(Prop_test_temp[:], Pred_test_temp[:], c='lawngreen', marker='s', s=60, edgecolors='dimgrey', alpha=0.9, label='Test')
    te = '%.2f' % rmse_test_fom
    tr = '%.2f' % rmse_train_fom
    ax4.text(4.33, 3.15, 'Test_rmse = ' + te, c='navy', fontsize=16)
    ax4.text(4.23, 2.8, 'Train_rmse = ' + tr, c='navy', fontsize=16)
    ax4.set_ylim([2.5, 6.2])
    ax4.set_xlim([2.5, 6.2])
    ax4.set_xticks([3, 4, 5, 6])
    ax4.set_yticks([3, 4, 5, 6])
    ax4.set_title('Figure of Merit (log$_{10}$)', c='k', fontsize=20, pad=12)
    #ax4.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':12})

    plt.savefig('plot_PBE_GPR_models.pdf', dpi=450)



if Data == 'HSE':
    fig, ( [ax1, ax2, ax3] ) = plt.subplots( nrows=1, ncols=3, figsize=(10,4) )
    fig.text(0.5, 0.03, 'DFT Calculation', ha='center', fontsize=32)
    fig.text(0.02, 0.5, 'ML Prediction', va='center', rotation='vertical', fontsize=32)
    plt.subplots_adjust(left=0.11, bottom=0.24, right=0.96, top=0.86, wspace=0.30, hspace=0.40)
    plt.rc('font', family='Arial narrow')

    Prop_train_temp = copy.deepcopy(Prop_train_latt_fl)
    Pred_train_temp = copy.deepcopy(Pred_train_latt_fl)
    Prop_test_temp  = copy.deepcopy(Prop_test_latt_fl)
    Pred_test_temp  = copy.deepcopy(Pred_test_latt_fl)
    a = [-175,0,125]
    b = [-175,0,125]
    ax1.plot(b, a, c='k', ls='-')
    ax1.xaxis.set_tick_params(labelsize=20)
    ax1.yaxis.set_tick_params(labelsize=20)
    ax1.scatter(Prop_train_temp[:], Pred_train_temp[:], c='orangered', marker='s', s=60, edgecolors='dimgrey', alpha=0.9, label='Training')
    ax1.scatter(Prop_test_temp[:], Pred_test_temp[:], c='lawngreen', marker='s', s=60, edgecolors='dimgrey', alpha=0.9, label='Test')
    te = '%.2f' % rmse_test_latt
    tr = '%.2f' % rmse_train_latt
    ax1.text(5.88, 5.45, 'Test_rmse = ' + te + ' $\AA$', c='navy', fontsize=14)
    ax1.text(5.85, 5.25, 'Train_rmse = ' + tr + ' $\AA$', c='navy', fontsize=14)
    ax1.set_ylim([5.1, 7.1])
    ax1.set_xlim([5.1, 7.1])
    ax1.set_xticks([5.5, 6.0, 6.5, 7.0])
    ax1.set_yticks([5.5, 6.0, 6.5, 7.0])
    ax1.set_title('Lattice Constant ($\AA$)', c='k', fontsize=20, pad=12)
    ax1.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':12})
    
    Prop_train_temp = copy.deepcopy(Prop_train_decomp_fl)
    Pred_train_temp = copy.deepcopy(Pred_train_decomp_fl)
    Prop_test_temp  = copy.deepcopy(Prop_test_decomp_fl)
    Pred_test_temp  = copy.deepcopy(Pred_test_decomp_fl)
    ax2.plot(b, a, c='k', ls='-')
    ax2.xaxis.set_tick_params(labelsize=20)
    ax2.yaxis.set_tick_params(labelsize=20)
    ax2.scatter(Prop_train_temp[:], Pred_train_temp[:], c='orangered', marker='s', s=60, edgecolors='dimgrey', alpha=0.9, label='Training')
    ax2.scatter(Prop_test_temp[:], Pred_test_temp[:], c='lawngreen', marker='s', s=60, edgecolors='dimgrey', alpha=0.9, label='Test')
    te = '%.2f' % rmse_test_decomp
    tr = '%.2f' % rmse_train_decomp
    ax2.text(0.60, -0.70, 'Test_rmse = ' + te + ' eV', c='navy', fontsize=14)
    ax2.text(0.47, -1.24, 'Train_rmse = ' + tr + ' eV', c='navy', fontsize=14)
    ax2.set_ylim([-1.7, 3.8])
    ax2.set_xlim([-1.7, 3.8])
    ax2.set_xticks([-1.0, 0.0, 1.0, 2.0, 3.0]) 
    ax2.set_yticks([-1.0, 0.0, 1.0, 2.0, 3.0])
    ax2.set_title('Decomposition Energy (eV)', c='k', fontsize=20, pad=12)
    #ax2.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':12})

    Prop_train_temp = copy.deepcopy(Prop_train_gap_fl)
    Pred_train_temp = copy.deepcopy(Pred_train_gap_fl)
    Prop_test_temp  = copy.deepcopy(Prop_test_gap_fl)
    Pred_test_temp  = copy.deepcopy(Pred_test_gap_fl)
    ax3.plot(b, a, c='k', ls='-')
    ax3.xaxis.set_tick_params(labelsize=20)
    ax3.yaxis.set_tick_params(labelsize=20)
    ax3.scatter(Prop_train_temp[:], Pred_train_temp[:], c='orangered', marker='s', s=60, edgecolors='dimgrey', alpha=0.9, label='Training')
    ax3.scatter(Prop_test_temp[:], Pred_test_temp[:], c='lawngreen', marker='s', s=60, edgecolors='dimgrey', alpha=0.9, label='Test')
    te = '%.2f' % rmse_test_gap
    tr = '%.2f' % rmse_train_gap
    ax3.text(3.55, 1.70, 'Test_rmse = ' + te + ' eV', c='navy', fontsize=14)
    ax3.text(3.41, 1.05, 'Train_rmse = ' + tr + ' eV', c='navy', fontsize=14)
    ax3.set_ylim([0.5, 7.2])
    ax3.set_xlim([0.5, 7.2])
    ax3.set_xticks([1.5, 3, 4.5, 6])
    ax3.set_yticks([1.5, 3, 4.5, 6])
    ax3.set_title('Band Gap (eV)', c='k', fontsize=20, pad=12)
    #ax3.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':12})

    plt.savefig('plot_HSE_GPR_models.pdf', dpi=450)



