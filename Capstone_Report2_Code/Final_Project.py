# Final Project

#%%
# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pylab
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn import neighbors
from sklearn import ensemble
from sklearn import feature_selection
from sklearn import neural_network
import xgboost as xgb

#%% 
# import train_stdaset
train_data = pd.read_csv("/Users/seacow/Documents/School Work/Spring 2020/STA 160/data/superconduct/train.csv")
unique_m = pd.read_csv("/Users/seacow/Documents/School Work/Spring 2020/STA 160/data/superconduct/unique_m.csv")

train_variables = train_data.drop(columns = 'critical_temp')
#%%
# Exploratory train_stda Analysis
# number of variables in train_data
display(train_data.shape)
# number of variables in unique_m
display(unique_m.shape)
# variables available in the train_stdaset
display(train_variables.columns)

#%%
# check for missing value
display(train_data.isnull().sum().sum())
display(unique_m.isnull().sum().sum())

unique_m = unique_m.drop(columns = 'material')
plt.spy(unique_m.iloc[1:100,], precision = 0.1, markersize = 5) 

#%%
# dense columns
unique_m.describe()

#%%
# check correlation matrix
train_corr = train_variables.corr()
density_corr = train_variables.iloc[:, 31:41].corr()

f, axes = plt.subplots(1, 2, figsize = (20, 8))
sns.heatmap(train_corr, cmap = "YlGnBu", ax = axes[0])
sns.heatmap(density_corr, cmap = "YlGnBu", ax = axes[1])
#%%
# check the distribution of critical_temp
critical_temp = train_data['critical_temp']
display(critical_temp.describe())
critical_temp.plot.kde(legend=False, label = "Density of Critical Tempreture")
critical_temp.plot.hist(density = "True", title = "Estimated density of the critical temperatrues")

#%%
# try cutoff values in the tail
critical_temp_cutoff = critical_temp[critical_temp <= 50]
critical_temp_cutoff.plot.kde(legend=False, label = "Density of Critical Tempreture")
critical_temp_cutoff.plot.hist(density = "True")

# %%
# Apply Box-Cox transformation
critical_temp_boxcox = list(stats.boxcox(critical_temp))
lambda_val = critical_temp_boxcox[1]
display(lambda_val)


#%%
critical_temp_boxcox = pd.DataFrame(critical_temp_boxcox[0])

fig = plt.figure(figsize = (20, 8))
plt.subplot(1, 3, 1)
critical_temp.plot.kde(legend=False, label = "Density of Critical Tempreture", title = "Estimated Density of the Critical Temperature")
plt.subplot(1, 3, 2)
critical_temp_boxcox[0].plot.kde(legend=False, label = "Density of Critical Tempreture", title = "Box-Cox Transformed Critical Temperature")
plt.subplot(1, 3, 3)
fig = sns.distplot(critical_temp_boxcox[0], kde = True, fit = stats.gamma, hist = False)
fig.set_title("Comparison of the Box-Cox Transformed train_stda with Normal Density")

# %%
# conduct PCA on the train train_stdaset (require to attain 95% FVE)
# standardization
scaler = preprocessing.StandardScaler()
train_std = scaler.fit_transform(train_variables)
#%%
pca = PCA(0.90)
pca.fit(train_std)

#%%
# check FVE
display(pca.explained_variance_ratio_)
display(sum(pca.explained_variance_ratio_))

# Scree plot
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title("Cumulative Variance Explained by PC")

#%%
# see if there is any clustering
pca_vars = pca.transform(train_variables)
pca_vars = pd.DataFrame(pca_vars)
#%%
# Visualize PCA Result
fig = plt.figure(figsize = (16, 8))
ax = plt.subplot(1, 2, 1)
ax.scatter(pca_vars.iloc[:,0], pca_vars.iloc[:,1], marker='o')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title("PC1 VS PC2")

ax = plt.subplot(1, 2, 2, projection = "3d")
ax.scatter(pca_vars.iloc[:,0], pca_vars.iloc[:,1], pca_vars.iloc[:,2], marker='o')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title("PC1 & PC2 & PC3")
# %%
# obtain the loadings of each variable
loadings = pca.components_
loadings

# %%
# train test split
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_variables, critical_temp_boxcox)

#%%
# apply PCA to the train and test train_stda
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
# %%
a = np.arange(0.05, 1, 0.05)
a = a.tolist()

#%%
# model the train_stda with LASSO
result = list()

for i in a:
    clf = linear_model.Lasso(alpha = i)
    clf.fit(X_train, Y_train)
    predict_Y = clf.predict(X_test)
    mse = metrics.mean_squared_error(Y_test, predict_Y)
    # inverse the box-cox transformation
    mse_original = (mse * 0.24 + 1) ** (1/0.24)
    display(clf.score(X_train, Y_train))
    result.append([mse, mse_original])

#%%
result
################################################################################################
#%%
# Attempt Adaboost Regressor
ada_clf = ensemble.AdaBoostRegressor(random_state = 0, n_estimators = 100)
ada_clf.fit(X_train, Y_train)

#%%
predict_Y = ada_clf.predict(X_test)

# %%
display(metrics.mean_squared_error(Y_test, predict_Y))
display(ada_clf.score(X_train, Y_train))
# %%
# inverse the box-cox transformation
(3.5441789916061763 * 0.24 + 1) ** (1/0.24)
# %%
# check the mean critical tempreture
np.mean(train_data['critical_temp'])

# %%
################################################################################################
# Try Gradient Boosting
gb_clf = ensemble.GradientBoostingRegressor(random_state = 0)
gb_clf.fit(X_train, Y_train)
predict_Y = gb_clf.predict(X_test)

display(metrics.mean_squared_error(Y_test, predict_Y))
display(gb_clf.score(X_train, Y_train))

#%%
# inverse the box-cox transformation
(1.9660428250097517 * 0.24 + 1) ** (1/0.24)
#%%
################################################################################################
# Attempt XGBoosting Regressor
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 10, alpha = 10, n_estimators = 10)

xg_reg.fit(X_train, Y_train)
predict_Y = xg_reg.predict(X_test)

display(metrics.mean_squared_error(Y_test, predict_Y))
display(xg_reg.score(X_train, Y_train))

#%%
# inverse the box-cox transformation
(4.58226774071773 * 0.24 + 1) ** (1/0.24)

#%%
################################################################################################
# Linear Regression
linear_clf = linear_model.LinearRegression()
linear_clf.fit(X_train, Y_train)
predict_Y = linear_clf.predict(X_test)

display(metrics.mean_squared_error(Y_test, predict_Y))
display(linear_clf.score(X_train, Y_train))

#%%
# inverse the box-cox transformation
(2.8286988119239953 * 0.24 + 1) ** (1/0.24)
# %%
##########################################################################################
#%%
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_variables, critical_temp_boxcox)

#%%
# Attempt Full model with LASSO
clf.fit(X_train, Y_train)
predict_Y = clf.predict(X_test)
display(metrics.mean_squared_error(Y_test, predict_Y))
display(clf.score(X_train, Y_train))

#%%
# inverse the box-cox transformation
(2.2993669635202783 * 0.24 + 1) ** (1/0.24)

#%%
################################################################################################
# Full model with Adaboost
ada_clf.fit(X_train, Y_train)
predict_Y = ada_clf.predict(X_test)
display(metrics.mean_squared_error(Y_test, predict_Y))
display(ada_clf.score(X_train, Y_train))

#%%
# inverse the box-cox transformation
(2.861413838082465 * 0.24 + 1) ** (1/0.24)

#%%
################################################################################################
# Full model with Gradient Boosting
gb_clf = ensemble.GradientBoostingRegressor(random_state = 0)
gb_clf.fit(X_train, Y_train)
predict_Y = gb_clf.predict(X_test)

display(metrics.mean_squared_error(Y_test, predict_Y))
display(gb_clf.score(X_train, Y_train))

#%%
# inverse the box-cox transformation
(1.2264828211428287 * 0.24 + 1) ** (1/0.24)

#%%
#
predict_result = pd.DataFrame([list(predict_Y), list(Y_test.iloc[:,0])])
predict_result = predict_result.T
predict_result.columns = ["Predicted Value", "True Label"]

fig = sns.kdeplot(predict_result.iloc[:,0], shade=True)
sns.kdeplot(predict_result.iloc[:,1], shade=True)
fig.set_title("Comparison of the density plot of the predicted and true label")
fig.set_xlabel("Center")
fig.set_ylabel("Density")

#%%
# plot real value against predicted value
sns.regplot(predict_result.iloc[:,0], predict_result.iloc[:,1], x_jitter=.1, line_kws = {"color": "black"})

# check residual
#%%
linear_res = predict_result.iloc[:,1] - predict_result.iloc[:,0]
#%%
plt.figure(figsize = (8, 10))
stats.probplot(linear_res, dist = "norm", plot = pylab)

#%%
################################################################################################
# XGBoost on full train_stdaset
xg_reg.fit(X_train, Y_train)
predict_Y = xg_reg.predict(X_test)

display(metrics.mean_squared_error(Y_test, predict_Y))
display(xg_reg.score(X_train, Y_train))

#%%
# inverse the box-cox transformation
(3.887573347155465 * 0.24 + 1) ** (1/0.24)

#%%
################################################################################################
# Linear Regression on full train_stdaset
linear_clf = linear_model.LinearRegression()
linear_clf.fit(X_train, Y_train)
predict_Y = linear_clf.predict(X_test)

display(metrics.mean_squared_error(Y_test, predict_Y))
display(linear_clf.score(X_train, Y_train))

#%%
# inverse the box-cox transformation
(1.9333555687675603 * 0.24 + 1) ** (1/0.24)

#%%
# feature selection based on full linear model
num_lst = np.arange(10, 80, 5)
result = []
for num in num_lst:
    selector = feature_selection.RFE(linear_clf,  n_features_to_select = num, step = 10)
    selector.fit(X_train, Y_train)
    predict_Y = selector.predict(X_test)

    result.append([metrics.mean_squared_error(Y_test, predict_Y),
                   selector.score(X_train, Y_train)])

    display("Important features are: ", X_train.columns[selector.get_support()])

#%%
# Visualize
result = pd.DataFrame(result)

fig = plt.figure(figsize = (16, 8))
plt.subplot(1, 2, 1)
plt.plot(np.arange(10, 80, 5), result.iloc[:,0], '-ok')
plt.xlabel("Number of Variables")
plt.ylabel("MSE Score")
plt.title("MSE Score against the number of variables used")

plt.subplot(1, 2, 2)
plt.plot(np.arange(10, 80, 5), result.iloc[:,1], '-ok')
plt.xlabel("Number of Variables")
plt.ylabel("Coefficient of Determination")
plt.title("R^2 score against the number of variables used")

#%%
# try multiplier perceptron
mlp_clf = neural_network.MLPRegressor()
mlp_clf.fit(X_train, Y_train)
predict_Y = mlp_clf.predict(X_test)

display(metrics.mean_squared_error(Y_test, predict_Y))
#%%
################################################################################################
# Now shift attention to the second train_stdaset
unique_m.describe()

#%%
# get correlation of this train_stdaset
m_corr = unique_m.corr()
sns.heatmap(m_corr)

# %%
# attempt to fit a linear model for this train_stdaset
m_variables = unique_m.drop(columns = ['critical_temp', 'material'])
m_label = unique_m['critical_temp']

#%%
# train test split and fit model
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(m_variables, m_label)
linear_clf.fit(X_train, Y_train)
predict_Y = linear_clf.predict(X_test)

metrics.mean_squared_error(Y_test, predict_Y)

#%%
linear_clf.score(X_train, Y_train)

#%%
########################################################################################
# reload data
train_data = pd.read_csv("/Users/seacow/Documents/School Work/Spring 2020/STA 160/data/superconduct/train.csv")

#%%
# divide data according to value of the response variable
train_data['label'] = pd.cut(train_data['critical_temp'], 3, labels = ["Low_Temp", "Mid_Temp", "High_Temp"])

#%%
# Get label information
train_data["label"].value_counts()

#%%
train_label = train_data["label"]
train_vars = train_data.iloc[:,0:81]

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(train_vars, train_label)

#%%
# KNN classifier
knn_clf = neighbors.KNeighborsClassifier(n_neighbors = 8)
knn_clf.fit(X_train, Y_train)

predict_Y = knn_clf.predict(X_test)
metrics.confusion_matrix(Y_test, predict_Y)

#%%
# adaboost
ada_clf = ensemble.AdaBoostClassifier()
ada_clf.fit(X_train, Y_train)

predict_Y = ada_clf.predict(X_test)
metrics.confusion_matrix(Y_test, predict_Y)

#%%
# Random Forest
rf_clf = ensemble.RandomForestClassifier()
rf_clf.fit(X_train, Y_train)

predict_Y = rf_clf.predict(X_test)
metrics.confusion_matrix(Y_test, predict_Y)

#%%
Y_test.value_counts()

#%%
Group1 = train_data[train_data['label'] == 'Low_Temp']
Group2 = train_data[train_data['label'] == 'Mid_Temp']
Group3 = train_data[train_data['label'] == 'High_Temp']

#%%
# Gradient Boosting Regressor
G1_vars = Group1.iloc[:,0:81]
G1_label = Group1['critical_temp']
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(G1_vars, G1_label)
gb_clf = ensemble.GradientBoostingRegressor(random_state = 0)
gb_clf.fit(X_train, Y_train)
predict_Y = gb_clf.predict(X_test)

display(metrics.mean_squared_error(predict_Y, Y_test))
display(gb_clf.score(X_train, Y_train))

#%%
# Gradient Boosting Regressor
G2_vars = Group2.iloc[:,0:81]
G2_label = Group2['critical_temp']
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(G2_vars, G2_label)
gb_clf = ensemble.GradientBoostingRegressor(random_state = 0)
gb_clf.fit(X_train, Y_train)
predict_Y = gb_clf.predict(X_test)

display(metrics.mean_squared_error(predict_Y, Y_test))
display(gb_clf.score(X_train, Y_train))

#%%
# Gradient Boosting Regressor
G3_vars = Group3.iloc[:,0:81]
G3_label = Group3['critical_temp']
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(G3_vars, G3_label)
gb_clf = ensemble.GradientBoostingRegressor(random_state = 0)
gb_clf.fit(X_train, Y_train)
predict_Y = gb_clf.predict(X_test)

display(metrics.mean_squared_error(predict_Y, Y_test))
display(gb_clf.score(X_train, Y_train))

#%% 
# Visualize
fig = plt.figure(figsize = (24, 8))
plt.subplot(1, 3, 1)
G1_label.plot.kde(legend=False, label = "Density of Critical Tempreture", title = "Density plot of Group 1")
plt.subplot(1, 3, 2)
G2_label.plot.kde(legend=False, label = "Density of Critical Tempreture", title = "Density plot of Group 2")
plt.subplot(1, 3, 3)
G3_label.plot.kde(legend=False, label = "Density of Critical Tempreture", title = "Density plot of Group 3")

#%%
# check mean response value of each group
display(G1_label.mean())
display(G2_label.mean())
display(G3_label.mean())