# https://medium.com/towards-data-science/8-plots-for-explaining-linear-regression-to-a-layman-489b753da696

# -------------------------------------------------------- #
# Importing Libraries                                      #
# -------------------------------------------------------- #
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
import shap

# -------------------------------------------------------- #
# Importing Dataset & Creating New Variables               #
# -------------------------------------------------------- #

# Load dataset and add squared term
credit_score = pd.read_csv("C:\\Users\\wallj\\DS_Projects\\Datasets\\credit_score.csv")

# Target variable
y = credit_score['CREDIT_SCORE']

# Add squared term
credit_score['R_EXPENDITURE_SQR'] = credit_score['R_EXPENDITURE']**2

# Select features
X = credit_score[['INCOME','R_DEBT_INCOME','R_EXPENDITURE','R_EXPENDITURE_SQR','R_ENTERTAINMENT','CAT_GAMBLING']].copy()

# One-hot encoding
X['GAMBLING_LOW'] = X['CAT_GAMBLING'].apply(lambda x: 1 if x == 'Low' else 0)
X['GAMBLING_HIGH']  = X['CAT_GAMBLING'].apply(lambda x: 1 if x == 'High' else 0)
X.drop(columns=['CAT_GAMBLING'], inplace=True)

X.head()


# -------------------------------------------------------- #
# Build Statsmodels Linear Regression Model                #
# -------------------------------------------------------- #
# Add a constant to the independent variables (intercept)
X = sm.add_constant(X)

# Fit a linear regression model
model = sm.OLS(y, X).fit()

# Output the summary of the model
print(model.summary())



# -------------------------------------------------------- #
# Plot 1: Residual PLot                                   #
# -------------------------------------------------------- #
# Get model predictions
y_pred = model.predict(X)

# Plot predicted vs actual
fig, ax = plt.subplots(figsize=(5, 5))
plt.scatter(y_pred,y)

# Plot y=x line
ax.plot([y.min(), y.max()], [y.min(), y.max()], color='tab:red')

plt.xlabel('Predicted',size=15)
plt.ylabel('Actual', size=15)



# -------------------------------------------------------- #
# Plot 2: Correlation Matrix                               #
# -------------------------------------------------------- #
# Correlation matrix
corr_matrix = X.drop('const',axis=1).corr()
sns.heatmap(corr_matrix, cmap="coolwarm", fmt=".1f", vmin=-1, vmax=1,annot=True)


# -------------------------------------------------------- #
# Plot 3: Weight Plot                                      #
# -------------------------------------------------------- #
# Get coefficients and standard errors
coefficients = model.params[1:][::-1]
se = model.bse[1:][::-1]
features = X.columns[1:][::-1]

plt.figure(figsize=(10, 6))

# Plot vertical dotted line at x=0
plt.axvline(x=0, color='tab:red', linestyle='--')

# Plot the coefficients and error bars
plt.errorbar(coefficients, 
             features, 
             xerr=se, 
             fmt='o', 
             capsize=5)

plt.xlabel('Coefficient (Weight)', size=15)



# -------------------------------------------------------- #
# Plot 4: Effect Plot                                      #
# -------------------------------------------------------- #
# Calculate the feature effects 
feature_effects = X * model.params

# Combine feature effects for related features
feature_effects['R_EXPENDITURE'] = feature_effects['R_EXPENDITURE'] + feature_effects['R_EXPENDITURE_SQR']
feature_effects['GAMBLING'] = feature_effects['GAMBLING_LOW']+feature_effects['GAMBLING_HIGH']
feature_effects.drop(['const','R_EXPENDITURE_SQR','GAMBLING_LOW','GAMBLING_HIGH'],axis=1,inplace=True)

# Create effect plots using boxplots
plt.figure(figsize=(12, 7))
plt.axvline(x=0, color='tab:red', linestyle='--')
sns.boxplot(data=feature_effects, orient="h", color="tab:Blue")
plt.xlabel("Effect on Credit Score", size=15)



# Calculate the feature effects
feature_effects = X * model.params
const = feature_effects['const'][0] 

# Combine feature effects for related features
feature_effects['R_EXPENDITURE'] = feature_effects['R_EXPENDITURE'] + feature_effects['R_EXPENDITURE_SQR']
feature_effects['GAMBLING'] = feature_effects['GAMBLING_LOW']+feature_effects['GAMBLING_HIGH']
feature_effects.drop(['const','R_EXPENDITURE_SQR','GAMBLING_LOW','GAMBLING_HIGH'],axis=1,inplace=True)

# Add the constant to the feature effects
feature_effects = feature_effects + const

# Create effect plots using boxplots
plt.figure(figsize=(12, 7))
plt.axvline(x=const, color='tab:red', linestyle='--')
sns.boxplot(data=feature_effects, orient="h", color="tab:Blue", showfliers=False)
plt.xlabel("Effect on Credit Score", size=15)

# -------------------------------------------------------- #
# PLot 5: Mean Effect Plot (Feature Importance)            #
# -------------------------------------------------------- #

# Calculate the feature effects 
feature_effects = X * model.params

# Combine feature effects for related features
feature_effects['R_EXPENDITURE'] = feature_effects['R_EXPENDITURE'] + feature_effects['R_EXPENDITURE_SQR']
feature_effects['GAMBLING'] = feature_effects['GAMBLING_LOW']+feature_effects['GAMBLING_HIGH']
feature_effects.drop(['const','R_EXPENDITURE_SQR','GAMBLING_LOW','GAMBLING_HIGH'],axis=1,inplace=True)
# Calculate the absolute values of the feature effects
feature_effects = abs(feature_effects)
mean_effects = feature_effects.mean(axis=0)
# Sort by mean effect
mean_effects.sort_values(inplace=True)
# Create a bar plot for feature importance
plt.figure(figsize=(12, 6))
plt.barh(mean_effects.index, mean_effects)
plt.xlabel('Mean Effect on Credit Score', size=15)


# -------------------------------------------------------- #
# Plot 6: Individual Effect Plot                           #
# -------------------------------------------------------- #
# Create effect plots using boxplots
plt.figure(figsize=(12, 8))
plt.axvline(x=const, color='black', linestyle='--')
sns.boxplot(data=feature_effects, orient="h", color="tab:Blue", showfliers=False)
plt.xlabel("Effect on Credit Score", size=15)

idx = 0  # You can change this to any valid index in your dataset
ind_feature_effect = X.iloc[idx] * model.params[1:]

# Combine feature effects for related features
ind_feature_effect['R_EXPENDITURE'] = ind_feature_effect['R_EXPENDITURE'] + ind_feature_effect['R_EXPENDITURE_SQR']
ind_feature_effect['GAMBLING'] = ind_feature_effect['GAMBLING_LOW']+ind_feature_effect['GAMBLING_HIGH']
ind_feature_effect.drop(['const','R_EXPENDITURE_SQR','GAMBLING_LOW','GAMBLING_HIGH'],axis=0,inplace=True)

# Add the constant to the feature effects
ind_feature_effect = ind_feature_effect + const
for i, feature in enumerate(ind_feature_effect):
    plt.scatter(feature, i, c='tab:red', marker='x', s=100, zorder=10)


# -------------------------------------------------------- #
# Plot 7: Effect Trend PLot                                #
# -------------------------------------------------------- #
# Calculate the feature effects 
feature_effects = X * model.params

# Plot effect of R_DEPT_INCOME
plt.figure(figsize=(6, 4))
plt.scatter(X['R_DEBT_INCOME'], feature_effects['R_DEBT_INCOME'])

plt.xlabel('R_DEBT_INCOME', size=12)
plt.ylabel('Effect on Credit Score', size=12)


# -------------------------------------------------------- #
# PLot 8: SHAP Values                                      #
# -------------------------------------------------------- #
# Calculate SHAP values
explainer = shap.KernelExplainer(model.predict,X)
shap_values = explainer(X)

shap.plots.waterfall(shap_values[0], show=False)

preds = model.predict(X)

# calculate average prediction
print(np.mean(preds))

# SHAP bar plot
shap.plots.bar(shap_values)