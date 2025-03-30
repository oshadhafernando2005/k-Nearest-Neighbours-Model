import pandas as pd
#Read in the data using pandas
df = pd.read_csv('/content/Network_Intrusion_Dataset.csv')
#Check a sample of the dataset
df.head()
list(df.columns)
#check the number of rows and columns in the dataset
df.shape
import plotly.express as px
#Construct a bar graph for Traffic_Type target variable
Traffic_Type_fig = px.bar(df, x = 'Traffic_Type', title = "Traffic Types")
#Construct a bar graph for Intrusion_Traffic_Type target variable
Intrusion_Traffic_Type_fig = px.bar(df, x ='Intrusion_Traffic_Type', title =" Attacks Types")
#Construct a bivariate bar graph for both target variable
Intrusion_Types_Per_Traffic_fig = px.bar(df,x ='Traffic_Type', color = 'Intrusion_Traffic_Type', title = "Attack Types In
Intrusion Traffic", color_discrete_sequence = px.colors.qualitative.Vivid)
#Remove the bar outline by setting the marker.line.width attribute to 0
Traffic_Type_fig.update_traces(dict(marker_line_width=0))
Intrusion_Traffic_Type_fig.update_traces(dict(marker_line_width=0))
Intrusion_Types_Per_Traffic_fig.update_traces(dict(marker_line_width=0))
#Plot all constructed bar graphs
Traffic_Type_fig.show()
Intrusion_Traffic_Type_fig.show()
Intrusion_Types_Per_Traffic_fig.show()
df.info()
df.describe().transpose()
# To expand e scientific notation
pd.set_option('display.float_format', '{:.2f}'.format)
df.describe().transpose()
df = df.drop(columns=df.loc[:, 'Packets_Rx_Dropped':'Packets_Tx_Errors'].columns)
df = df.drop(columns=df.loc[:, 'Delta_Packets_Rx_Dropped':'Delta_Packets_Tx_Errors'].columns)
df = df.drop(['Is_Valid', 'Table_ID', 'Max_Size'], axis=1)
df.describe(include="all").transpose()
df.isna().sum()/len(df)*100
df.to_csv(r'/content/Prepared_Network_Intrusion_Dataset.csv', index = False)
df_prepared = pd.read_csv('/content/Prepared_Network_Intrusion_Dataset.csv')
#create a dataframe with all training data except the target column
X = df_prepared.drop(columns=['Traffic_Type','Intrusion_Traffic_Type'])
# here, we select one target variable to model, Traffic_Type
y = df_prepared['Traffic_Type']
#check that the list of input variables
list(X)
#check that the list of target variable
y.head()
from sklearn.model_selection import train_test_split
#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=14, stratify=y)
#This is to show the number of instances and input features in the training and test sets
print('X_train Instances', X_train.shape)
print('X_test Instances', X_test.shape)
from sklearn.neighbors import KNeighborsClassifier
# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors = 9)
# Fit the classifier to the data
knn.fit(X_train,y_train)
#Perform predictions on the test data
y_pred=knn.predict(X_test)
#Create a dataframe for comparing the actual vs predicted results by kNN mode
compare_results_knn_df = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})
compare_results_knn_df.to_csv(r'/content/knn_pred_comparison.csv', index=True)
compare_results_knn_df
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
#Import the packages for costructing the confusion matrix
from sklearn.metrics import confusion_matrix
#Import the packages for plotting the confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay
#Costruct the confusion matrix based on…
#comparing actual values (y_test) vs predicted (y_pred) in test data
cm_knn = confusion_matrix(y_test, y_pred, labels = knn.classes_)
#Plot the confusion matrix
disp_knn_cm = ConfusionMatrixDisplay(cm_knn, display_labels=knn.classes_)
disp_knn_cm.plot()
from sklearn.metrics import RocCurveDisplay
knn_roc = RocCurveDisplay.from_estimator(knn, X_test, y_test)
# Calculating error for K values between 1 and 40
error = []
import numpy as np
import matplotlib.pyplot as plt
# Calculating error for K values between 1 and 40
for i in range(1, 40):
knn2 = KNeighborsClassifier(n_neighbors=i)
knn2.fit(X_train, y_train)
pred_i = knn2.predict(X_test)
error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
from sklearn.neighbors import KNeighborsClassifier
# Create KNN classifier
knn1 = KNeighborsClassifier(n_neighbors = 1)
# Fit the classifier to the data
knn1.fit(X_train,y_train)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm_knn1 = confusion_matrix(y_test, y_pred, labels = knn1.classes_)
disp_knn1_cm = ConfusionMatrixDisplay(cm_knn1, display_labels=knn1.classes_)
disp_knn1_cm.plot()
y_pred=knn1.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.metrics import RocCurveDisplay
knn_roc = RocCurveDisplay.from_estimator(knn1, X_test, y_test)


from sklearn.model_selection import GridSearchCV
import numpy as np
#create new a knn model
knn = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors and distances
param_grid = {'n_neighbors': np.arange(1, 25), 'metric': ['euclidean', 'manhattan']}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn, param_grid, cv=5, scoring = 'roc_auc')
#fit model to data
knn_gscv.fit(X, y)

knn_gscv.best_params_
# Perform testing on test dataset
y_pred = knn_gscv.predict(X_test)
# Construct a confusion matrix
cm_knn_gscv = confusion_matrix(y_test, y_pred, labels = knn_gscv.classes_)
disp_knn_gscv_cm = ConfusionMatrixDisplay(cm_knn_gscv, display_labels=knn_gscv.classes_)
disp_knn_gscv_cm.plot()
# Display the classification report
print(classification_report(y_test, y_pred))















                                         











                                         
