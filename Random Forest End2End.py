import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, power_transform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, roc_curve, roc_auc_score

df1 = pd.read_csv('winequality-red.csv', ';')
df2 = pd.read_csv('winequality-white.csv', ';')

# DELETE after testing
df1['type'] = int(0) # zero means red wine
df2['type'] = int(1) # one means white wine

# change to quality after testing
data = pd.concat([df1, df2], ignore_index=True)
labels = data['type']
data = data.drop(['quality', 'type'], axis=1)

train_X, test_X, train_Y, test_Y = train_test_split(data, labels, test_size = 0.2, random_state=7, shuffle=True)

# The Drop_Fill function:
# 1- removes the attributes with missing values higher than 20%
# 2- fill the rest using the average of the last and next valid values
# 3- Apart from the updated dataframe, it also returns a summary of the data quality based on the proportion of missing values
# input --> pandas dataframe
def Drop_Fill(dataframe):
        Quality=[]
        tollerance = 0.2
        for column in dataframe:
            quality = dataframe[column].isnull().value_counts()
            qual_percent = np.round(quality.iloc[0]/len(dataframe)*100, decimals=2)
            Quality.append(f"{column}: {qual_percent}")
            if qual_percent < 1 - tollerance:
                dataframe = dataframe.drop([column], axis=1)
            else:
                pass
            front_fill = dataframe.ffill()
            back_fill = dataframe.bfill()
            front_fill = front_fill.bfill() # The reverse method covers the case of NaN values on the first row
            back_fill = back_fill.ffill()   # The same here for NaN values at the last row
            df = (front_fill+back_fill)/2
        return(df, Quality)
        

# The Distribution function:
# 1- plots the distribution of each attribute (histogram)
# 2- calculates the Anderson Darling normality test and adds that into the graph
# input --> pandas dataframe, number of columns returned in the figure
def Distribution(dataframe, columns_num):
        length = len(dataframe.columns)
        lines = length%columns_num + length//columns_num
        figure = plt.figure(
                figsize= (2.4*columns_num,3.3*lines),
                facecolor='w'
                )
        plt.subplots_adjust(
                left= 0.1, 
                right= 1, 
                wspace= 0.1, 
                hspace= 0.15
                )
        sb.set_style("dark")
        for i, column in enumerate(dataframe):
            t_stat, crit_vals, _ = stats.anderson(dataframe[column], dist='norm')
            ax = plt.subplot(lines,columns_num, i+1)
            hist, bins = np.histogram(dataframe[column], bins= 'auto', density= True)
            sb.distplot(np.array(dataframe[column]), bins = bins, hist= True, kde=True, color= 'b')
            ax.set_title(dataframe.columns[i], weight = 'bold', size = 10)
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            plt.text(
              bins[-1],
              hist.max(),
              'AD' + ': ' + str(np.around((crit_vals[2]-t_stat), 3)),
              fontsize = 8,
              fontweight = 'bold',
              color = 'r',
              ha = 'right',
              va = 'top'
              )
        plt.show()
        return (figure)

# The Outlier_Values function:
# 1- detects the outlier values
# 2- plots the box-plot of each attribute
# input --> pandas dataframe, number of columns returned in the figure
def Outlier_Values(dataframe, columns_num):
        length = len(dataframe.columns)
        lines = length%columns_num + length//columns_num
        figure = plt.figure(
                figsize= (2.4*columns_num,4*lines),
                facecolor='w'
                )
        plt.subplots_adjust(
                left= 0.1, 
                right= 1, 
                wspace= 0.1, 
                hspace= 0.15
                )
        sb.set_style("dark")
        for i, column in enumerate(dataframe):
            ax = plt.subplot(lines,columns_num, i+1)
            dataframe.boxplot(column = column,
                            grid = False, 
                            autorange = True,
                            showmeans = False,
                            flierprops = dict(markerfacecolor='r', marker='o'),
                            )
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            ax.set_title(dataframe.columns[i], weight = 'bold', size = 10)
        plt.show()
        return (figure)
    
# The Normalisation function:
# 1- Normalise the data using the Box_Cox transformation
# input --> pandas dataframe
def Normalisation(dataframe):
        normalised_df = pd.DataFrame(columns= dataframe.columns, index = dataframe.index)
        for column in dataframe:
            if dataframe[column].min() <= 0:
                dataframe[column] = dataframe[column] + dataframe[column].min() + 1
            else:
                pass
            Box_trans = power_transform(dataframe[[column]], method='box-cox')
            normalised_df[column] = Box_trans[:,0]
        return (normalised_df)

# The Scaling function:
# 1- Scales data from 0.5 to 1.5
# input --> pandas dataframe
def Scaling(dataframe):
        scaled_df = pd.DataFrame(columns= dataframe.columns)
        # the MinMaxScaler is used because it preserves the shape of the original data
        scaler = MinMaxScaler(feature_range=(0.5, 1.5)) 
        for column in dataframe:
            a = scaler.fit_transform(dataframe[[column]])
            scaled_df[column] = a[:,0]
        df = scaled_df
        return (df)

Number_of_columns = 5
  
df_filled, qual = Drop_Fill(train_X)
distrib =  Distribution(df_filled, Number_of_columns)
out_val = Outlier_Values(df_filled, Number_of_columns)
norm_data = Normalisation(df_filled)
df = Scaling(norm_data)
final_distrib =  Distribution(df, Number_of_columns)

# initialising the model
model = RandomForestClassifier(random_state=7)

# the hyperparemters for optimisation
params = {"n_estimators":[1,2,3],
          "criterion":['gini', 'entropy'],
          "max_depth":[3, 4, 5, 6, 7],
          "min_samples_leaf": [2,4,6,8,10],
          "bootstrap":[True, False]}

# Random Search is used instead of Grid search to reduce the computational complexity
# A Cross Validation with 10 folds is used through the hyperparameter optimisation process 
# to minnimise overfitting

optimised_model = RandomizedSearchCV(model, params, cv=10)
optimised_model.fit(train_X, train_Y)
print("tuned parmas: {}".format(optimised_model.best_params_))
print("best score{}".format(optimised_model.best_score_))
opt_params = optimised_model.best_params_ # save the optimal parametres

# Comment out the lines below to use the opimal parametres BUT skip Cross Validation and Grid Search

#optimised_model = RandomForestClassifier(n_estimators = opt_params['n_estimators'],
#                                         criterion = opt_params['criterion'],
#                                         max_depth = opt_params['max_depth'],
#                                         min_samples_leaf = opt_params['min_samples_leaf'],
#                                         bootstrap = opt_params['bootstrap'])
#
#optimised_model.fit(train_X, train_Y)

prediction = optimised_model.predict(test_X)
prob_predict = optimised_model.predict_proba(test_X)[:,1]

cm = confusion_matrix(test_Y, prediction)

# The fuction below plots a confusion matrix
# input --> confusion matrix data (TP, FP, TN, FN), field of study (string), algorithm name (string)
def Confusion_Matrix(conf_matrix_data, field, algorithm_name):
        figure = plt.figure(figsize=(6,3))
        sb.heatmap(conf_matrix_data,
                        yticklabels=labels.unique(),
                        xticklabels=labels.unique(),
                        annot=True,
                        annot_kws={"size":10, "color": "black", "weight":"bold"},
                        fmt="d",
                        cmap='winter',
                        linewidths=.5)
        plt.title(f'{field} - {algorithm_name}', weight='bold')
        plt.xlabel('Actual values', weight='bold', size=10)
        plt.ylabel('Predicted values', weight='bold', size=10)
        return(figure)

cm_plot = Confusion_Matrix(cm, "Wine Quality", "Random forest")

# The function below plots the Receiver Operating Characteristic curve (ROC)
# In other words it shows the Area Under Curve (AUC)
# input --> actual labels, probabilities predicted by model, algorithm name (string)
def ROC_Plot(targets, probabilities_predicted, algorithm_name):
        false_pos_rate, true_pos_rate, threshold = roc_curve(targets, probabilities_predicted)
        auc = roc_auc_score(targets, probabilities_predicted)
        figure = plt.figure(figsize=(6,3))
        plt.plot(false_pos_rate, true_pos_rate,'-', linewidth=3, label=f"{algorithm_name} AUC = {np.around(auc,3)}")
        plt.plot([0,1],[0,1],"--")
        plt.xlabel("False Positive rate")
        plt.ylabel("True Positive rate")
        plt.xlim(-0.01, 1)
        plt.ylim(0, 1.01)
        plt.legend(loc=4)
        return(figure)

roc_plot = ROC_Plot(test_Y, prob_predict, "Random Forest")

accuracy = accuracy_score(test_Y, prediction)
recall = recall_score(test_Y, prediction, average="macro")
precision = precision_score(test_Y, prediction, average="macro")
f_score = f1_score(test_Y, prediction,average="macro")

print(f"The Accuracy is {accuracy}")
print(f"The Recall is {recall}")
print(f"The Precision is {precision}")
print(f"The F score is {f_score}")