from pandas.core.algorithms import mode
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
#install plotly to view table
import plotly.graph_objects as go


st.write("""
# Covid-19 Prediction Application
This app is used to measure **mean square error** for each prediction model base on different countries dataset
""")

# Sidebar
# User input
st.sidebar.write(""" 
## Input
""")
st.sidebar.header('Country and Prediction Model Selection')
country = st.sidebar.selectbox(
    'Country', ('India', 'America', 'South Korea', 'Italy'))
if country == "India":
    st.sidebar.write("""
The researcher for this country suggest using  **Decision Tree** 
""")
elif country == "Italy":
    st.sidebar.write("""
The researcher for this country suggest using  **Decision Tree** 
""")
elif country == "South Korea":
    st.sidebar.write("""
The researcher for this country suggest using  **Decision Tree** 
""")
elif country == "America":
    st.sidebar.write("""
The researcher for this country suggest using  **Decision Tree** 
""")

modelSelection = st.sidebar.selectbox(
    'Select Model', ('SVM', 'KNN', 'Decision Tree'))
if modelSelection == 'SVM':
    criteria = st.sidebar.selectbox(
        'Select Kernal', ('Linear', 'Poly', 'RBF', 'Sigmoid'))
elif modelSelection == 'KNN':
    criteria = st.sidebar.selectbox('Set Neighbour', ('1', '11', '21', '31'))
else:
    criteria = None
st.markdown("***")
if country == 'India':

    predictCase = st.sidebar.slider(
        'Predict the number of cases with actual cases based on day:', 1, 46, 91)
elif country == 'America':
    predictCase = st.sidebar.slider(
        'Predict the number of cases with actual cases based on day:', 1, 25, 50)
elif country == 'South Korea':
    predictCase = st.sidebar.slider(
        'Predict the number of cases with actual cases based on day:', 1, 17, 33)
else:
    predictCase = st.sidebar.slider(
        'Predict the number of cases with actual cases based on day:', 1, 30, 60)

# Process
# read csv file
if country == "India":
    train_data = pd.read_csv('dataset/covid19_india_training.csv')
    input_data_train = train_data.drop(
        columns=['429', 'Sno', 'Date', 'Time', 'State/UnionTerritory', 'Confirmed'])
    target_data_train = train_data['Confirmed']
    test_data = pd.read_csv('dataset/covid19_india_testing.csv')
    input_data_test = test_data.drop(
        columns=['851', 'Sno', 'Date', 'Time', 'State/UnionTerritory', 'Confirmed'])
    target_data_test = test_data['Confirmed']
    variableInput = "'ConfirmedIndianNational','ConfirmedForeignNational','Cured','Deaths'"
    variableTarget = "'Confirmed'"
    # Select Model from saved pickle(train model)
    if modelSelection == 'SVM':
        if criteria == 'RBF':
            regression = pickle.load(open('model/india/india_rbf_model', 'rb'))
        elif criteria == 'Linear':
            regression = pickle.load(
                open('model/india/india_linear_model', 'rb'))
        elif criteria == 'Poly':
            regression = pickle.load(
                open('model/india/india_poly_model', 'rb'))
        elif criteria == 'Sigmoid':
            regression = pickle.load(
                open('model/india/india_sigmoid_model', 'rb'))
    elif modelSelection == 'KNN':
        if criteria == '1':
            regression = pickle.load(open('model/india/india_1_model', 'rb'))
        elif criteria == '11':
            regression = pickle.load(open('model/india/india_11_model', 'rb'))
        elif criteria == '21':
            regression = pickle.load(open('model/india/india_21_model', 'rb'))
        elif criteria == '31':
            regression = pickle.load(open('model/india/india_31_model', 'rb'))
    else:
        tree = pickle.load(open('model/india/india_tree_model', 'rb'))
        
elif country == "America":
    train_data = pd.read_csv('dataset/us_covid19_daily_train.csv')
    input_data_train = train_data.drop(columns=['positive', 'date', 'states', 'hash', 'dateChecked', 'lastModified', 'hospitalizedCumulative', 'pending', 'inIcuCurrently', 'inIcuCumulative',
                                                'onVentilatorCurrently', 'onVentilatorCumulative', 'posNeg', 'deathIncrease', 'hospitalizedIncrease', 'negativeIncrease', 'positiveIncrease', 'totalTestResultsIncrease'])
    target_data_train = train_data['positive']
    test_data = pd.read_csv('dataset/us_covid19_daily_test.csv')
    input_data_test = test_data.drop(columns=['positive', 'date', 'states', 'hash', 'dateChecked', 'lastModified', 'hospitalizedCumulative', 'pending', 'inIcuCurrently', 'inIcuCumulative',
                                              'onVentilatorCurrently', 'onVentilatorCumulative', 'posNeg', 'deathIncrease', 'hospitalizedIncrease', 'negativeIncrease', 'positiveIncrease', 'totalTestResultsIncrease'])
    target_data_test = test_data['positive']
    variableInput = "'negative','hospitalizedCurrently','recovered','death','hospitalized','totalTestResults','total'"
    variableTarget = "'positive'"
    # Select Model
    if modelSelection == 'SVM':
        if criteria == 'RBF':
            regression = pickle.load(open('model/us/us_rbf_model', 'rb'))
        elif criteria == 'Linear':
            regression = pickle.load(open('model/us/us_linear_model', 'rb'))
        elif criteria == 'Poly':
            regression = pickle.load(open('model/us/us_poly_model', 'rb'))
        elif criteria == 'Sigmoid':
            regression = pickle.load(open('model/us/us_sigmoid_model', 'rb'))
    elif modelSelection == 'KNN':
        if criteria == '1':
            regression = pickle.load(open('model/us/us_1_model', 'rb'))
        elif criteria == '11':
            regression = pickle.load(open('model/us/us_11_model', 'rb'))
        elif criteria == '21':
            regression = pickle.load(open('model/us/us_21_model', 'rb'))
        elif criteria == '31':
            regression = pickle.load(open('model/us/us_31_model', 'rb'))
    else:
        tree = pickle.load(open('model/us/us_tree_model', 'rb'))
elif country == "Italy":
    train_data = pd.read_csv('dataset/covid19_italy_region_Training.csv')
    input_data_train = train_data.drop(columns=['SNo', 'Date', 'Country', 'RegionCode', 'RegionName', 'Latitude',
                                                'Longitude', 'IntensiveCarePatients', 'TotalPositiveCases', 'HomeConfinement', 'TestsPerformed'])
    target_data_train = train_data['TotalPositiveCases']
    test_data = pd.read_csv('dataset/covid19_italy_region_Testing.csv')
    input_data_test = test_data.drop(columns=['SNo', 'Date', 'Country', 'RegionCode', 'RegionName', 'Latitude',
                                              'Longitude', 'IntensiveCarePatients', 'TotalPositiveCases', 'HomeConfinement', 'TestsPerformed'])
    target_data_test = test_data['TotalPositiveCases']
    variableInput = "'HospitalizedPatients','TotalHospitalizedPatients','CurrentPositiveCases','NewPositiveCases','Recovered','Deaths'"
    variableTarget = "TotalPositiveCases"
    # Select Model
    if modelSelection == 'SVM':
        if criteria == 'RBF':
            regression = pickle.load(open('model/italy/italy_rbf_model', 'rb'))
        elif criteria == 'Linear':
            regression = pickle.load(
                open('model/italy/italy_linear_model', 'rb'))
        elif criteria == 'Poly':
            regression = pickle.load(
                open('model/italy/italy_poly_model', 'rb'))
        elif criteria == 'Sigmoid':
            regression = pickle.load(
                open('model/italy/italy_sigmoid_model', 'rb'))
    elif modelSelection == 'KNN':
        if criteria == '1':
            regression = pickle.load(open('model/italy/italy_1_model', 'rb'))
        elif criteria == '11':
            regression = pickle.load(open('model/italy/italy_11_model', 'rb'))
        elif criteria == '21':
            regression = pickle.load(open('model/italy/italy_21_model', 'rb'))
        elif criteria == '31':
            regression = pickle.load(open('model/italy/italy_31_model', 'rb'))
    else:
        tree = pickle.load(open('model/italy/italy_tree_model', 'rb'))

elif country == "South Korea":
    # st.write("""test""")
    train_data = pd.read_csv('dataset/korea_train.csv')
    input_data_train = train_data.drop(
        columns=['No.', 'date', 'time', 'confirmed'])
    target_data_train = train_data['confirmed']
    test_data = pd.read_csv('dataset/korea_test.csv')
    input_data_test = test_data.drop(
        columns=['No.', 'date', 'time', 'confirmed'])
    target_data_test = test_data['confirmed']
    variableInput = 'test,positive,negative,released,deceased'
    variableTarget = 'confirmed'
    # Select Model
    if modelSelection == 'SVM':
        if criteria == 'RBF':
            regression = pickle.load(open('model/korea/korea_rbf_model', 'rb'))
        elif criteria == 'Linear':
            regression = pickle.load(
                open('model/korea/korea_linear_model', 'rb'))
        elif criteria == 'Poly':
            regression = pickle.load(
                open('model/korea/korea_poly_model', 'rb'))
        elif criteria == 'Sigmoid':
            regression = pickle.load(
                open('model/korea/korea_sigmoid_model', 'rb'))
    elif modelSelection == 'KNN':
        if criteria == '1':
            regression = pickle.load(open('model/korea/korea_1_model', 'rb'))
        elif criteria == '11':
            regression = pickle.load(open('model/korea/korea_11_model', 'rb'))
        elif criteria == '21':
            regression = pickle.load(open('model/korea/korea_21_model', 'rb'))
        elif criteria == '31':
            regression = pickle.load(open('model/korea/korea_31_model', 'rb'))
    else:
        tree = pickle.load(open('model/korea/korea_tree_model', 'rb'))

# Find MSE
if (modelSelection == 'SVM' or modelSelection == 'KNN'):
    predicted_output = regression.predict(input_data_test)
    mse = mean_squared_error(predicted_output, target_data_test)
    MSE = str(mse)

# Output
st.write(""" 
## Results
""")

# calculate MSE
if modelSelection == 'SVM':
    st.write("""The MSE value for the prediction model SVM kernal  """ +
             criteria+""" is:  **"""+MSE+"""**""")
elif modelSelection == 'KNN':
    st.write("""The MSE value for the prediction model KNN numbner of neighbour """ +
             criteria+""" is:  **"""+MSE+"""**""")
else:
    predicted_output = tree.predict(input_data_test)
    mse = mean_squared_error(predicted_output, target_data_test)
    MSE = str(mse)
    st.write("""The MSE value for the decision tree is """ +MSE+"""""")
    df = pd.DataFrame({'Actual': target_data_test,
                      'Predicted': predicted_output})
    st.write(df)

# Display details about the dataset and prediction model
pCases = str(predictCase)
pOutput = str(predicted_output[predictCase-1].astype(int))
aOutput = str(target_data_test[predictCase-1].astype(int))
st.write("""Prediction for number of cases for day """ +
         pCases+""" : """+pOutput+""" Cases""")
st.write("""Actual for number of cases for day """ +
         pCases+""" : """+aOutput+""" Cases""")
st.write(""" 
### Table of comparison between actual and predicted number of cases
""")
df = pd.DataFrame({'Actual': target_data_test, 'Predicted': predicted_output})
st.write(df)
# Display comparison between algorithm
st.write(""" 
## Details of Comparison between algorithm
""")
st.write("""Install plotly incase the table are not appear (pip install dash)""")
if country == "India":
    st.write("""### k-Nearest Neighbour""")
    fig = go.Figure(data=[go.Table(header=dict(values=["Set of Neighbour",'Randomly Split', 'Manualy Split']),
                 cells=dict(values=[["1 Nearest Neighbour","11 Nearest Neighbour","21 Nearest Neighbour","31 Nearest Neighbour"],
                 [38.7111111, 168.8855831 , 282.5164525 , 355.072968], [38.54945055, 139.3713559, 244.9379781, 312.7723525]]))
                     ])
    st.write(fig)
    st.write("""The best result: 1 Nearest Neighbour""")
    st.write("""Explaination: Based on the table above the output result when comparing the split method, the 
value needs to find the best similar value to be chosen as the best k-neighbor to be compared 
with SVM and Decision Tree. The result table shows the best KNN to be compared with 
others algorithm is 1 Nearest Neighbor which MSE 38.54945055 for manually split and 
38.7111111 for randomly split. This is because the others k-neighbor technique has a big 
different value between manually split and randomly split. As a result, in KNN will be 
choosing randomly split in 1-Nearest Neighbor.""")
    st.write("""### Support Vector Regression""")
    fig = go.Figure(data=[go.Table(header=dict(values=["Kernel",'Randomly Split', 'Manualy Split']),
                 cells=dict(values=[["RBF","Linear","Sigmoid","Poly"],
                 [244.8823219, 0.008180545 , 745.3822712, 310.4004194], [569.7492923, 0.008397905, 1570.689184, 267.7788274]]))
                     ])
    st.write(fig)
    st.write("""The best result: Linear""")
    st.write("""Explaination: Based on the table above the output result when comparing the split method, the 
value needs to find the best similar value to be chosen as the best SVM kernel to be compared 
with KNN and Decision Tree. The result table shows the best SVM to be compared with 
others algorithm is Linear kernel which MSE 0.0084 for manually split and 0.0082 for 
randomly split. This is because the others kernel technique has a big different value between 
manually split and randomly split. For the Poly kernel the difference is more than 100 of 
MSE. For the RBF kernel the difference is more than 300 of MSE. Lastly, for the Sigmoid 
kernel the difference is more than 400 of MSE. As a result, in SVM will be choosing 
randomly split in linear kernel.
""")
    st.write("""### Decision Tree""")
    fig = go.Figure(data=[go.Table(header=dict(values=['Randomly Split', 'Manualy Split']),
                 cells=dict(values=[[1],[1]
                 ]))
                     ])
    st.write(fig)
    st.write("""The best result: Both""")
    # st.write("""Explaination: """)
    st.write("""### Comparison between algorithm""")
    fig = go.Figure(data=[go.Table(header=dict(values=["Model","Manually Split","Randomly Split"]),
                 cells=dict(values=[["SVM Linear","KNN Set 1","Decision Tree"],
                 [ 0.008397905 , 38.54945055, 1], [0.008180545, 38.7111111, 1]]))
                     ])
    st.write(fig)
    st.write("""The best result overall: Decision Tree Manual Split""")
    st.write("""Explaination: We can conclude 
the best model for predicting the confirmed cases in India is Decision Tree 
model with MSE value 1 for manually split and 1 for randomly split. 
Choosing by the manually sort split because manually are not biased 
because we are using the fixed dataset rather than using the randomly
train_test split library that are the result is biased.
""")
elif country == "America":
    st.write("""### k-Nearest Neighbour""")
    fig = go.Figure(data=[go.Table(header=dict(values=["Set of Neighbour",'Randomly Split', 'Manualy Split']),
                 cells=dict(values=[["1 Nearest Neighbour","11 Nearest Neighbour","21 Nearest Neighbour","31 Nearest Neighbour"],
                 [0.0, 8320109215.818905 , 48165068146.419624 , 142886368771.67838], [11037923909879.74, 12642314481469.068, 14140135465142.3, 15629446599289.967]]))
                     ])
    st.write(fig)
    st.write("""The best result: 1 Nearest Neighbour""")
    st.write("""Explaination: Based on table above, the MSE value result when comparing the split method is 
that the value needs to find the best value to be chosen as the best model in KNN to be 
compared with SVM and Decision Tree. This project will choose K nearest neighbor 1 as 
the best number of nearest neighbor with MSE value 11037923909879.74 for manually split 
and 0.0 for randomly split. This is because by using 1 nearest neighbor, the MSE value result 
is the best for predicting the confirmed positive cases after compared with other different 
number of nearest neighbors. As a result, for KNN model, the number of nearest neighbor 1 
will be compared with the SVM and Decision Tree model""")
    st.write("""### Support Vector Regression""")
    fig = go.Figure(data=[go.Table(header=dict(values=["Kernel",'Randomly Split', 'Manualy Split']),
                 cells=dict(values=[["RBF","Linear","Sigmoid","Poly"],
                 [13341280969055.424, 10569372659.615751 , 13341604508205.129, 13330743639749.688], [61369674951439.34, 129357494272.44592, 61370510179915.35, 61217098769813.26]]))
                     ])
    st.write(fig)
    st.write("""The best result: Linear""")
    st.write("""Explaination:  Based on the table above, the output result for MSE when comparing the split 
method, the value needs to find the best similar value to be chosen as the best SVM kernel 
to be compared with KNN and Decision Tree. The result table shows the best SVM to be 
compared with others algorithm is Linear kernel which the MSE for manually split is 
129357494272.44592 and for randomly split is 10569372659.615751. This is because the 
Linear kernel has the smallest MSE value for manually and randomly split to be compare 
with RBF, Sigmoid and Poly. The MSE value for RBF, Sigmoid and Poly kernel has biggest 
value and it makes them to not be the best kernel to compare with KNN and Decision Tree 
model. As a result, in SVM will be choosing randomly split in Linear kernel as it is the 
smallest value for MSE""")
    st.write("""### Decision Tree""")
    fig = go.Figure(data=[go.Table(header=dict(values=['Randomly Split', 'Manualy Split']),
                 cells=dict(values=[[11037923909879.74],[575334062]
                 ]))
                     ])
    st.write(fig)
    st.write("""The best result: Manual because time period prediction machine required fixed dataset""")
    # st.write("""Explaination: """)
    st.write("""### Comparison between algorithm""")
    fig = go.Figure(data=[go.Table(header=dict(values=["Model","Manually Split","Randomly Split"]),
                 cells=dict(values=[["SVM Linear","KNN Set 1","Decision Tree"],
                 [ 129357494272.44592 , 11037923909879.74, 575334062], [10569372659.615751, 0.0,11037923909879.74 ]]))
                     ])
    st.write(fig)
    st.write("""The best result overall: Decision Tree Manual Split because time period prediction machine required fixed dataset""")
    st.write("""Explaination:  The best model for predicting Covid-19 positive 
cases in the USA is the Decision Tree model, which has an MSE value of 
8028550.0 predicted for manually split and 224040.0 predicted for 
randomly split. As a result, the manual split method will be used to 
predict positive cases of Covid-19 in the United States, as the result for this 
method is the smallest among all of the 3 algorithms.Choosing by the manually sort split because manually are not biased because we are using the fixed dataset rather than using the randomly train_test split library that are the result is biased.""")
elif country == "Italy":
    st.write("""### k-Nearest Neighbour""")
    fig = go.Figure(data=[go.Table(header=dict(values=["Set of Neighbour",'Randomly Split', 'Manualy Split']),
                 cells=dict(values=[["1 Nearest Neighbour","11 Nearest Neighbour","21 Nearest Neighbour","31 Nearest Neighbour"],
                 [37206.51667 ,  1221718.294 ,  34916577.56 , 59952279.8], [244704976.1, 244704976.1, 244704976.1, 244704976.1]]))
                     ])
    st.write(fig)
    st.write("""The best result: 11 Nearest Neighbour""")
    st.write("""Explaination: Based on the table above the output result when comparing the split method, the 
value needs to find the best similar value to be chosen as the best k-neighbor to be compared 
with SVM and Decision Tree. The result table shows the best KNN to be compared with 
others algorithm is 11 Nearest Neighbor which MSE 1916180.661 for manually split and 
1221718.294 for randomly split. This is because the others k-neighbor technique has a big 
different value between manually split and randomly split. As a result, in KNN will be 
choosing randomly split in 11-Nearest Neighbor.""")
    st.write("""### Support Vector Regression""")
    fig = go.Figure(data=[go.Table(header=dict(values=["Kernel",'Randomly Split', 'Manualy Split']),
                 cells=dict(values=[["RBF","Linear","Sigmoid","Poly"],
                 [139243698.9, 0.00593514 , 139405297.8, 57748944.56], [61369674951439.34, 129357494272.44592, 61370510179915.35, 61217098769813.26]]))
                     ])
    st.write(fig)
    st.write("""The best result: Linear""")
    st.write("""Explaination: Based on the table above the output result when comparing the split method, the 
value needs to find the best similar value to be chosen as the best SVM kernel to be compared 
with KNN and Decision Tree. The result table shows the best SVM to be compared with 
others algorithm is Linear kernel which MSE 244704976.1 for manually split and 
0.00593514 for randomly split. This is because the others kernel technique has a big different 
value between manually split and randomly split. As a result, in SVM will be choosing
randomly split in linear kernel.""")
    st.write("""### Decision Tree""")
    fig = go.Figure(data=[go.Table(header=dict(values=['Randomly Split', 'Manualy Split']),
                 cells=dict(values=[[297.2],[1088.0]
                 ]))
                     ])
    st.write(fig)
    st.write("""The best result: Manual because time period prediction machine required fixed dataset""")
    # st.write("""Explaination: """)
    st.write("""### Comparison between algorithm""")
    fig = go.Figure(data=[go.Table(header=dict(values=["Model","Manually Split","Randomly Split"]),
                 cells=dict(values=[["SVM Linear","KNN Set 11","Decision Tree"],
                 [ 244704976.87 , 98758.51667, 1088.0], [0.00593514, 37206.51667,297.2 ]]))
                     ])
    st.write(fig)
    st.write("""The best result overall: Decision Tree Manual Split because time period prediction machine required fixed dataset""")
    st.write("""Explaination: We can conclude 
the best model for predicting the confirmed cases in Italy is DecisionTree 
model with MSE value 1088.0 for manually split and 297.2 for randomly 
split. Choosing by the manually sort split because manually are not biased 
because we are using the fixed dataset rather than using the randomly 
train_test split library that are the result is biased.""")
elif country == "South Korea":
    st.write("""### k-Nearest Neighbour""")
    fig = go.Figure(data=[go.Table(header=dict(values=["Set of Neighbour",'Randomly Split', 'Manualy Split']),
                 cells=dict(values=[["1 Nearest Neighbour","11 Nearest Neighbour","21 Nearest Neighbour","31 Nearest Neighbour"],
                 [20531.79, 43346.93 , 275075.3 , 489692.67], [54858.48, 96966.59, 370433.72, 841319.16]]))
                     ])
    st.write(fig)
    st.write("""The best result: 1 Nearest Neighbour""")
    st.write("""Explaination: Based on the table above, 1 nearest neighbour is the best k-neighbor in predicting while the 
KNN 31 nearest neighbour is the worst.""")
    st.write("""### Support Vector Regression""")
    fig = go.Figure(data=[go.Table(header=dict(values=["Kernel",'Randomly Split', 'Manualy Split']),
                 cells=dict(values=[["RBF","Linear","Sigmoid","Poly"],
                 [35919194.34, 7544334948 , 36028832.15, 33097237.87 ], [16712937.26, 9614113737, 16799320.25, 14624094.77]]))
                     ])
    st.write(fig)
    st.write("""The best result: Poly""")
    st.write("""Explaination: Based on the table above the output result when comparing the split method, the 
value needs to find the best similar value to be chosen as the best SVM kernel to be compared 
with KNN and Decision Tree. The result table shows the best SVM to be compared with 
others algorithm is Poly kernel which MSE 14624094.77for manually split and 33097237.87 
for randomly split. This is because the others kernel technique has a big different value 
between manually split and randomly split. As a result, in SVM will be choosing randomly 
split in linear kernel""")
    st.write("""### Decision Tree""")
    fig = go.Figure(data=[go.Table(header=dict(values=['Randomly Split', 'Manualy Split']),
                 cells=dict(values=[[51403.21],[51672.57]]))
                     ])
    st.write(fig)
    st.write("""The best result: Manual because time period prediction machine required fixed dataset""")
    st.write("""Explaination: Based on table above, the best result in predicting is randomly split with MSE 
value, 51403.21.""")
    st.write("""### Comparison between algorithm""")
    fig = go.Figure(data=[go.Table(header=dict(values=["Model","Manually Split","Randomly Split"]),
                 cells=dict(values=[["SVM Poly","KNN Set 1","Decision Tree"],
                 [ 14624094.77  , 54858.48, 51403.21], [33097237.87, 20531.79,51672.57 ]]))
                     ])
    st.write(fig)
    st.write("""The best result overall: Decision Tree Manual Split because time period prediction machine required fixed dataset""")
    st.write("""Explaination: We can conclude the best model 
for predicting the confirmed cases in South Korea is Decision Tree model with 
MSE value 51403.21 for manually split and 51672.57 for randomly split. 
Choosing by the manually sort split because manually are not biased because we 
are using the fixed dataset rather than using the randomly train_test split library 
that are the result is biase""")    
# Display details of training and testing dataset
st.write(""" 
## Details of Trainning and Testing Dataset
""")
st.write("""Variable target: """+variableTarget)
st.write("""Variable input: """+variableInput)
st.write(""" 
### Training Dataset
""")
st.write(train_data)
st.write(""" 
### Testing Dataset
""")
st.write(test_data)

