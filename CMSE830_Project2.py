import streamlit as st
import pandas as pd
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)

df_project = pd.read_csv('Murders_2010s.csv')

# pd.unique(df_project['Perpetrator Age'])

df_project['Perpetrator Age'].astype('str')
mask = df_project['Perpetrator Age'] == ' '
# mask
# df_project[mask]

df_project['Perpetrator Age'][634666] = np.nan

# df_project.iloc[634666]['Perpetrator Age']

df_project.dropna(subset=['Perpetrator Age'], inplace = True)

df_project['Perpetrator Age'].astype('float')
year_mask = df_project['Year'] >= 2010

df_2010s = df_project[year_mask]

df_2010s.reset_index(drop=True, inplace=True)

df_trunc = df_project[['State', 'Year', 'Month', 'Incident', 'Crime Solved', 'Victim Age', 'Victim Sex', 'Victim Ethnicity', 'Perpetrator Sex', 'Perpetrator Age', 'Perpetrator Race', 'Relationship', 'Weapon']]

# df_trunc

# pd.unique(df_trunc['Perpetrator Age'])

df_trunc['Perpetrator Age'] = df_trunc['Perpetrator Age'].astype(int)

# pd.unique(df_trunc['Perpetrator Age'])

df_trunc10s = df_2010s[['State', 'Year', 'Month', 'Incident', 'Crime Solved', 'Victim Age', 'Victim Sex', 'Victim Ethnicity', 'Perpetrator Sex', 'Perpetrator Age', 'Perpetrator Race', 'Relationship', 'Weapon']]

# df_trunc10s

# pd.unique(df_trunc10s['Perpetrator Age'])

df_trunc10s['Perpetrator Age'] = df_trunc10s['Perpetrator Age'].astype(int)

# pd.unique(df_trunc10s['Perpetrator Age'])

df_final = df_2010s[['State', 'Year', 'Month', 'Incident', 'Crime Solved', 'Victim Age', 'Victim Sex', 'Victim Ethnicity', 'Weapon']]

df_data = df_trunc.apply(lambda col: pd.factorize(col, sort=False)[0])

df_data10s = df_trunc10s.apply(lambda col: pd.factorize(col, sort=False)[0])
df_fdata = df_final.apply(lambda col: pd.factorize(col, sort=False)[0])

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#X = df_data.drop("Crime Solved", axis=1)
X = df_data10s.drop("Crime Solved", axis=1)
# How to create a streamlit slider where I add or subtract columns that I test with?

#y = df_data['Crime Solved']
y = df_data10s['Crime Solved']

#X = df_data.drop("Crime Solved", axis=1)
X1 = df_fdata.drop("Crime Solved", axis=1)
# How to create a streamlit slider where I add or subtract columns that I test with?

#y = df_data['Crime Solved']
y1 = df_fdata['Crime Solved']

start_state = 42
test_fraction = 0.3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction, random_state=start_state)

my_scaler = StandardScaler()
my_scaler.fit(X_train)

X_train_scaled = my_scaler.transform(X_train)

X_test_scaled = my_scaler.transform(X_test)

# X_train.shape, X_test.shape, y_train.shape, y_test.shape

start_state1 = 42
test_fraction1 = 0.3

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=test_fraction1, random_state=start_state1)

my_scaler1 = StandardScaler()
my_scaler1.fit(X1_train)

X1_train_scaled = my_scaler1.transform(X1_train)

X1_test_scaled = my_scaler1.transform(X1_test)

# X1_train.shape, X1_test.shape, y1_train.shape, y1_test.shape

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.title("Can Machine Learning Help Solve Crimes?")

tab1, tab2, tab3, tab4 = st.tabs(["Data", "First Test", "Second Test", "Scores"])

with tab1:
    st.header("Cleaned up Database")
    st.dataframe(df_trunc10s)

clf_svm = svm.SVC(gamma=0.001)
clf_neighbor = KNeighborsClassifier(5)
clf_Boost = AdaBoostClassifier()
clf_forest = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
clf_naive = GaussianNB()
clf_MLP = MLPClassifier(alpha=1, max_iter=1000)
clf_Quad = QuadraticDiscriminantAnalysis()

clf_svm.fit(X_train, y_train);
clf_neighbor.fit(X_train, y_train);
clf_Boost.fit(X_train, y_train);
clf_forest.fit(X_train, y_train);
clf_naive.fit(X_train, y_train);
clf_MLP.fit(X_train, y_train);
clf_Quad.fit(X_train, y_train);

predicted_svm = clf_svm.predict(X_test)
predicted_KN = clf_neighbor.predict(X_test)
predicted_Boost = clf_Boost.predict(X_test)
predicted_forest = clf_forest.predict(X_test)
predicted_naive = clf_naive.predict(X_test)
predicted_MLP = clf_MLP.predict(X_test)
predicted_Quad = clf_Quad.predict(X_test)

print(predicted_svm)
print(predicted_KN)
print(predicted_Boost)
print(predicted_forest)
print(predicted_naive)
print(predicted_MLP)
print(predicted_Quad)

Score_svm = precision_score(y_test, predicted_svm, average='micro')
Score_KN = precision_score(y_test, predicted_KN, average='micro')
Score_Boost = precision_score(y_test, predicted_Boost, average='micro')
Score_forest = precision_score(y_test, predicted_forest, average='micro')
Score_naive = precision_score(y_test, predicted_naive, average='micro')
Score_MLP = precision_score(y_test, predicted_MLP, average='micro')
Score_Quad = precision_score(y_test, predicted_Quad, average='micro')

with tab2:
    st.header("Confusion Matrices of Predictions")
    st.text("")
    st.subheader("SVM Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_test, predicted_svm).plot()
    st.pyplot()
    st.text("")

    st.metric(label="SVM Score", value= Score_svm)

    st.text("")
    st.subheader("K Nearest Neighbors Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_test, predicted_KN).plot()
    st.pyplot()
    st.text("")

    st.metric(label="K Nearest Neighbors Score", value= Score_KN)

    st.text("")
    st.subheader("Ada Boost Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_test, predicted_Boost).plot()
    st.pyplot()
    st.text("")

    st.metric(label="Ada Boost Score", value= Score_Boost)

    st.text("")
    st.subheader("Random Forest Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_test, predicted_forest).plot()
    st.pyplot()
    st.text("")

    st.metric(label="Random Forest Score", value= Score_forest)

    st.text("")
    st.subheader("Gaussian NB Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_test, predicted_naive).plot()
    st.pyplot()
    st.text("")

    st.metric(label="Gaussian NB Score", value= Score_naive)

    st.text("")
    st.subheader("MLP Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_test, predicted_MLP).plot()
    st.pyplot()
    st.text("")

    st.metric(label="MLP Score", value= Score_MLP)

    st.text("")
    st.subheader("Quad Descriminant Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_test, predicted_Quad).plot()
    st.pyplot()
    st.text("")

    st.metric(label="Quad Descriminant", value= Score_Quad)


    #disp2 = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted_Boost)
#disp.figure_.suptitle("Confusion Matrix")

#disp3 = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted_forest)
#disp.figure_.suptitle("Confusion Matrix")

#disp4 = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted_naive)
#disp.figure_.suptitle("Confusion Matrix")

#disp5 = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted_MLP)
#disp.figure_.suptitle("Confusion Matrix")

#disp6 = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted_Quad)
#disp.figure_.suptitle("Confusion Matrix")
    #st.metric(label="KNeighbors Score", value= Score_KN)
    #st.pyplot(disp2)
    #st.metric(label="AdaBoost Score", value= Score_Boost)
    #st.pyplot(disp3)
    #st.metric(label="RandomForest Score", value= Score_forest)
    #st.pyplot(disp4)
    #st.metric(label="GaussianNB Score", value= Score_naive)
    #st.pyplot(disp5)
    #st.metric(label="MLP Score", value= Score_MLP)
    #st.pyplot(disp6)
    #st.metric(label="QuadDiscriminant Score", value= Score_Quad)
    st.subheader("Scores look great! but..")
    st.subheader("Why is this too good to be true?")

# predicted_svm.shape
# predicted_KN.shape

clf_svm.fit(X1_train, y1_train);
clf_neighbor.fit(X1_train, y1_train);
clf_Boost.fit(X1_train, y1_train);
clf_forest.fit(X1_train, y1_train);
clf_naive.fit(X1_train, y1_train);
clf_MLP.fit(X1_train, y1_train);
clf_Quad.fit(X1_train, y1_train);

predicted_svm1 = clf_svm.predict(X1_test)
predicted_KN1 = clf_neighbor.predict(X1_test)
predicted_Boost1 = clf_Boost.predict(X1_test)
predicted_forest1 = clf_forest.predict(X1_test)
predicted_naive1 = clf_naive.predict(X1_test)
predicted_MLP1 = clf_MLP.predict(X1_test)
predicted_Quad1 = clf_Quad.predict(X1_test)

disp7 = plot_confusion_matrix(clf_svm, X1_test, y1_test)  
disp8 = plot_confusion_matrix(clf_neighbor, X1_test, y1_test)
disp9 = plot_confusion_matrix(clf_Boost, X1_test, y1_test)
disp10 = plot_confusion_matrix(clf_forest, X1_test, y1_test)
disp11 = plot_confusion_matrix(clf_naive, X1_test, y1_test)
disp12 = plot_confusion_matrix(clf_MLP, X1_test, y1_test)
disp13 = plot_confusion_matrix(clf_Quad, X1_test, y1_test)

print(predicted_svm1)
print(predicted_KN1)
print(predicted_Boost1)
print(predicted_forest1)
print(predicted_naive1)
print(predicted_MLP1)
print(predicted_Quad1)

#Score_svm = precision_score(y_test, predicted_svm, average='micro')
#Score_KN = precision_score(y_test, predicted_KN, average='micro')
#Score_Boost = precision_score(y_test, predicted_Boost, average='micro')
#Score_forest = precision_score(y_test, predicted_forest, average='micro')
#Score_naive = precision_score(y_test, predicted_naive, average='micro')
#Score_MLP = precision_score(y_test, predicted_MLP, average='micro')
#Score_Quad = precision_score(y_test, predicted_Quad, average='micro')

Score_svm1 = precision_score(y1_test, predicted_svm1, average='micro')
Score_KN1 = precision_score(y1_test, predicted_KN1, average='micro')
Score_Boost1 = precision_score(y1_test, predicted_Boost1, average='micro')
Score_forest1 = precision_score(y1_test, predicted_forest1, average='micro')
Score_naive1 = precision_score(y1_test, predicted_naive1, average='micro')
Score_MLP1 = precision_score(y1_test, predicted_MLP1, average='micro')
Score_Quad1 = precision_score(y1_test, predicted_Quad1, average='micro')

Delta = Score_svm - Score_svm1
Delta1 = Score_KN - Score_KN1
Delta2 = Score_Boost - Score_Boost1
Delta3 = Score_forest - Score_forest1
Delta4 = Score_naive - Score_naive1
Delta5 = Score_MLP - Score_MLP1
Delta6 = Score_Quad - Score_Quad1

fig = plt.figure(figsize = (10, 5))

data = {'SVM':Score_svm1, 'KNeighbors':Score_KN1, 'AdaBoost':Score_Boost1, 'RandomForest':Score_forest1, 'GaussianNB':Score_naive1, 'MLP':Score_MLP1, 'QuadDiscriminant':Score_Quad1}
Classifiers = list(data.keys())
Scores = list(data.values())

plt.bar(Classifiers, Scores, color ='maroon', width = 0.4)

plt.xlabel("Classifiers")
plt.ylabel("Scores")
plt.title("Prediction Scores For Classifiers")

#st.title("Can Machine Learning Help Solve Crimes?")

#tab1, tab2, tab3, tab4 = st.tabs(["Data", "First Test", "Second Test", "Scores"])

#with tab1:
#    st.header("Cleaned up Database")
#    st.dataframe(df_trunc10s)

#with tab2:
#    st.header("Confusion Matrices of Predictions")
#    st.pyplot(disp)
#    st.metric(label="SVM Score", value= Score_svm)
#    st.pyplot(disp1)
#    st.metric(label="KNeighbors Score", value= Score_KN)
#    st.pyplot(disp2)
#    st.metric(label="AdaBoost Score", value= Score_Boost)
#    st.pyplot(disp3)
#    st.metric(label="RandomForest Score", value= Score_forest)
#    st.pyplot(disp4)
#    st.metric(label="GaussianNB Score", value= Score_naive)
#    st.pyplot(disp5)
#    st.metric(label="MLP Score", value= Score_MLP)
#    st.pyplot(disp6)
#    st.metric(label="QuadDiscriminant Score", value= Score_Quad)
#    st.subheader("Scores look great! but..")
#    st.subheader("Why is this too good to be true?")

with tab3:
    st.header("Much More Reasonable Results")
    st.dataframe(df_final)
    st.header("Confusion Matrices of Predictions")

    st.text("")
    st.subheader("SVM Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_test, predicted_svm1).plot()
    st.pyplot()
    st.text("")

    st.metric(label="SVM Score", value= Score_svm1, delta = -Delta)

    st.text("")
    st.subheader("K Nearest Neighbors Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_test, predicted_KN1).plot()
    st.pyplot()
    st.text("")

    st.metric(label="K Nearest Neighbors Score", value= Score_KN1, delta = -Delta1)

    st.text("")
    st.subheader("Ada Boost Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_test, predicted_Boost1).plot()
    st.pyplot()
    st.text("")

    st.metric(label="Ada Boost Score", value= Score_Boost1, delta = -Delta2)

    st.text("")
    st.subheader("Random Forest Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_test, predicted_forest1).plot()
    st.pyplot()
    st.text("")

    st.metric(label="Random Forest Score", value= Score_forest1, delta = -Delta3)

    st.text("")
    st.subheader("Gaussian NB Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_test, predicted_naive1).plot()
    st.pyplot()
    st.text("")

    st.metric(label="Gaussian NB Score", value= Score_naive1, delta = -Delta4)

    st.text("")
    st.subheader("MLP Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_test, predicted_MLP1).plot()
    st.pyplot()
    st.text("")

    st.metric(label="MLP Score", value= Score_MLP1, delta = -Delta5)

    st.text("")
    st.subheader("Quad Descriminant Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_test, predicted_Quad1).plot()
    st.pyplot()
    st.text("")

    st.metric(label="Quad Descriminant", value= Score_Quad1, delta = -Delta6)
    st.subheader("Much worse results but more realistic")
    st.text("The problem was, if we know the perpetrators information then obviously")
    st.text( "the crime was solved!")
    st.text("So that information was positively skewing our predictions")
    
with tab4:
    st.header("Classifier Scores")
    st.pyplot(fig)
    st.text("")
    
    st.subheader("We can see that the ADA Boost classifier performed the best")
    
    st.text("")
    st.subheader("So to answer our question from the beginning... for this database machine learning did not help us solve crimes haha")
