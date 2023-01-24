import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from DecisionTree import Decision_Tree
import time

from sklearn.ensemble import RandomForestClassifier
from RandomForest import RandomForestClassifier as CustomForest

if __name__ == '__main__':
    #Preprocessing for train dataset
    df_train = pd.read_csv("titanic/train.csv")
    df_train['Has_Cabin'] = df_train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
    df_train.drop('Cabin', axis=1, inplace=True)
    df_train['Age'] = df_train.Age.fillna(df_train.Age.median())
    df_train['Fare'] = df_train.Fare.fillna(df_train.Fare.median())
    df_train.dropna(inplace=True)
    df_train['Sex'] = df_train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    df_train['Embarked'] = df_train['Embarked'].map( {'C': 0, 'S': 1, 'Q': 2} ).astype(int)
    X = df_train[['Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Sex', 'Embarked', 'Has_Cabin']]
    Y = df_train['Survived']

    X_train_titanic, X_test_titanic, Y_train_titanic, Y_test_titanic = train_test_split(X, Y, test_size = 0.2, random_state = 1, stratify = Y)



    ############################
    ######### TITANIC ##########
    ############################

    ### DECISON TREE
    #Predict values for decision tree for sklearn titanic dataset
    print("-TITANIC-")
    scikit_tree_start = time.time()
    decision_tree = DecisionTreeClassifier(max_depth = 3)
    decision_tree.fit(X_train_titanic, Y_train_titanic)
    Y_pred_standard = decision_tree.predict(X_test_titanic)
    scikit_tree_time = time.time() - scikit_tree_start
    print("scikit decison tree finished...")

    #Predict values for custom implementation titanic dataset
    custom_tree_start = time.time()
    custom_decision_tree = Decision_Tree(X_train_titanic, Y_train_titanic, max_depth = 3)
    custom_decision_tree.build_tree()
    Y_pred_custom = custom_decision_tree.predict(X_test_titanic)
    custom_tree_time = time.time() - custom_tree_start
    print("custom decison tree finished...")

    ######################################


    scikit_tree_start2 = time.time()
    decision_tree2 = DecisionTreeClassifier(max_depth = 20)
    decision_tree2.fit(X_train_titanic, Y_train_titanic)
    Y_pred_standard2 = decision_tree2.predict(X_test_titanic)
    scikit_tree_time2 = time.time() - scikit_tree_start2
    print("scikit decison tree finished...")

    #Predict values for custom implementation titanic dataset
    custom_tree_start2 = time.time()
    custom_decision_tree2 = Decision_Tree(X_train_titanic, Y_train_titanic, max_depth = 20)
    custom_decision_tree2.build_tree()
    Y_pred_custom2 = custom_decision_tree2.predict(X_test_titanic)
    custom_tree_time2 = time.time() - custom_tree_start
    print("custom decison tree finished...")

    #######################################

    ### RANDOM FOREST
    #Predict values for random forest for sklearn titanic dataset
    '''scikit_forest_start = time.time()
    random_forest = RandomForestClassifier(n_estimators = 100,max_depth = 3)
    random_forest.fit(X_train_titanic, Y_train_titanic)
    Y_pred_forest = random_forest.predict(X_test_titanic)
    scikit_forest_time = time.time() - scikit_forest_start
    print("scikit random forest finished...")

    #Predict values for custom implementation titanic dataset
    custom_forest_start = time.time()
    custom_forest_tree = CustomForest( X_train_titanic, Y_train_titanic,max_depth = 3,n_estimators = 100)
    custom_forest_tree.grow_random_forest()
    Y_pred_custom_forest = custom_forest_tree.predict(X_test_titanic)
    custom_forest_time = time.time() - custom_forest_start
    print("custom random forest finished...")'''


    print("--------------------------------Decision tree for Titanic dataset----------------------------------------------------------")
    print("Accuracy for standard implementation decision tree:", round(metrics.accuracy_score(Y_test_titanic, Y_pred_standard) * 100, 2))
    print("Accuracy for custom implementation of decision tree:", round(metrics.accuracy_score(Y_test_titanic, Y_pred_custom) * 100, 2))
    print(f"Standard tree action time: {scikit_tree_time}s")
    print(f"Custom tree action time: {custom_tree_time}s")
    print("---------------------------------------------------------------------------------------------------------------------------")

    print("-------------------------------- Random forest for Titanic dataset --------------------------------------------------------")
    #print("Accuracy for standard implementation of random forest:",round(metrics.accuracy_score(Y_test_titanic, Y_pred_forest) * 100, 2))
    #print("Accuracy for custom implementation of random forest:",round(metrics.accuracy_score(Y_test_titanic, Y_pred_custom_forest) * 100, 2))
    #print(f"Standard random forest action time: {scikit_forest_time}s")
    #print(f"Custom random forest action time: {custom_forest_time}s")
    print("---------------------------------------------------------------------------------------------------------------------------")


    ############################
    ######## PARKINSON #########
    ############################

    #Preprocessing data for Parkinson disease dataset
    df = pd.read_csv('parkinson/parkinsons.csv')
    features_parkinson = df.loc[:, ~df.columns.isin([ 'status', 'name' ])]
    Y_parkinson = df['status']

    #Scaling values to 
    scaler = MinMaxScaler((-1,1))
    scaled_features = scaler.fit_transform(features_parkinson.values)
    X_parkinson = pd.DataFrame(scaled_features, index = features_parkinson.index, columns = features_parkinson.columns)

    X_train_parkinson, X_test_parkinson, Y_train_parkinson, Y_test_parkinson = train_test_split(X_parkinson, Y_parkinson, test_size = 0.2, random_state = 1, stratify = Y_parkinson)

    print("-PARKINSON-")
    #Predict values for decision tree for sklearn Parkinson dataset
    scikit_tree_start_parkinson = time.time()
    decision_tree_parkinson = DecisionTreeClassifier(max_depth = 3)
    decision_tree_parkinson.fit(X_train_parkinson, Y_train_parkinson)
    Y_pred_standard_parkinson = decision_tree_parkinson.predict(X_test_parkinson)
    scikit_tree_time_parkinson = time.time() - scikit_tree_start_parkinson
    print("scikit decison tree finished...")

    #Predict values for custom implementation Parkinson dataset
    custom_tree_start_parkinson = time.time()
    custom_decision_tree = Decision_Tree(X_train_parkinson, Y_train_parkinson, max_depth = 3)
    custom_decision_tree.build_tree()
    Y_pred_custom_parkinson = custom_decision_tree.predict(X_test_parkinson)
    custom_tree_time_parkinson = time.time() - custom_tree_start
    print("custom decison tree finished...")

    #Predict values for sklearn random forest for Parkinson dataset
    '''scikit_random_forest_start_parkinson = time.time()
    scikit_random_forest_parkinson = RandomForestClassifier(max_depth = 3, n_estimators = 100)
    scikit_random_forest_parkinson.fit(X_train_parkinson, Y_train_parkinson)
    Y_pred_scikit_random_forest_parkinson = scikit_random_forest_parkinson.predict(X_test_parkinson)
    scikit_random_forest_time_parkinson = time.time() - scikit_random_forest_start_parkinson
    print("scikit random forest finished...")'''

    #Predict values for custom implementation of random forest for Parkinson dataset
    '''custom_forest_start_parkinson = time.time()
    custom_forest_tree_parkinson = CustomForest( X_train_parkinson, Y_train_parkinson, max_depth = 3, n_estimators = 100)
    custom_forest_tree_parkinson.grow_random_forest()
    Y_pred_custom_forest_parkinson = custom_forest_tree_parkinson.predict(X_test_parkinson)
    custom_forest_time_parkinson = time.time() - custom_forest_start_parkinson
    print("custom random forest finished...")'''


    print("--------------------------------Decision tree for Parkinson dataset----------------------------------------------------------")
    print("Accuracy for standard implementation decision tree:", round(metrics.accuracy_score(Y_test_parkinson, Y_pred_standard_parkinson) * 100, 2))
    print("Accuracy for custom implementation of decision tree:", round(metrics.accuracy_score(Y_test_parkinson, Y_pred_custom_parkinson) * 100, 2))

    print("Precison for standard implementation decision tree:", round(metrics.precision_score(Y_test_parkinson, Y_pred_standard_parkinson) * 100, 2))
    print("Precison for custom implementation of decision tree:", round(metrics.precision_score(Y_test_parkinson, Y_pred_custom_parkinson) * 100, 2))

    print("Recall for standard implementation decision tree:", round(metrics.recall_score(Y_test_parkinson, Y_pred_standard_parkinson) * 100, 2))
    print("Recall for custom implementation of decision tree:", round(metrics.recall_score(Y_test_parkinson, Y_pred_custom_parkinson) * 100, 2))

    print(f"Standard tree action time: {scikit_tree_time_parkinson}s")
    print(f"Custom tree action time: {custom_tree_time_parkinson}s")
    print("---------------------------------------------------------------------------------------------------------------------------")

    print("-------------------------------- Random forest for Parkinson dataset-------------------------------------------------------")
    #print("Accuracy for standard implementation random forest:", round(metrics.accuracy_score(Y_test_parkinson, Y_pred_scikit_random_forest_parkinson) * 100, 2))
    #print("Accuracy for custom implementation of random forest:", round(metrics.accuracy_score(Y_test_parkinson,  Y_pred_custom_forest_parkinson) * 100, 2))

    #print(f"Standard tree action time: {scikit_random_forest_time_parkinson}s")
    #print(f"Custom tree action time: {custom_forest_time_parkinson}s")
    print("---------------------------------------------------------------------------------------------------------------------------")



    '''values = [[round(metrics.accuracy_score(Y_test_titanic, Y_pred_standard) * 100, 2),
               round(metrics.accuracy_score(Y_test_titanic, Y_pred_custom) * 100, 2),
               round(metrics.accuracy_score(Y_test_titanic, Y_pred_forest) * 100, 2),
               round(metrics.accuracy_score(Y_test_titanic, Y_pred_custom_forest) * 100, 2)],
              [round(metrics.precision_score(Y_test_titanic, Y_pred_standard) * 100, 2),
               round(metrics.precision_score(Y_test_titanic, Y_pred_custom) * 100, 2),
               round(metrics.precision_score(Y_test_titanic, Y_pred_forest) * 100, 2),
               round(metrics.precision_score(Y_test_titanic, Y_pred_custom_forest) * 100, 2)],
              [round(metrics.recall_score(Y_test_titanic, Y_pred_standard) * 100, 2),
               round(metrics.recall_score(Y_test_titanic, Y_pred_custom) * 100, 2),
               round(metrics.recall_score(Y_test_titanic, Y_pred_forest) * 100, 2),
               round(metrics.recall_score(Y_test_titanic, Y_pred_custom_forest) * 100, 2)]]
    col = ['Scikit Drzewo Decyzyjne', 'Własna impelementacja Drzewa Decyzyjnego', 'Scikit Las Losowy', 'Własna impelementacja Lasu Losowego']
    names = ['Accuracy', 'Precison', 'Recall']
    df2 = pd.DataFrame(values, columns=col, index=names)
    df2.plot(kind='bar');
    plt.legend(loc='upper right')

    plt.title("Titanic")

    values_parkinson   = [[metrics.mean_absolute_error(Y_test_parkinson, Y_pred_standard_parkinson),
                           metrics.mean_absolute_error(Y_test_parkinson, Y_pred_custom_parkinson),
                           metrics.mean_absolute_error(Y_test_parkinson, Y_pred_scikit_random_forest_parkinson),
                           metrics.mean_absolute_error(Y_test_parkinson,  Y_pred_custom_forest_parkinson)],
                          [metrics.mean_squared_error(Y_test_parkinson, Y_pred_standard_parkinson),
                           metrics.mean_squared_error(Y_test_parkinson, Y_pred_custom_parkinson),
                           metrics.mean_squared_error(Y_test_parkinson, Y_pred_scikit_random_forest_parkinson),
                           metrics.mean_squared_error(Y_test_parkinson, Y_pred_custom_forest_parkinson)],
                          [metrics.r2_score(Y_test_parkinson, Y_pred_standard_parkinson),
                           metrics.r2_score(Y_test_parkinson, Y_pred_custom_parkinson),
                           metrics.r2_score(Y_test_parkinson, Y_pred_scikit_random_forest_parkinson),
                           metrics.r2_score(Y_test_parkinson, Y_pred_custom_forest_parkinson)]]
    col = ['Scikit Drzewo Decyzyjne', 'Własna impelementacja Drzewa Decyzyjnego', 'Scikit Las Losowy', 'Własna impelementacja Lasu Losowego']
    names = ['Mean \nabsolute \nerror', 'Mean \nsquared \nerror', 'r2 score']
    df_p = pd.DataFrame(values_parkinson, columns=col, index=names)
    df_p.plot(kind='bar');
    plt.title("Parkinson")






    values_time   = [[scikit_tree_time,
                           custom_tree_time,
                           scikit_forest_time,
                           custom_forest_time],
                          [scikit_tree_time_parkinson,
                           custom_tree_time_parkinson,
                           scikit_random_forest_time_parkinson,
                           custom_forest_time_parkinson]]
    col = ['Scikit Drzewo Decyzyjne', 'Własna impelementacja Drzewa Decyzyjnego', 'Scikit Las Losowy', 'Własna impelementacja Lasu Losowego']
    names = ['Titanic', 'Parkinson']
    df_t = pd.DataFrame(values_time, columns=col, index=names)
    df_t.plot(kind='bar')
    plt.title("Czas wykoanania")'''


    values = [[round(metrics.accuracy_score(Y_test_titanic, Y_pred_standard) * 100, 2),
               round(metrics.accuracy_score(Y_test_titanic, Y_pred_custom) * 100, 2),
               round(metrics.accuracy_score(Y_test_titanic, Y_pred_standard2) * 100, 2),
               round(metrics.accuracy_score(Y_test_titanic, Y_pred_custom2) * 100, 2)],
              [round(metrics.precision_score(Y_test_titanic, Y_pred_standard) * 100, 2),
               round(metrics.precision_score(Y_test_titanic, Y_pred_custom) * 100, 2),
               round(metrics.precision_score(Y_test_titanic, Y_pred_standard2) * 100, 2),
               round(metrics.precision_score(Y_test_titanic, Y_pred_custom2) * 100, 2)],
              [round(metrics.recall_score(Y_test_titanic, Y_pred_standard) * 100, 2),
               round(metrics.recall_score(Y_test_titanic, Y_pred_custom) * 100, 2),
               round(metrics.recall_score(Y_test_titanic, Y_pred_standard2) * 100, 2),
               round(metrics.recall_score(Y_test_titanic, Y_pred_custom2) * 100, 2)]]
    col = ['Scikit Drzewo Decyzyjne głębokość=3', 'Własna impelementacja Drzewa Decyzyjnego głębokość=3', 'Scikit Drzewo Decyzyjne głębokośc=20', 'Własna impelementacja Drzewa Decyzyjnego głębokość=20']
    names = ['Accuracy', 'Precison', 'Recall']
    df2 = pd.DataFrame(values, columns=col, index=names)
    df2.plot(kind='bar')
    plt.legend(loc='lower right')



    plt.title("Titanic")

    values_parkinson   = [[metrics.mean_absolute_error(Y_test_parkinson, Y_pred_standard_parkinson),
                           metrics.mean_absolute_error(Y_test_parkinson, Y_pred_custom_parkinson)],
                          [metrics.mean_squared_error(Y_test_parkinson, Y_pred_standard_parkinson),
                           metrics.mean_squared_error(Y_test_parkinson, Y_pred_custom_parkinson)],
                          [metrics.r2_score(Y_test_parkinson, Y_pred_standard_parkinson),
                           metrics.r2_score(Y_test_parkinson, Y_pred_custom_parkinson)]]
    col = ['Scikit Drzewo Decyzyjne', 'Własna impelementacja Drzewa Decyzyjnego']
    names = ['Mean \nabsolute \nerror', 'Mean \nsquared \nerror', 'r2 score']
    df_p = pd.DataFrame(values_parkinson, columns=col, index=names)
    df_p.plot(kind='bar')
    plt.title("Parkinson")


    plt.show()