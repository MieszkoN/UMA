import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from DecisionTree import Decision_Tree
import time

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
    Y = df_train['Survived'].values.tolist()

    X_train_titanic, X_test_titanic, Y_train_titanic, Y_test_titanic = train_test_split(X, Y, test_size = 0.2, random_state = 1, stratify = Y)



    #Predict values for decision tree for sklearn titanic dataset
    scikit_tree_start = time.time()
    decision_tree = DecisionTreeClassifier(max_depth = 3)
    decision_tree.fit(X_train_titanic, Y_train_titanic)
    Y_pred_standard = decision_tree.predict(X_test_titanic)
    scikit_tree_time = time.time() - scikit_tree_start


    #Predict values for custom implementation titanic dataset
    custom_tree_start = time.time()
    custom_decision_tree = Decision_Tree(X_train_titanic, Y_train_titanic, max_depth = 3)
    custom_decision_tree.build_tree()
    Y_pred_custom = custom_decision_tree.predict(X_test_titanic)
    custom_tree_time = time.time() - custom_tree_start



    print("--------------------------------Decision tree for Titanic dataset----------------------------------------------------------")
    print("Accuracy for standard implementation decision tree:", round(metrics.accuracy_score(Y_test_titanic, Y_pred_standard) * 100, 2))
    print("Accuracy for custom implementation of decision tree:", round(metrics.accuracy_score(Y_test_titanic, Y_pred_custom) * 100, 2))

    print(f"Standard tree action time: {scikit_tree_time}s")
    print(f"Custom tree action time: {custom_tree_time}s")
    print("---------------------------------------------------------------------------------------------------------------------------")




    #Preprocessing data for Parkinson disease dataset
    df = pd.read_csv('parkinson/parkinsons.csv')
    features_parkinson = df.loc[:, ~df.columns.isin([ 'status', 'name' ])]
    Y_parkinson = df['status']

    #Scaling values to 
    scaler = MinMaxScaler((-1,1))
    scaled_features = scaler.fit_transform(features_parkinson.values)
    X_parkinson = pd.DataFrame(scaled_features, index = features_parkinson.index, columns = features_parkinson.columns)

    X_train_parkinson, X_test_parkinson, Y_train_parkinson, Y_test_parkinson = train_test_split(X_parkinson, Y_parkinson, test_size = 0.2, random_state = 1, stratify = Y_parkinson)

    #Predict values for decision tree for sklearn Parkinson dataset
    scikit_tree_start_parkinson = time.time()
    decision_tree_parkinson = DecisionTreeClassifier(max_depth = 3)
    decision_tree_parkinson.fit(X_train_parkinson, Y_train_parkinson)
    Y_pred_standard_parkinson = decision_tree_parkinson.predict(X_test_parkinson)
    scikit_tree_time_parkinson = time.time() - scikit_tree_start_parkinson


    #Predict values for custom implementation Parkinson dataset
    custom_tree_start_parkinson = time.time()
    custom_decision_tree = Decision_Tree(X_train_parkinson, Y_train_parkinson, max_depth = 3)
    custom_decision_tree.build_tree()
    Y_pred_custom_parkinson = custom_decision_tree.predict(X_test_parkinson)
    custom_tree_time_parkinson = time.time() - custom_tree_start

    print("--------------------------------Decision tree for Parkinson dataset----------------------------------------------------------")
    print("Accuracy for standard implementation decision tree:", round(metrics.accuracy_score(Y_test_parkinson, Y_pred_standard_parkinson) * 100, 2))
    print("Accuracy for custom implementation of decision tree:", round(metrics.accuracy_score(Y_test_parkinson, Y_pred_custom_parkinson) * 100, 2))

    print(f"Standard tree action time: {scikit_tree_time_parkinson}s")
    print(f"Custom tree action time: {custom_tree_time_parkinson}s")
    print("---------------------------------------------------------------------------------------------------------------------------")