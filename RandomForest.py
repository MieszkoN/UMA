import pandas as pd
import numpy
import random
from collections import Counter


class RF_Tree():

    def __init__(
            self,
            Y,
            X,
            min_samples_split=None,
            max_depth=None,
            depth=None,
            X_features_fraction=None,
            node_type=None,
            rule=None
    ):

        self.Y = Y
        self.X = X
        self.min_samples_split = min_samples_split if min_samples_split else 20
        self.max_depth = max_depth if max_depth else 5
        self.depth = depth if depth else 0
        self.X_features_fraction = X_features_fraction if X_features_fraction is not None else 1.0
        self.node_type = node_type if node_type else 'root'
        self.rule = rule if rule else ""
        
        self.features = list(X.columns)
        self.counts = Counter(Y)
        self.gini_impurity = self.get_GINI()
        self.n_features = len(self.features)
        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))
        predictedClass = None
        if len(counts_sorted) > 0:
            predictedClass = counts_sorted[-1][0]

        self.predictedClass = predictedClass
        self.n = len(Y)
        
        self.left = None
        self.right = None
        self.best_feature = None
        self.best_value = None

    def get_random_X_colsample(self):

        n = int(self.n_features * self.X_features_fraction)
        features = random.sample(self.features, n)
        X = self.X[features].copy()
        return X

    @staticmethod
    def GINI_impurity(y1_count: int, y2_count: int) -> float:

        if y1_count is None:
            y1_count = 0

        if y2_count is None:
            y2_count = 0

        n = y1_count + y2_count

        if n == 0:
            return 0.0

        p1 = y1_count / n
        p2 = y2_count / n

        gini = 1 - (p1 ** 2 + p2 ** 2)

        return gini

    @staticmethod
    def ma(x: numpy.array, window: int) -> numpy.array:
        return numpy.convolve(x, numpy.ones(window), 'valid') / window

    def get_GINI(self):

        y1_count, y2_count = self.counts.get(0, 0), self.counts.get(1, 0)

        return self.GINI_impurity(y1_count, y2_count)

    def best_split(self) -> tuple:

        df = self.X.copy()
        df['Y'] = self.Y
        GINI_base = self.get_GINI()
        max_gain = 0
        best_feature = None
        best_value = None

        n = int(self.n_features * self.X_features_fraction)
        features_subsample = random.sample(self.features, n)

        for feature in features_subsample:

            Xdf = df.dropna().sort_values(feature)
            xmeans = self.ma(Xdf[feature].unique(), 2)

            for value in xmeans:

                left_counts = Counter(Xdf[Xdf[feature] < value]['Y'])
                right_counts = Counter(Xdf[Xdf[feature] >= value]['Y'])

                y0_left, y1_left, y0_right, y1_right = left_counts.get(0, 0), left_counts.get(1, 0), right_counts.get(0,0), right_counts.get(1, 0)

                gini_left = self.GINI_impurity(y0_left, y1_left)
                gini_right = self.GINI_impurity(y0_right, y1_right)

                n_left = y0_left + y1_left
                n_right = y0_right + y1_right

                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)

                wGINI = w_left * gini_left + w_right * gini_right

                GINIgain = GINI_base - wGINI

                if GINIgain > max_gain:
                    best_feature = feature
                    best_value = value

                    max_gain = GINIgain

        return (best_feature, best_value)

    def grow_tree(self):

        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):

            best_feature, best_value = self.best_split()

            if best_feature is not None:
                self.best_feature = best_feature
                self.best_value = best_value

                left_index, right_index = self.X[self.X[best_feature] <= best_value].index, self.X[
                    self.X[best_feature] > best_value].index

                left_X, right_X = self.X[self.X.index.isin(left_index)], self.X[self.X.index.isin(right_index)]

                left_X.reset_index(inplace=True, drop=True)
                right_X.reset_index(inplace=True, drop=True)

                left_Y, right_Y = [self.Y[x] for x in left_index], [self.Y[x] for x in right_index]

                left = RF_Tree(
                    left_Y,
                    left_X,
                    depth=self.depth + 1,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    node_type='left_node',
                    rule=f"{best_feature} <= {round(best_value, 3)}"
                )

                self.left = left
                self.left.grow_tree()

                right = RF_Tree(
                    right_Y,
                    right_X,
                    depth=self.depth + 1,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    node_type='right_node',
                    rule=f"{best_feature} > {round(best_value, 3)}"
                )

                self.right = right
                self.right.grow_tree()

    def predict(self, X: pd.DataFrame):

        predictions = []

        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})

            predictions.append(self.predict_obs(values))

        return predictions

    def predict_obs(self, values: dict) -> int:

        cur_node = self
        while cur_node.depth < cur_node.max_depth:

            best_feature = cur_node.best_feature
            best_value = cur_node.best_value

            if (cur_node.n < cur_node.min_samples_split) | (best_feature is None):
                break

            if (values.get(best_feature) < best_value):
                if self.left is not None:
                    cur_node = cur_node.left
            else:
                if self.right is not None:
                    cur_node = cur_node.right

        return cur_node.predictedClass


class RandomForestClassifier():

    def __init__(
            self,
            Y: list,
            X: pd.DataFrame,
            min_samples_split=None,
            max_depth=None,
            n_trees=None,
            X_features_fraction=None,
            X_obs_fraction=None
    ):

        self.Y = Y
        self.X = X

        self.min_samples_split = min_samples_split if min_samples_split else 20
        self.max_depth = max_depth if max_depth else 5

        self.features = list(X.columns)

        self.n_features = len(self.features)

        self.n_trees = n_trees if n_trees is not None else 30
        self.X_features_fraction = X_features_fraction if X_features_fraction is not None else 1.0
        self.X_obs_fraction = X_obs_fraction if X_obs_fraction is not None else 1.0

    def random_sample(self):

        X_rand_sample = self.X.sample(frac=self.X_obs_fraction, replace=True)
        indexes = X_rand_sample.index
        Y_rand_sample = [self.Y[x] for x in indexes]
        X_rand_sample.reset_index(inplace=True, drop=True)

        return X_rand_sample, Y_rand_sample

    def grow_random_forest(self):

        random_forest = []

        for _ in range(self.n_trees):
            X, Y = self.random_sample()

            tree = RF_Tree(
                Y=Y,
                X=X,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                X_features_fraction=self.X_features_fraction
            )

            tree.grow_tree()
            random_forest.append(tree)

        self.random_forest = random_forest

    def tree_predictions(self, X: pd.DataFrame) -> list:

        predictions = []
        for i in range(self.n_trees):
            predictedClass = self.random_forest[i].predict(X)
            predictions.append(predictedClass)

        return predictions

    def predict(self, X: pd.DataFrame) -> list:

        predictedClass = self.tree_predictions(X)

        n = X.shape[0]

        predictedClass_final = []

        for i in range(n):
            predictedClass_obs = [x[i] for x in predictedClass]

            counts = Counter(predictedClass_obs)
            most_common = counts.most_common(1)[0][0]

            predictedClass_final.append(most_common)

        return predictedClass_final
