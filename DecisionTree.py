from collections import Counter
import pandas as pd 
import numpy as np 


class Decision_Tree: 
    def __init__(
        self, 
        X: pd.DataFrame,
        Y: list,
        min_samples_split=None,
        max_depth=None,
        depth=None
    ):
        self.X = X
        self.Y = Y 

        self.min_samples_split = min_samples_split if min_samples_split else 20
        self.max_depth = max_depth if max_depth else 10

        self.depth = depth if depth else 0
        self.features = list(self.X.columns)

        self.counts = Counter(Y)
        self.gini_impurity = self.get_GINI_impurity()
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

    @staticmethod
    def calculate_GINI_impurity(y1_count: int, y2_count: int) -> float:
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
    def calculate_moving_average(x: np.array, window: int) -> np.array:
        return np.convolve(x, np.ones(window), 'valid') / window

    def get_GINI_impurity(self):
        y1_count, y2_count = self.counts.get(0, 0), self.counts.get(1, 0)
        return self.calculate_GINI_impurity(y1_count, y2_count)

    def generate_best_split(self) -> tuple:
        df = self.X.copy()
        df['Y'] = self.Y

        GINI_base = self.get_GINI_impurity()
        max_gain = 0

        best_feature = None
        best_value = None

        for feature in self.features:
            X_df = df.dropna().sort_values(feature)

            xmeans = self.calculate_moving_average(X_df[feature].unique(), 2)

            for value in xmeans:
                left_counts = Counter(X_df[X_df[feature] < value]['Y'])
                right_counts = Counter(X_df[X_df[feature] >= value]['Y'])

                y0_left, y1_left, y0_right, y1_right = left_counts.get(0, 0), left_counts.get(1, 0), right_counts.get(0, 0), right_counts.get(1, 0)

                gini_left = self.calculate_GINI_impurity(y0_left, y1_left)
                gini_right = self.calculate_GINI_impurity(y0_right, y1_right)

                left_counter = y0_left + y1_left
                right_counter = y0_right + y1_right

                weight_left = left_counter / (left_counter + right_counter)
                weight_right = right_counter / (left_counter + right_counter)

                weight_GINI = weight_left * gini_left + weight_right * gini_right

                GINI_gain = GINI_base - weight_GINI

                if GINI_gain > max_gain:
                    best_feature = feature
                    best_value = value 

                    max_gain = GINI_gain

        return (best_feature, best_value)

    def build_tree(self):
        df = self.X.copy()
        df['Y'] = self.Y

        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):

            best_feature, best_value = self.generate_best_split()

            if best_feature is not None:
                self.best_feature = best_feature
                self.best_value = best_value

                left_df, right_df = df[df[best_feature]<=best_value].copy(), df[df[best_feature]>best_value].copy()

                left = Decision_Tree (
                    left_df[self.features], 
                    left_df['Y'].values.tolist(), 
                    depth=self.depth + 1, 
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split
                )

                self.left = left 
                self.left.build_tree()

                right = Decision_Tree (
                    right_df[self.features], 
                    right_df['Y'].values.tolist(), 
                    depth = self.depth + 1, 
                    max_depth = self.max_depth, 
                    min_samples_split = self.min_samples_split
                )

                self.right = right
                self.right.build_tree()

    def predict(self, X:pd.DataFrame):
        predictions = []

        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})
        
            predictions.append(self.predict_class(values))
        
        return predictions

    def predict_class(self, values: dict) -> int:
        cur_node = self
        while cur_node.depth < cur_node.max_depth:
            best_feature = cur_node.best_feature
            best_value = cur_node.best_value

            if cur_node.n < cur_node.min_samples_split:
                break 

            if values.get(best_feature) is None:
                break

            if best_value is None:
                break

            if (values.get(best_feature) < best_value):
                if self.left is not None:
                    cur_node = cur_node.left
            else:
                if self.right is not None:
                    cur_node = cur_node.right
            
        return cur_node.predictedClass
