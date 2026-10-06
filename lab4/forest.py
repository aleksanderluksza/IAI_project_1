#Cross validation Balance Accuracy = 93.31 %
#To run this code, numpy v.2.5.3, pandas v.3.0.6, scikit-learn v.1.9.1 have to be installed.
#After installation, run python forest.py
import itertools
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score

def stratified_folding(values, k, seed): #folding for cross validation, stratified for reliable cross validation, as amounts of different class names are not equal
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(values):
        idx = np.where(values == c)[0]
        rng.shuffle(idx)
        for i, sample_idx in enumerate(idx):
            folds[i % k].append(sample_idx)
    return [np.array(sorted(f)) for f in folds]

class Forest:
    def __init__(self, n_trees=100, n_features=None, max_depth=None,
                 min_samples_leaf=1, criterion="gini", seed=0):
        self.n_trees = n_trees
        self.n_features = n_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.seed = seed

    def tree_training(self, column_values, label_column_values): #tree training for better accuracy
        rng = np.random.RandomState(self.seed)
        n_samples, n_input_features = column_values.shape
        self.n_classes_ = int(label_column_values.max()) + 1
        n_selected_features = self.n_features or n_input_features
        self.trees_ = []
        for _ in range(self.n_trees):
            sample_indices = rng.randint(0, n_samples, size=n_samples)
            feature_indices = np.sort(
                rng.choice(n_input_features, size=n_selected_features, replace=False)
            )
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                random_state=rng.randint(1 << 30),
            )
            tree.fit(
                column_values[sample_indices][:, feature_indices],
                label_column_values[sample_indices],
            )
            self.trees_.append((tree, feature_indices))
        return self

    def predict(self, column_values): #simple voting method for samples classification
        votes = np.zeros((column_values.shape[0], self.n_classes_), dtype=int)
        for tree, feature_indices in self.trees_:
            predicted_classes = tree.predict(column_values[:, feature_indices])
            votes[np.arange(column_values.shape[0]), predicted_classes] += 1
        return votes.argmax(axis=1)                            

def cross_validate(seed, params, column_values, label_column_values, folds): #our forest's performance is checked on values, that this forest has never seen
    scores = []
    for fold_index, validation_indices in enumerate(folds):
        training_indices = np.concatenate(
            [fold for index, fold in enumerate(folds) if index != fold_index]
        )
        model = Forest(seed=seed + fold_index, **params).tree_training(
            column_values[training_indices], label_column_values[training_indices]
        )
        scores.append(
            balanced_accuracy_score(
                label_column_values[validation_indices],
                model.predict(column_values[validation_indices]),
            )
        )
    return float(np.mean(scores)), float(np.std(scores))


def main():
    trainData = pd.read_csv("dry_bean_train.csv")
    testData = pd.read_csv("dry_bean_test.csv")
    label_col = "Class"
    trainData_column_names = [c for c in trainData.columns if c != label_col]

    trainData_column_values = trainData[trainData_column_names].values
    bean_types = np.array(sorted(trainData[label_col].unique()))
    bean_types_values = np.searchsorted(bean_types, trainData[label_col].values)
    testData_column_values = testData[trainData_column_names].values
    seed = bean_types.size #setting a "random" value for the seed

    folds = stratified_folding(bean_types_values, 5, seed)

    #basic case - one tree
    tree = []
    for i, val_idx in enumerate(folds):
        tr = np.concatenate([f for j, f in enumerate(folds) if j != i])
        t = DecisionTreeClassifier(random_state = seed).fit(trainData_column_values[tr], bean_types_values[tr])
        tree.append(balanced_accuracy_score(bean_types_values[val_idx], t.predict(trainData_column_values[val_idx])))
    print(f"Single tree  CV balanced accuracy: {np.mean(tree)*100:.2f}%")

    #grid values that may be change in regard to how powerful the device is
    grid = {
        "n_trees": [50],
        "n_features": [8, 10, 12],
        "max_depth": [None],
        "min_samples_leaf": [3, 5],
    }
    #here the different grid values are cross validated to find the best parameters for the final forest
    best_score = -1
    best_params = None
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        mean, std = cross_validate(seed, params, trainData_column_values, bean_types_values, folds)
        print(f"{params}  ->  {mean*100:.2f}% (+/- {std*100:.2f})")
        if mean > best_score:
            best_score, best_params = mean, params

    print(f"\nBest params: {best_params}")
    print(f"Cross validation Balance Accuracy = {best_score*100:.2f} %")

    #final forest is made here to be saved into a csv file
    final_params = dict(best_params, n_trees=300)
    final = Forest(seed=seed, **final_params).tree_training(trainData_column_values, bean_types_values)
    out = testData.copy()
    out["Target"] = bean_types[final.predict(testData_column_values)] # saving the type for each bean
    print("Saving forest.csv...")
    out.to_csv("forest.csv", index=False)
    print("Saved forest.csv", out.shape)


if __name__ == "__main__":
    main()