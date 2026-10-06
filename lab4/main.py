# Cross validation Balance Accuracy = 93.30 %
import itertools
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score

SEED = 42
K_FOLDS = 5
TRAIN_DATA_PATH = "dry_bean_train.csv"
TEST_DATA_PATH = "dry_bean_test.csv"
LABEL_COL = "Class"


# ----------------------------------------------------------------------------
# Cross validation (hand-written, stratified so rare classes like BOMBAY appear
# in every fold)
# ----------------------------------------------------------------------------
def stratified_kfold_indices(y, k, seed):
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for i, sample_idx in enumerate(idx):
            folds[i % k].append(sample_idx)
    return [np.array(sorted(f)) for f in folds]


# ----------------------------------------------------------------------------
# Forest (hand-written): bootstrap samples + random feature subset per tree,
# simple majority voting
# ----------------------------------------------------------------------------
class MyForest:
    def __init__(self, n_trees=100, n_features=None, max_depth=None,
                 min_samples_leaf=1, criterion="gini", seed=0):
        self.n_trees = n_trees
        self.n_features = n_features      # features given to each tree
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.seed = seed

    def fit(self, X, y):
        rng = np.random.RandomState(self.seed)
        n, d = X.shape
        self.n_classes_ = int(y.max()) + 1
        nf = self.n_features or d
        self.trees_ = []
        for _ in range(self.n_trees):
            rows = rng.randint(0, n, size=n)                   # bootstrap sample
            cols = np.sort(rng.choice(d, size=nf, replace=False))  # feature subset
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                random_state=rng.randint(1 << 30),
            )
            tree.fit(X[rows][:, cols], y[rows])
            self.trees_.append((tree, cols))
        return self

    def predict(self, X):
        votes = np.zeros((X.shape[0], self.n_classes_), dtype=int)
        for tree, cols in self.trees_:
            pred = tree.predict(X[:, cols])
            votes[np.arange(X.shape[0]), pred] += 1
        return votes.argmax(axis=1)                            # majority vote


def cross_validate(params, X, y, folds):
    scores = []
    for i, val_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != i])
        model = MyForest(seed=SEED + i, **params).fit(X[train_idx], y[train_idx])
        scores.append(balanced_accuracy_score(y[val_idx], model.predict(X[val_idx])))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    trainData = pd.read_csv(TRAIN_DATA_PATH)
    testData = pd.read_csv(TEST_DATA_PATH)
    column_names = [c for c in trainData.columns if c != LABEL_COL]

    column_values = trainData[column_names].values
    bean_types = np.array(sorted(trainData[LABEL_COL].unique()))
    y = np.searchsorted(bean_types, trainData[LABEL_COL].values)
    X_test = testData[column_names].values

    folds = stratified_kfold_indices(y, K_FOLDS, SEED)

    # Baseline: a single tree
    base = []
    for i, val_idx in enumerate(folds):
        tr = np.concatenate([f for j, f in enumerate(folds) if j != i])
        t = DecisionTreeClassifier(random_state=SEED).fit(column_values[tr], y[tr])
        base.append(balanced_accuracy_score(y[val_idx], t.predict(column_values[val_idx])))
    print(f"Single tree  CV balanced accuracy: {np.mean(base)*100:.2f}%")

    # Grid search with CV
    grid = {
        "n_trees": [50],
        "n_features": [8, 10, 12],
        "max_depth": [None],
        "min_samples_leaf": [3, 5],
    }
    best_score, best_params = -1, None
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        mean, std = cross_validate(params, column_values, y, folds)
        print(f"{params}  ->  {mean*100:.2f}% (+/- {std*100:.2f})")
        if mean > best_score:
            best_score, best_params = mean, params

    print(f"\nBest params: {best_params}")
    print(f"Cross validation Balance Accuracy = {best_score*100:.2f} %")

    # Final forest on all training data (more trees for stability)
    final_params = dict(best_params, n_trees=300)
    final = MyForest(seed=SEED, **final_params).fit(column_values, y)
    out = testData.copy()
    out["Target"] = bean_types[final.predict(X_test)]
    out.to_csv("forest.csv", index=False)
    print("Saved forest.csv", out.shape)


if __name__ == "__main__":
    main()