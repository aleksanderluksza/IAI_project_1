# Cross validation Balance Accuracy = 94.21 %
#To run this code, numpy v.2.5.3, pandas v.3.0.6, scikit-learn v.1.9.1, torch v.2.14.1(+cu132 for cuda support) and tensorboard v.0.29.1(+cu132) have to be installed.
#After installation, run python network.py
#After running network.py, to check logged losses visualizations, enter 'tensorboard --logdir runs' to terminal and enter the linked page hosted on localhost
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #device checkup - to run this code on gpu, torch version with cuda support is needed (requires a gpu with cuda cores)

def stratified_folding(values, k, seed): #folding for cross validation, stratified for reliable cross validation, as amounts of different class names are not equal
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(values):
        idx = np.where(values == c)[0]
        rng.shuffle(idx)
        for i, sample_idx in enumerate(idx):
            folds[i % k].append(sample_idx)
    return [np.array(sorted(f)) for f in folds]

class BeanDataset(Dataset): #wraps features and labels, so DataLoader can make mini-batches from them
    def __init__(self, column_values, label_column_values=None):
        self.x = torch.tensor(column_values, dtype=torch.float32)
        self.y = None if label_column_values is None else torch.tensor(label_column_values, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        if self.y is None:
            return self.x[i]
        return self.x[i], self.y[i]

class Network(nn.Module): #input layer -> hidden layer(s) (Linear + BatchNorm + activation + Dropout) -> output layer
    def __init__(self, n_inputs, n_classes, hidden_sizes=(64,), activation="relu", dropout=0.0):
        super().__init__()
        activations = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}
        layers = []
        size = n_inputs
        for h in hidden_sizes:
            layers += [nn.Linear(size, h), nn.BatchNorm1d(h), activations[activation](), nn.Dropout(dropout)]
            size = h
        layers.append(nn.Linear(size, n_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def cross_entropy(logits, targets): #checking how wrong the network is at giving labels to beans
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -log_probs[torch.arange(len(targets)), targets].sum()

def standardise(train_values, other_values): #standardisation of values for better neural network training
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0) + 1e-8
    return (train_values - mean) / std, (other_values - mean) / std

def predict(model, column_values, batch_size=512): 
    model.eval()
    predictions = []
    with torch.no_grad():
        for x in DataLoader(BeanDataset(column_values), batch_size=batch_size):
            predictions.append(model(x.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)

def network_training(column_values, label_column_values, n_classes, params, seed, validation=None, writer=None, tag="train"): #training of the network, done by passing through training data(epoch), split into mini batches
    torch.manual_seed(seed)
    model = Network(column_values.shape[1], n_classes, params["hidden_sizes"], params["activation"], params["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loader = DataLoader(BeanDataset(column_values, label_column_values), batch_size=params["batch_size"],
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    for epoch in range(params["epochs"]):
        model.train() #switches the model to training behaviour
        total_loss = 0.0 
        for x, y in loader:
            x = x.to(device)
            y = y.to(device) 
            optimizer.zero_grad() #resets gradients
            loss = cross_entropy(model(x), y) / len(y) #calculating the difference between real results and network's results within the batch
            loss.backward() 
            optimizer.step() #updates optimizer
            total_loss += loss.item() * len(y)
        if writer is not None:
            writer.add_scalar(f"{tag}/loss", total_loss / len(loader.dataset), epoch)
            if validation is not None:
                val_x, val_y = validation
                writer.add_scalar(f"{tag}/validation_balanced_accuracy",
                                  balanced_accuracy_score(val_y, predict(model, val_x)), epoch)
    return model

def cross_validate(seed, params, column_values, label_column_values, folds, n_classes, writer=None, tag="cv"): #our network's performance is checked on values, that this network has never seen
    scores = []
    for fold_index, validation_indices in enumerate(folds):
        training_indices = np.concatenate(
            [fold for index, fold in enumerate(folds) if index != fold_index]
        )
        train_x, val_x = standardise(column_values[training_indices], column_values[validation_indices])
        model = network_training(train_x, label_column_values[training_indices], n_classes, params, seed + fold_index,
                                 validation=(val_x, label_column_values[validation_indices]),
                                 writer=writer, tag=f"{tag}/fold{fold_index}")
        scores.append(balanced_accuracy_score(label_column_values[validation_indices], predict(model, val_x)))
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
    n_classes = bean_types.size

    folds = stratified_folding(bean_types_values, 5, seed)
    writer = SummaryWriter("runs")

    base_params = {"hidden_sizes": (64,), "activation": "relu", "dropout": 0.0,
                   "weight_decay": 0.0, "lr": 0.003, "batch_size": 128, "epochs": 40}

    #basic case
    mean, std = cross_validate(seed, base_params, trainData_column_values, bean_types_values, folds, n_classes, writer, "baseline")
    print(f"Baseline CV balanced accuracy: {mean*100:.2f}% (+/- {std*100:.2f})")

    #grid values that may be change in regard to how powerful the device is
    grid = {
        "hidden_sizes": [(64,), (128,), (64, 64)],
        "activation": ["relu", "tanh"],
        "dropout": [0.0, 0.2],
        "weight_decay": [0.0, 1e-4],
    }
    #here the different grid values are cross validated to find the best parameters for the final network
    best_score = -1
    best_params = None
    for values in itertools.product(*grid.values()):
        params = dict(base_params, **dict(zip(grid.keys(), values)))
        mean, std = cross_validate(seed, params, trainData_column_values, bean_types_values, folds, n_classes)
        shown = {k: params[k] for k in grid}
        print(f"{shown}  ->  {mean*100:.2f}% (+/- {std*100:.2f})")
        writer.add_text("grid", f"{shown} -> {mean*100:.2f}%")
        if mean > best_score:
            best_score, best_params = mean, params

    print(f"\nBest params: { {k: best_params[k] for k in grid} }")
    print(f"Cross validation Balance Accuracy = {best_score*100:.2f} %")

    #final network is trained on all training data, saved with state_dict, loaded again and used for the csv file
    train_x, test_x = standardise(trainData_column_values, testData_column_values)
    final = network_training(train_x, bean_types_values, n_classes, best_params, seed, writer=writer, tag="final")
    torch.save(final.state_dict(), "network.pt")
    loaded = Network(train_x.shape[1], n_classes, best_params["hidden_sizes"], best_params["activation"], best_params["dropout"]).to(device)
    loaded.load_state_dict(torch.load("network.pt", map_location=device))
    out = testData.copy()
    out["Target"] = bean_types[predict(loaded, test_x)] # saving the type for each bean
    print("Saving network.csv...")
    out.to_csv("network.csv", index=False)
    print("Saved network.csv", out.shape)
    writer.close()


if __name__ == "__main__":
    main()