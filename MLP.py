# imports
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from tqdm import tqdm
from itertools import product # used for hyperparameter grid search, unused if not doing hyperparameter tuning
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import warnings

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, index):
        features = self.features.iloc[index].to_numpy()
        label = self.labels.iloc[index]
        return features, label

    def __len__(self):
        return len(self.features)
    
def split_data(data, batch_size, task=1):
    
    dataloaders = []
    global class_weights

    if task == 0 or task == 1: # Known subjects and items
        features, labels = data
        dataset = CustomDataset(features, labels)
        n = len(dataset)
        if task == 0:
            k = 10 # k-fold cross-validation
        elif task == 1:
            k = n # leave-one-out cross-validation
        fold_size = n // k
        folds = []
        for i in range(k):
            start = i * fold_size
            end = (i + 1) * fold_size if i < k - 1 else n
            folds.append(torch.utils.data.Subset(dataset, range(start, end)))

        for i in range(k):
            # splits for cross-validation, validation set = test set (since we're doing k-fold, we won't use a separate test set)
            validation_dataset = folds[i]
            t = i + 1 if i < k - 1 else 0
            test_dataset = folds[t]
            train_folds = [folds[j] for j in range(k) if j != i]# and j != t]
            train_dataset = torch.utils.data.ConcatDataset(train_folds)

            # class weights for weighted cross-entropy loss (to handle class imbalance)
            y = torch.tensor([label for _, label in train_dataset], dtype=torch.long)
            
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y.numpy())
            class_weights = torch.tensor(class_weights, dtype=torch.float)

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            dataloaders.append((train_dataloader, validation_dataloader, test_dataloader))
            #dataloaders.append((train_dataloader, validation_dataloader))

        return dataloaders
    
    elif task == 2: # Held-out subjects, known items
        test_items_count = 0
        for i, subject in enumerate(data.groups.keys()):
            test = data.get_group(subject)
            test_items_count += len(test)
            train_eval = pd.concat([data.get_group(i) for i in data.groups.keys() if i != subject])
            shuffled = train_eval.sample(frac = 1, random_state=seed) # shuffle the data -> wrecked.
            
            # splitting data into features and labels for dataset creation
            test_labels = test["condition"].copy()
            test_features = test.copy().drop(["condition", "sentenceCondition", "RECORDING_SESSION_LABEL", "trial"], axis=1)
            test_dataset = CustomDataset(test_features, test_labels)
            
            train_eval_labels = shuffled["condition"].copy()
            train_eval_features = shuffled.copy().drop(["condition", "sentenceCondition", "RECORDING_SESSION_LABEL", "trial"], axis=1)
            train_eval_dataset = CustomDataset(train_eval_features, train_eval_labels)


            train_eval_split = 0.9
            train_size = int(train_eval_split * len(train_eval_dataset))
            validation_size = len(train_eval_dataset) - train_size
            train_dataset, validation_dataset = torch.utils.data.random_split(train_eval_dataset, [train_size, validation_size])

            y = torch.tensor([label for _, label in train_dataset], dtype=torch.long)

            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y.numpy())
            class_weights = torch.tensor(class_weights, dtype=torch.float)

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            dataloaders.append((train_dataloader, validation_dataloader, test_dataloader))
        return dataloaders
    
    elif task == 3: # Held-out items, known subjects
        test_items_count = 0
        for i, item in enumerate(data.groups.keys()):
            test = data.get_group(item)
            test_items_count += len(test)
            train_eval = pd.concat([data.get_group(i) for i in data.groups.keys() if i != item])
            shuffled = train_eval.sample(frac = 1, random_state=seed) # shuffle the data -> wrecked.
            
            # splitting data into features and labels for dataset creation
            test_labels = test["condition"].copy()
            test_features = test.copy().drop(["condition", "sentenceCondition", "RECORDING_SESSION_LABEL", "trial"], axis=1)
            test_dataset = CustomDataset(test_features, test_labels)
            
            train_eval_labels = shuffled["condition"].copy()
            train_eval_features = shuffled.copy().drop(["condition", "sentenceCondition", "RECORDING_SESSION_LABEL", "trial"], axis=1)
            train_eval_dataset = CustomDataset(train_eval_features, train_eval_labels)


            train_eval_split = 0.9
            train_size = int(train_eval_split * len(train_eval_dataset))
            validation_size = len(train_eval_dataset) - train_size
            train_dataset, validation_dataset = torch.utils.data.random_split(train_eval_dataset, [train_size, validation_size])

            y = torch.tensor([label for _, label in train_dataset], dtype=torch.long)

            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y.numpy())
            class_weights = torch.tensor(class_weights, dtype=torch.float)

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            dataloaders.append((train_dataloader, validation_dataloader, test_dataloader))
        return dataloaders
    else:
        raise ValueError("Task argument must be either 1, 2, or 3")

    
def preprocess_and_split_data(data, batch_size=32, task=1):
    
# all tasks
    
    data_copy = data.loc[data['is_critical'] == 1].copy()
    dropped = data_copy.drop(['composite', 'LF', 'HF', "IA_ID", "item", "list", "IA_LABEL", "wordlength", "is_critical", 
                'is_spill1', 'is_spill2', 'is_spill3', 'filler', 'function_word', 'other_filler'], axis=1)
    #print("Original dataset size: ", len(data_copy))

    # normalizing input features beforehand, increased performance vs adding batchnorm layer to model
    temp = dropped[['fixation_duration',
        'duration_firstpass', 'duration_firstfixation', 'fix_count',
        'avg_pupil', 'IA_REGRESSION_IN_COUNT', 'IA_REGRESSION_OUT_COUNT',
        'saccade_length', 'saccade_duration', 'go_past_time']]
    temp = (temp - temp.mean()) / temp.std()
    dropped[['fixation_duration',
        'duration_firstpass', 'duration_firstfixation', 'fix_count',
        'avg_pupil', 'IA_REGRESSION_IN_COUNT', 'IA_REGRESSION_OUT_COUNT',
        'saccade_length', 'saccade_duration', 'go_past_time']] = temp
    normalized = dropped
    # mapping condition and sentenceCondition to 0 and 1 for critical word classification
    normalized[["condition", "sentenceCondition"]] = normalized[["condition", "sentenceCondition"]].apply(lambda x: x.replace("none", "0"))
    normalized[["condition", "sentenceCondition"]] = normalized[["condition", "sentenceCondition"]].apply(lambda x: x.replace("control", "0"))
    normalized[["condition", "sentenceCondition"]] = normalized[["condition", "sentenceCondition"]].apply(lambda x: x.replace("pseudo", "1"))
    normalized[["condition", "sentenceCondition"]] = normalized[["condition", "sentenceCondition"]].apply(lambda x: x.replace("filler", "0"))
    normalized[["condition", "sentenceCondition"]] = normalized[["condition", "sentenceCondition"]].astype(int)
    mapped = normalized

# task 1 specific steps
    if task == 0 or task == 1: # Known subjects and items
        shuffled = mapped.sample(frac = 1, random_state=seed) # shuffle the data -> wrecked.
        # splitting data into features and labels for dataset creation
        labels = shuffled["condition"].copy()
        features = shuffled.copy().drop(["condition", "sentenceCondition", "RECORDING_SESSION_LABEL", "trial"], axis=1)
        #print("Preprocessed dataset size: ", len(features))
        data = (features, labels)
        return split_data(data, batch_size, task)
    
    elif task == 2: # Held-out subjects, known items
        subjects = mapped.groupby('RECORDING_SESSION_LABEL')
        return split_data(subjects, batch_size, task)
    
    elif task == 3: # Held-out items, known subjects
        items = mapped.groupby('trial')
        return split_data(items, batch_size, task)
    
    else:
        raise ValueError("Task argument must be either 1, 2, or 3")
    
def train_test(model, dataloader, optimizer, training="train"):
   
    loss_function = torch.nn.BCEWithLogitsLoss()#weight=class_weights.to(device))

    if training == "train":
        model.train()
    elif training == "validation":
        model.eval()
    elif training == "test":
        model.eval()
    else:
        raise ValueError("training argument must be either 'train', 'validation' or 'test'")
        
    total = 0
    correct = 0
    cumulative_loss = 0
    prediction_list = []
    label_list = []
    sigmoid = torch.nn.Sigmoid()
    for sample in dataloader:
   
        data, targets = sample[0].float().to(device), sample[1].type(torch.LongTensor).to(device)
        output = model(data)
        loss_value = loss_function(output, targets.unsqueeze(1).float())
        cumulative_loss += loss_value.item()

        if training == "train":
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
        
        predictions = [round(x) for x in sigmoid(output).to('cpu').detach().squeeze(1).numpy().tolist()]#.argmax(axis=1)
        target_labels = targets.to('cpu').detach().numpy()
        total += len(predictions)
        correct += accuracy_score(target_labels, predictions, normalize=False)
        prediction_list.extend(predictions)
        label_list.extend(target_labels)
    if training == "test":
        return label_list, prediction_list
    f1 = f1_score(label_list, prediction_list)
    accuracy = accuracy_score(label_list, prediction_list)
    confusion = confusion_matrix(label_list, prediction_list)

    return cumulative_loss, accuracy, f1, confusion

class TuneableModel(torch.nn.Module):
    def __init__(self, input_size, layer_size, dropout_rate, n_layers):
        super(TuneableModel, self).__init__()
        self.n_layers = n_layers
        self.input_layer = torch.nn.LazyLinear(layer_size)
        self.linear2 = torch.nn.Linear(layer_size, layer_size)
        self.linear3 = torch.nn.Linear(layer_size, layer_size)
        self.linear4 = torch.nn.Linear(layer_size, layer_size)
        self.linear5 = torch.nn.Linear(layer_size, layer_size)
        self.linear6 = torch.nn.Linear(layer_size, layer_size)
        self.linear7 = torch.nn.Linear(layer_size, layer_size)
        self.linear8 = torch.nn.Linear(layer_size, layer_size)
        self.linear9 = torch.nn.Linear(layer_size, layer_size)
        self.linear10 = torch.nn.Linear(layer_size, layer_size)
        self.output_layer = torch.nn.Linear(layer_size, 1)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = torch.nn.LeakyReLU()
        self.batchnorm = torch.nn.BatchNorm1d(layer_size)

    def forward(self, x):
        x = self.input_layer(x)
        #x = self.batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.n_layers > 1:
            x = self.linear2(x)
            x = self.activation(x)
            x = self.dropout(x)
            if self.n_layers > 2:
                x = self.linear3(x)
                x = self.activation(x)
                x = self.dropout(x)
                if self.n_layers > 3:
                    x = self.linear4(x)
                    x = self.activation(x)
                    x = self.dropout(x)
                    if self.n_layers > 4:
                        x = self.linear5(x)
                        x = self.activation(x)
                        x = self.dropout(x)
                        if self.n_layers > 5:
                            x = self.linear6(x)
                            x = self.activation(x)
                            x = self.dropout(x)
                            if self.n_layers > 6:
                                x = self.linear7(x)
                                x = self.activation(x)
                                x = self.dropout(x)
                                if self.n_layers > 7:
                                    x = self.linear8(x)
                                    x = self.activation(x)
                                    x = self.dropout(x)
                                    if self.n_layers > 8:
                                        x = self.linear9(x)
                                        x = self.activation(x)
                                        x = self.dropout(x)
                                        if self.n_layers > 9:
                                            x = self.linear10(x)
                                            x = self.activation(x)
                                            x = self.dropout(x)
        x = self.output_layer(x)
        #x = self.activation(x)
        return x

# Training sample
def evaluate(data, parameters, task):
    assert task in [0, 1, 2, 3], "Task argument must be either 1, 2 or 3"
    
    dropout, hidden_size, learning_rate, batch_size, n_hidden, beta_1, beta_2 = parameters

    max_epochs = 1000

    dataloaders = preprocess_and_split_data(data, batch_size, task)

    input_size = 10 # number of features :( -> this is hardcoded for now, try to get it from the dataset
    best_epochs = []
    predictions = []
    labels = []
    torch.manual_seed(seed)
    model = TuneableModel(input_size, hidden_size, dropout, n_hidden)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta_1, beta_2), weight_decay=1e-2)


    
    for i, dataloader in tqdm(enumerate(dataloaders)):
        max_patience = 10 if i < 35 else 2
        last_loss = 1000000
        best_epoch = 0
        PATH = f"model_{i}.pt"
        train_dataloader, validation_dataloader, test_dataloader = dataloader[0], dataloader[1], dataloader[2]
        for epoch in range(max_epochs):
            # training
            train_loss, train_accuracy, train_f1, train_confusion = train_test(model, train_dataloader, optimizer, training="train")
            train_loss, train_accuracy, train_f1 = round(train_loss, 2), round(train_accuracy, 4), round(train_f1, 2)
            # validation at end of epoch
            validation_loss, validation_accuracy, validation_f1, validation_confusion = train_test(model, validation_dataloader, optimizer, training="validation")
            validation_loss, validation_accuracy, validation_f1 = round(validation_loss, 2), round(validation_accuracy, 4), round(validation_f1, 2)
            if validation_loss < last_loss:
                last_loss = validation_loss
                best_epoch = epoch
                current_patience = 0
            else:
                if current_patience == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': last_loss,
                        }, PATH)
                current_patience += 1
            if current_patience == max_patience:
                break   
            # if epoch % 100 == 0 and epoch != 0:
            #     print(f"Epoch {epoch}: Train loss: {train_loss}, Train accuracy: {train_accuracy}, Train f1: {train_f1}")
            #     print(f"Epoch {epoch}: Validation loss: {validation_loss}, Validation accuracy: {validation_accuracy}, Validation f1: {validation_f1}")

        # Testing once patience is reached
        torch.manual_seed(seed)
        model = TuneableModel(input_size, hidden_size, dropout, n_hidden)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta_1, beta_2), weight_decay=1e-2)
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        prediction_list, label_list = train_test(model, test_dataloader, optimizer, training="test")
        predictions.extend(prediction_list)
        labels.extend(label_list)
        best_epochs.append(best_epoch)
    #print("Average training epochs for best model:", round(np.mean(best_epochs), 1))
    #print("Best epochs:\n\t", best_epochs)
    return accuracy_score(labels, predictions), f1_score(labels, predictions), confusion_matrix(labels, predictions)
    # print(f"Average accuracy: {round(np.mean(accuracies), 2)}%")
    # print(f"Average f1: {round(np.mean(f1s), 2)}")


if __name__ == "__main__":

    warnings.filterwarnings('ignore', category=UserWarning) # because of LazyLinear layer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    seed = 42 # for reproducibility
    torch.manual_seed(seed)

    data = pd.read_csv('data.csv', delimiter=';')

    parameters = (0.0, 500, 0.001, 16, 6, 0.999, 0.999)

    tasks = ["Known subjects and items, 10-fold CV", "Known subjects and items, LOOCV", "Held-out subjects, known items", "Held-out items, known subjects"]

    for task in [0, 1, 2, 3]:
        print(f"Task {tasks[task]}:")
        accuracy, f1, confusion = evaluate(data, parameters, task)
        print(f"Acc: {round(accuracy*100,2)}%\nF1: {round(f1,4)}")
        print("Confusion:\n", confusion)


