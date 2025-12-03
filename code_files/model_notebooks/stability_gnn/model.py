from .data import generate_graphs, get_state_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch_geometric.loader import DataLoader
from .visualization import display_geo_map_optimized, plot_loss_curves, plot_tsne

## This function is to compute class weights given a set of graphs
def find_class_weights(train_graphs):
    print("...Computing Class Weights...")
    ## To get a list of class labels, later used for class balancing
    stratify_labels = [graph.y.item() for graph in train_graphs.values()]
    print(f"The number of zero locations: {len(stratify_labels) - sum(stratify_labels)} out of {len(stratify_labels)} locations")

    ## Computing class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=[0, 1],
        y=stratify_labels
    )

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    return class_weights_tensor

## This function is to perform one forward pass while training
def training_pass(train_loader, model, criterion, optimizer):
    model.train() ## Toggles training mode. Enables dropout, updates running statistics, etc. READ MORE.

    total_loss = 0

    for train_graph in train_loader:
        out = model(train_graph.x.float(), train_graph.edge_index, train_graph.batch) ## A single forward pass
        loss = criterion(out, train_graph.y.long()) ## Compute cross entropy loss
        loss.backward() ## Find the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() ## Update parameters based on gradients
        optimizer.zero_grad() ## Clear gradients
        total_loss += loss.item() * train_graph.num_graphs
    
    return total_loss / len(train_loader.dataset)

## This function is to evaluate the results of a forward pass
def evaluate_pass(data_loader, model):
    model.eval() ## Toggles testing behaviour. READ MORE.

    correct = 0

    for graph in data_loader:
        out = model(graph.x.float(), graph.edge_index, graph.batch)
        prediction = out.argmax(dim=1) ## Use the class with highest probability
        correct += int((prediction == graph.y).sum())
    
    return correct / len(data_loader.dataset)

## This function is to train the model
def train_model(train_dataframe, model, threshold_percentile=25):
    print("...Training Model...")
    
    ## Generating graphs from the training dataframe
    train_graphs, validation_graphs, graphs, threshold_value = generate_graphs(
        dataframe=train_dataframe,
        train=True,
        threshold_percentile=threshold_percentile
    )

    ## Setting up loaders for both sets
    train_loader = DataLoader(
        dataset=list(train_graphs.values()),
        batch_size=128,
        shuffle=True
    )
    validation_loader = DataLoader(
        dataset=list(validation_graphs.values()),
        batch_size=32,
        shuffle=True
    )

    ## Computing class weights
    class_weights_tensor = find_class_weights(train_graphs=train_graphs)

    ## Initializing the training optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )

    ## Defining the evaluation criterion
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor) 

    ## Variables to store training loop variables
    losses = list()
    train_accuracies = list()
    validate_accuracies = list()

    best_test_accuracy = 0
    patience_counter = 0
    patience = 10

    ## Main training loop
    for epoch in range(1, 120):
        loss = training_pass(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer
        )
        losses.append(loss)
        train_accuracy = evaluate_pass(
            data_loader=train_loader,
            model=model
        )
        validate_accuracy = evaluate_pass(
            data_loader=validation_loader,
            model=model
        )
        train_accuracies.append(train_accuracy)
        validate_accuracies.append(validate_accuracy)
        print(f'Epoch: {epoch:03d}, Train Accuracy: {train_accuracy:.4f}, Validate Accuracy: {validate_accuracy:.4f}, Patience Counter: {patience_counter}')

        scheduler.step(validate_accuracy)

        if (validate_accuracy > best_test_accuracy):
            best_test_accuracy = validate_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1

        if patience_counter > patience:
            print("----- Early stopping triggered -----")
            break

    ## Saving the model
    torch.save(model.state_dict(), 'Model_full_india.pth')

    ## Plotting the results of training
    plot_loss_curves(# type: ignore
        losses=losses,
        train_accuracies=train_accuracies,
        validate_accuracies=validate_accuracies,
        savename='../images/loss_curve.pdf'
    )

    ## Plotting the t-SNE plot
    plot_tsne(# type: ignore
        model=model,
        graphs=graphs,
        savename='../images/tsne.pdf'
    )

    return train_graphs, validation_graphs, graphs, threshold_value

## This function is to make predictions
def make_predictions(model, graphs):
    print("...Making predictions...")
    model.eval()

    y_pred = dict()

    for id, graph in graphs.items():

        with torch.no_grad():
            num_nodes = graph.x.size(0)
            batch = torch.zeros(num_nodes, dtype=torch.long)
            out = model(graph.x.float(), graph.edge_index, batch)
            prediction = out.argmax(dim=1).item()

        y_pred[id] = prediction

    return y_pred

## This function is to get the true labels for graphs
def get_true_labels(graphs):
    print("...Getting true labels...")

    y_true = dict()

    for id, graph in graphs.items():
        y_true[id] = graph.y.item()

    return y_true

## This function is to test the model on a subset of the dataset by giving the confusion matrix and the geography plot
def test_and_evaluate(model, dataframe, threshold_value, lat_range=None, lon_range=None, cm_savename=None, geo_map_savename=None):

    if (lat_range == None):
        lat_range = [dataframe['LATITUDE'].min(), dataframe['LATITUDE'].max()]

    if (lon_range == None):
        lon_range = [dataframe['LONGITUDE'].min(), dataframe['LONGITUDE'].max()]

    df_from_full = get_state_dataset(
        dataframe=dataframe,
        lat_extent=[lat_range[0], lat_range[1]],
        lon_extent=[lon_range[0], lon_range[1]]
    )

    df_graphs = generate_graphs(
        dataframe=df_from_full,
        train=False,
        threshold_value=threshold_value
    )

    df_y_pred = make_predictions(
        model=model,
        graphs=df_graphs
    )

    df_y_true = get_true_labels(
        graphs=df_graphs
    )

    df_cm = confusion_matrix(list(df_y_true.values()), list(df_y_pred.values()))
    print(df_cm)

    ConfusionMatrixDisplay(df_cm, display_labels=['Erratic', 'Consistent']).plot()
    plt.grid(False)
    plt.title("Confusion Matrix", fontweight='bold', fontsize=21)
    if cm_savename is not None:
        plt.savefig(cm_savename)

    print('\nClassification Report:')
    print(classification_report(list(df_y_true.values()), list(df_y_pred.values()), target_names=['Erratic', 'Consistent']))

    display_geo_map_optimized(
        dataframe=df_from_full,
        y_pred=df_y_pred,
        shapefile_path='/home/prad/code/precipitation_gauge_gnn/shapefiles/india/india_updated_state_boundary.shp',
        ids=df_from_full['LOC_ID'].values,
        transform=False,
        figsize=(9, 9),
        savename=geo_map_savename
    )

    return df_y_pred, df_y_true