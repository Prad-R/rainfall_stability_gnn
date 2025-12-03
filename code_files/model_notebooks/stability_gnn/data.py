import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import torch
from torch_geometric.data import Data

## This function is to load a .csv file into a pandas dataframe of desired format
def load_dataset(dataset_path, log_transform=False, bias=1e-2, add_loc=True):
    print("...Loading Dataset...")
    raw_dataset = pd.read_csv(dataset_path)

    ## To add LOC_ID column
    if add_loc:
        loc_id = pd.Series([i for i in range(raw_dataset.shape[0])], name='LOC_ID')
        raw_dataset = pd.concat([loc_id.reset_index(drop=True), raw_dataset.reset_index(drop=True)], axis=1)

    ## If the data is to be bias + log_transformed
    if (log_transform == True):
        meta_columns = ['LOC_ID', 'LATITUDE', 'LONGITUDE', 'VALID_POINTS']
        dataset = raw_dataset.copy()
        dataset.loc[:, ~dataset.columns.isin(meta_columns)] = np.log1p(raw_dataset.loc[:, ~dataset.columns.isin(meta_columns)] + bias)
    else:
        dataset = raw_dataset

    return dataset

def load_dataset_yeo_johnson(dataset_path, transform=None, bias=1e-2):
    """
    Loads a CSV file into a pandas DataFrame and applies a specified transformation.

    Args:
        dataset_path (str): The path to the CSV file.
        transform (str, optional): The type of transformation to apply. 
                                   Can be 'log', 'yeo-johnson', or None.
        bias (float, optional): The constant to add to the data before log transformation.

    Returns:
        pandas.DataFrame: The loaded and optionally transformed DataFrame.
    """
    print("...Loading Dataset...")
    raw_dataset = pd.read_csv(dataset_path)

    ## To add LOC_ID column
    loc_id = pd.Series([i for i in range(raw_dataset.shape[0])], name='LOC_ID')
    dataset = pd.concat([loc_id.reset_index(drop=True), raw_dataset.reset_index(drop=True)], axis=1)

    ## Identify columns to transform, excluding metadata
    meta_columns = ['LOC_ID', 'LATITUDE', 'LONGITUDE', 'VALID_POINTS']
    cols_to_transform = dataset.columns.difference(meta_columns)

    if transform == 'log':
        print("...Applying Bias + Log Transformation...")
        # The np.log1p function computes log(1+x), so a simple log transform
        # with a bias is achieved by raw_data + bias.
        dataset.loc[:, cols_to_transform] = np.log1p(dataset.loc[:, cols_to_transform] + bias)
    
    elif transform == 'yeo-johnson':
        print("...Applying Yeo-Johnson Transformation...")
        # Create a dictionary to store lambda values for each column
        lambdas = {}
        for col in cols_to_transform:
            # Apply Yeo-Johnson transformation and get the optimal lambda
            transformed_col, lmbda = stats.yeojohnson(dataset[col])
            dataset[col] = transformed_col
            lambdas[col] = lmbda
        
        print("\nOptimal lambda values for every column found:")
        # for col, lmbda in lambdas.items():
        #     print(f" - {col}: {lmbda:.4f}")
    
    elif transform is not None:
        print(f"Warning: Unknown transformation '{transform}'. No transformation applied.")

    return dataset

## This function is to obtain a state's dataset from the full dataset
def get_state_dataset(dataframe, lat_extent, lon_extent):
    lat_min = lat_extent[0]
    lat_max = lat_extent[1]
    lon_min = lon_extent[0]
    lon_max = lon_extent[1]

    lat_predicate = dataframe['LATITUDE'].between(lat_min, lat_max)
    lon_predicate = dataframe['LONGITUDE'].between(lon_min, lon_max)

    df_from_full = dataframe[lat_predicate & lon_predicate]

    return df_from_full

## This function is to generate year-wise daily rainfall for every location in a dictionary format
def make_year_wise_dict(dataframe, years=list(range(2015, 2023))):
    print("...Making year wise dictionary...")
    year_wise_dict = dict()

    year_columns = {
        year: [col for col in dataframe.columns if str(year) in str(col)]
        for year in years
    }
    
    col_positions = {col: i for i, col in enumerate(dataframe.columns)}

    def is_leap_year(year):
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    ## itertuples is faster than iterrows, but it cannot alter the accessed data
    for row in dataframe.itertuples(index=False):
        loc_id = row.LOC_ID

        year_wise_dict[loc_id] = {}

        for year in years:
            cols = year_columns[year]

            if is_leap_year(year):
                cols = [col for col in cols if not "02-29" in str(col)]

            rainfall_vector = [row[col_positions[col]] for col in cols]
            year_wise_dict[loc_id][year] = np.array(rainfall_vector)

    return year_wise_dict

## This function to compute cosine similarity matrix for every location
def make_cos_sim_dict(year_wise_dict, ids, years=list(range(2015, 2023))):
    print("...Making cosine similarity dictionary...")
    cos_sim_dict = dict()

    for id in ids:

        vectors = [year_wise_dict[id][year] for year in years]
        tensor_data = torch.tensor(vectors, dtype=torch.float32)

        norm_data = torch.nn.functional.normalize(tensor_data, p=2, dim=1)
        sim_matrix = torch.mm(norm_data, norm_data.T)
        
        cos_sim_dict[id] = sim_matrix

    return cos_sim_dict

## This function is to make a mean node for all locations
def make_mean_node(year_wise_dict, ids, years=list(range(2015, 2023))):
    print("...Making mean nodes...")    
    for id in ids:
        sum = np.zeros(np.shape(year_wise_dict[id][2022]))

        for year in years:
            sum += year_wise_dict[id][year]

        year_wise_dict[id]['MEAN'] = sum / len(years)

    return year_wise_dict

## This function is to make a dictionary of percentile based threshold values for the similarity score
def make_threshold_dict(cos_sim_dict, train_ids):
    print("...Making threshold dictionary...")
    all_cos_sims = []

    for train_id in train_ids:
        cos_sim = cos_sim_dict[train_id]
        cos_sim_no_diag = cos_sim.clone()
        cos_sim_no_diag.fill_diagonal_(float('nan'))

        flat_sims = cos_sim_no_diag[~torch.isnan(cos_sim_no_diag)]
        all_cos_sims.append(flat_sims)

    ## This is the tensor that has all the cosine-similarity scores (exclusing self-similarites) for all TRGCODEs and YEARs
    all_cos_sims = torch.cat(all_cos_sims)

    global_percentile_dict = {}

    for i in range(5, 100, 5):
        global_percentile_dict[i] = torch.quantile(all_cos_sims, i/100).item()

    return global_percentile_dict

## This function is to generate labels for the locations
def generate_labels(cos_sim_dict, threshold, ids, years):
    print("...Generating Labels...")
    stable_rain = set()
    unstable_rain = set()

    for id in ids: ## Labels for all graphs are generated using threshold computed on training set
        sim_matrix = cos_sim_dict[id]
        loc_mean = (sim_matrix.sum() - sim_matrix.trace()) / (len(years) ** 2 - len(years))

        if (loc_mean >= threshold):
            stable_rain.add(id)
        else:
            unstable_rain.add(id)

    return stable_rain, unstable_rain

## This function is to make the actual graphs
def make_graphs(year_wise_dict, cos_sim_dict, ids, threshold, years=list(range(2015, 2023))):
    print("...Making Graphs...")
    graphs = dict()

    stable_rain, unstable_rain = generate_labels(
        cos_sim_dict=cos_sim_dict,
        threshold=threshold,
        ids=ids,
        years=years
    )

    for id in ids:

        edge_list = []

        for i in range(len(years)):
            
            ## 0 is considered as the average node, and each year node has the index (year - 2014)
            edge_list.append([[i + 1], [0]])

            for j in range(len(years)):
                if i < j:
                    score = cos_sim_dict[id][i][j]

                    if score >= threshold:
                        ## Each year node has the index (year - 2014)
                        edge_list.append([[i + 1], [j + 1]])

        edge_list = np.hstack(edge_list)
        edge_list = torch.tensor(edge_list, dtype=torch.long)
        
        node_features = [year_wise_dict[id]['MEAN']]

        for year in years:
            node_features.append(year_wise_dict[id][year])

        node_features = np.vstack(node_features)
        node_features = torch.tensor(node_features, dtype=torch.float)

        if (id in stable_rain):
            label = torch.tensor([1])
        else: 
            label = torch.tensor([0])

        graph = Data(x=node_features, edge_index=edge_list, y=label)

        graphs[id] = graph

    return graphs

## This function is to generate graphs given a dataframe
def generate_graphs(dataframe, train=False, threshold_percentile=75, threshold_value=None):
    print(f"...Generating graphs: training_mode: {train}, threshold_percentile: {threshold_percentile}...")

    ## The unique IDs for each location
    loc_ids = dataframe['LOC_ID'].values

    ## The various years in the dataset
    years = pd.to_datetime(dataframe.columns[4:-2]).year
    years = sorted(years.unique())
    print(f"The years under consideration are: {years}")

    ## Obtain year-wise daily rainfall for every location in a dictionary format
    year_wise_dict = make_year_wise_dict(
        dataframe=dataframe,
        years=years
    )

    ## Compute cosine similarity matrix for every location
    cos_sim_dict = make_cos_sim_dict(
        year_wise_dict=year_wise_dict,
        ids=loc_ids,
        years=years
    )

    ## To make a mean node for all locations
    year_wise_dict = make_mean_node(
        year_wise_dict=year_wise_dict,
        ids=loc_ids,
        years=years
    )

    if (train == True): ## For generating thresholds during training
        ## Splitting the data into training and validating sets
        train_loc_ids, validate_loc_ids = train_test_split(
            loc_ids,
            test_size=0.2,
            random_state=3707
        )

        ## Generating the various thresholds to select for training
        global_percentile_dict = make_threshold_dict(
            cos_sim_dict=cos_sim_dict,
            train_ids=train_loc_ids
        )
        
        ## Defining the threshold
        threshold_value = global_percentile_dict[threshold_percentile]

        ## Threshold to return
        return_threshold = threshold_value

    ## Making the actual graphs
    graphs = make_graphs(
        year_wise_dict=year_wise_dict,
        cos_sim_dict=cos_sim_dict,
        ids=loc_ids,
        years=years,
        threshold=threshold_value
    )

    if (train  == True):
        ## Converting the ids to sets for faster lookups
        train_loc_ids = set(train_loc_ids)
        validate_loc_ids = set(validate_loc_ids)

        ## Initializing the graph dictionaries
        train_graphs = dict()
        validation_graphs = dict()

        ## Assignign the graphs to the correct dictionary
        for loc_id, graph in graphs.items():
            if loc_id in train_loc_ids:
                train_graphs[loc_id] = graph
            else:
                validation_graphs[loc_id] = graph

        return train_graphs, validation_graphs, graphs, return_threshold
    else:
        return graphs