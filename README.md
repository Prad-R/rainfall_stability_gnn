# README

This file contains some information about the various directories and files, what they're used for, and their version order.

> [!NOTE]
> Any data files (*.csv) will not be available on the repository.

## Repository Structure

- [code_files](code_files/) conatins all the relevant programming files.

    - [model_notebooks](code_files/model_notebooks/) contains all the `.ipynb` notebooks that contain the code for the various models.

        - [precipitation_gauge_gnn.ipynb](code_files/model_notebooks/precipitation_gauge_gnn.ipynb) is the first iteration which was trained only on Karnataka precipitation gauge data found in [karnataka_precipitation_gauge_data](karnataka_precipitation_gauge_data/).
        
        - [precipitation_gauge_gnn_split.ipynb](code_files/model_notebooks/precipitation_gauge_gnn_split.ipynb) is the second iteration where the model preparation sequence is finalized, with splitting occuring earlier on. It still uses only the precipitation gauge data found in [karnataka_precipitation_gauge_data](karnataka_precipitation_gauge_data/). This is also the version with the magnitude-stability analysis across Karnataka.

        - [precipitation_gauge_gnn_split_bias.ipynb](code_files/model_notebooks/precipitation_gauge_gnn_split_bias.ipynb) is the third iteration where the model is trained using data after adding a small bias and log transforming the data. It still uses the precipitation gauge data found in [karnataka_precipitation_gauge_data](karnataka_precipitation_gauge_data/) for training. However, for evaluation, it uses the GSMaP_ISRO data for Karnataka and Maharashtra found in [extracted_gsmap_isro_data](extracted_gsmap_isro_data).

        - [precipitation_gauge_tester.ipynb](code_files/model_notebooks/precipitation_gauge_tester.ipynb) is the fourth iteration where the code is modularized into functions for easier running and debugging. From here on, precipitation data from the GSMaP_ISRO dataset is exclusively used for training and evaluation. Training is performed only over Karnataka. The GSMaP_ISRO dataset can be found in [extracted_gsmap_isro_data](extracted_gsmap_isro_data). No ground truths are generated for the GSMaP_ISRO data locations in this version.

        - [precipitation_gauge_tester_1_minus_cos.ipynb](code_files/model_notebooks/precipitation_gauge_tester_1_minus_cos.ipynb) is the fifth iteration where we try experimenting with $1 - cos$ as the similarity metric instead of the $cos$. We also use north-rast Indian and centra India for evaluation, along with ground truth generation.

        - [precipitation_gauge_tester_vanilla_cos.ipynb](code_files/model_notebooks/precipitation_gauge_tester_vanilla_cos.ipynb) is the sixth iteration where we use the $cos$ as the similarity metric.

        - [precipitation_gauge_full_india.ipynb](code_files/model_notebooks/precipitation_gauge_full_india.ipynb) is the seventh iteration where we train the model on TN, KA and MP states and evaluate the performance all over India.

    - [supplementary_files](code_files/supplementary_files/) contains all the other code files that aid with the model training pipeline

        - [precipitation_data_combiner.ipynb](code_files/supplementary_files/precipitation_data_combiner.ipynb) is a code that takes the various years' gauge data from [karnataka_precipitation_gauge_data](karnataka_precipitation_gauge_data) and restructures them to contain data of only the locations present across every year.

        - [hdf5_to_grid_csv.py](code_files/supplementary_files/hdf5_to_grid_csv.py) is a code that takes latitude and longitude ranges along with the output file name as input and then extracts the GSMaP_ISRO data from `.hdf5` format to `.csv` format. The code is parallely executated.

        - [filter_land_points.py](code_files/supplementary_files/filter_land_points.py) is a code that takes the full extracted dataset and then filters to keep only the points that are over Indian landmass.

- [extracted_gsmap_isro_data](extracted_gsmap_isro_data) contains the `.csv` files extracted from the GSMaP_ISRO dataset.

- [gsmap_isro](gsmap_isro) is a soft link to the GSMap_ISRO dataset directory present in the HDD.

- [karnataka_precipitation_gauge_data](karnataka_precipitation_gauge_data) contains soft links to the daily precipitation gauge data across Karnataka from 2015 to 2022.

- [shapefiles](shapefiles) contains various shapefiles for different portions of the country.