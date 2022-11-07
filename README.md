### Data
- /data/wine
- /data/fire

### Source
- /src/include/
  - Dataset.py : base class for dataset specific classes. During clustering analysis transformed data (from dimensionality reduction) and split data (train/test) are stored in the objects, then picked as the last step in cluster-analysis.ipynb for use in nn.ipynb
  - Fire_data.py : data object for Fire
  - Wine_data.py : data object for Wine
  - const.py : defines where data is located
- cluster-analysis.ipynb : all cluster analysis for the first section of the assignment
- nn.ipynb : all neural network analysis.

### Running
The two jupyter notebooks contain all of the analysis. const.py in /src/include/ defines a relative path for data. The last step in the cluster notebook saves a pickle of the wine data object that contains cluster models, transformed X_train data for each dimensionality reduction technique for use by the neural network notebook.