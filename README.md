ml_config_management
==============================

development of pipeline management and model_training system

# development of pipeline management and model_training system

### How to run:
**Step 1:** 
Download and unzip all files to a folder of your choice
**Step 2:**
*Option 1:*
Open folder "src". Open __init with a python editor and run the file. This will give you printed updates in your shell.
*Option 2:*
Open folder "src" and run __init. This will run the file through the command terminal.
**Step 3:** 
Analyze results in *../remla/reports* and *../remla/reports/figures*

### Content:
# build_features.py
*This script loads text data from files specified in a YAML configuration file or through direct file paths, preprocesses the data by tokenizing and padding sequences, encodes labels, and then saves the processed data into numpy arrays.*

# train_model.py
*This script loads data, builds and trains the phishing detection model using a neural network defined in model_definition, and then saves the trained model.*

# model_definition.py
*This script defines a function build_model that constructs a convolutional neural network (CNN) model for text classification using Keras. The architecture includes several convolutional layers with different kernel sizes, max pooling layers, dropout layers, and a final dense layer for classification. The model is configured based on parameters loaded from a YAML configuration file.*

# predict_model.py
*This script loads a trained neural network model for phishing detection, evaluates its performance on test data, generates a classification report, confusion matrix, and accuracy score, and visualizes the confusion matrix as a heatmap, saving the results and figures for reporting purposes.*

# make_dataset.py
*This script creates and downloads the dataset from a cloud service. Due to our DVC setup currently not in use*

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
