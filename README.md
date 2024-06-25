# ml_config_management

development of pipeline management and model_training system

# development of pipeline management and model_training system

### How to run:

**Step 1:**

Download and unzip all files to a folder of your choice.
Also download and unzip the project dvc remote repository from [DRIVE](https://drive.google.com/file/d/1GY0a_DOoJeKwlM2ffvq5au2X45jPN3H5/view?usp=drive_link). Store the remote directory in the same folder as the remla folder, and import the training/testing/evaluation data using `dvc pull` from within the remla folder.
**Step 2:**

_Option 1:_

Open folder "src". Open \_\_init with a python editor and run the file. This will give you printed updates in your shell.

_Option 2:_

Open folder "src" and run \_\_init. This will run the file in a command terminal.

**Step 3:**
Analyze results in _../remla/reports_ and _../remla/reports/figures_

### Content:

# build_features.py

_This script loads text data from files specified in a YAML configuration file or through direct file paths, preprocesses the data by tokenizing and padding sequences, encodes labels, and then saves the processed data into numpy arrays._

# model_definition.py

_This script defines a function build_model that constructs a convolutional neural network (CNN) model for text classification using Keras. The architecture includes several convolutional layers with different kernel sizes, max pooling layers, dropout layers, and a final dense layer for classification. The model is configured based on parameters loaded from a YAML configuration file._

# train_model.py

_This script loads data, builds and trains the phishing detection model using a neural network defined in model_definition, and then saves the trained model._

# predict_model.py

_This script loads a trained neural network model for phishing detection, evaluates its performance on test data, generates a classification report, confusion matrix, and accuracy score, and visualizes the confusion matrix as a heatmap, saving the results and figures for reporting purposes._

# make_dataset.py

_This script creates and downloads the dataset from a cloud service. Due to our DVC setup currently not in use_

# Project Organization

---

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

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

# Running the linter

Both pylint and flake8 have been used as linters. To run both, run the following command:
lint.bat

If this does not work, you can run the following commands to run both linters separately:
pylint src src/checkers > pylint_output.txt
flake8 > flake8_output.txt

Pylint gives a quality score. 10/10 is the best possible score.
flake8 gives the number of issues. 0 is the best possible score.

New pylint checkers have also been added according to common ML smells. These can be found in src/checkers.

# Run automatic tests locally

Automatic tests are run by executing the shell script run_tests.sh. This can be done with the command
```
./run_tests.sh
```
