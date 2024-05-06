'''
This file is used to run all the scripts in the src folder in sequence.
'''

# import data.make_dataset
import features.build_features
import models.train_model
import models.predict_model
import visualization.visualize

if __name__ == "__main__":
    print("Running src.data.make_dataset.py...")
    # src.data.make_dataset.main()

    print("Running src.features.build_features.py...")
    features.build_features.main()

    print("Running src.models.train_model.py...")
    models.train_model.main()

    print("Running src.models.predict_model.py...")
    models.predict_model.main()

    print("Running src.visualization.visualize.py...")
    visualization.visualize.main()
