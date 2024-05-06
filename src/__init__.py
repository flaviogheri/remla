#import data.make_dataset
import features.build_features
import models.train_model
import models.predict_model
import visualization.visualize

def main():
    print("Running src.data.make_dataset.py...")
    #src.data.make_dataset.main()

    print("Running src.features.build_features.py...")
    src.features.build_features.main()

    print("Running src.models.train_model.py...")
    src.models.train_model.main()

    print("Running src.models.predict_model.py...")
    src.models.predict_model.main()

    print("Running src.visualization.visualize.py...")
    src.visualization.visualize.main()

if __name__ == "__main__":
    main()
