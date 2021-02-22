from core.config.paths import static_path_train, static_path_test
from core.helper.load_dataset import LoadData
from core.helper.preprocess import PreProcess
from core.helper.feature_extraction_and_implementation import FeatureExtractAndImplementation


def start():
    print("Load data")
    trainNews_df, testNews_df, target_names = LoadData(static_path_train, static_path_test).start()
    print("Pre-Process")
    trainNews_df = PreProcess().start(trainNews_df)
    testNews_df = PreProcess().start(testNews_df)
    print("Feature Extraction and Implementation")
    FeatureExtractAndImplementation(trainNews_df, testNews_df, target_names).start()


if __name__ == "__main__":
    start()


