import os

root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
static_path_train = os.path.abspath(os.path.join(root_path, "core", "static", "train"))
static_path_test = os.path.abspath(os.path.join(root_path, "core", "static", "test"))

