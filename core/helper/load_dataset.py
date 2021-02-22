from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt



class LoadData:
    def __init__(self, static_path_train, static_path_test):
        self.static_path_train = static_path_train
        self.static_path_test = static_path_test

    def load_files(self, path, encoding='gbk'):
        """
        :param filename:
         structure such as the following:
            container_folder/
                category_1_folder/
                    file_1.txt
                    file_2.txt
                    ...
                    file_42.txt
                category_2_folder/
                    file_43.txt
                    file_44.txt
        :param path:
        :param encoding:
        :return: Bunch object
        """
        return datasets.load_files(path, encoding=encoding, decode_error='ignore', shuffle=False)

    def load_into_dataframe(self, news):
        df = pd.DataFrame({"target": news.target, "data": news.data})
        return df

    def graph_plot(self, dataframe, target_names):
        dataframe = dataframe[['data', 'target']].groupby(['target']).agg(['count'])
        dataframe = dataframe.reset_index()
        dataframe['label_value'] = dataframe['target'].map(target_names)
        dataframe.plot.bar(x='label_value', y='data', rot=90, figsize=(15, 10,), fontsize=10)
        plt.show()

    def start(self):
        trainNews = self.load_files(path=self.static_path_train, encoding='utf-8')
        testNews = self.load_files(path=self.static_path_test, encoding='utf-8')
        target_names = {i: value for i, value in enumerate(trainNews.target_names)}
        trainNews_df = self.load_into_dataframe(trainNews)
        testNews_df = self.load_into_dataframe(testNews)
        print("Training set dimensions: ", trainNews_df.shape)
        print("Testing set dimensions: ", testNews_df.shape)
        self.graph_plot(trainNews_df, target_names)
        self.graph_plot(testNews_df, target_names)

        return trainNews_df, testNews_df, target_names
