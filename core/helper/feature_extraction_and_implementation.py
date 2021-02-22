from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")


class FeatureExtractAndImplementation:
    def __init__(self, trainNews_df, testNews_df, target_names):
        self.tfidfVectorizer = TfidfVectorizer(tokenizer= lambda x: x.split(" "),
                                      sublinear_tf=True, encoding='utf-8',
                                      decode_error='ignore',
                                      max_df=0.5,
                                      min_df=10)
        self.trainNews_df = trainNews_df
        self.testNews_df = testNews_df
        self.target_names = target_names

    def pipline_for_all_algorithm(self):
        clf1 = Pipeline([
            ('vect', self.tfidfVectorizer),
            ('clf', MultinomialNB(alpha=0.01, fit_prior=True))
        ])
        # Bernoulli Naive Bayes
        clf2 = Pipeline([
            ('vect', self.tfidfVectorizer),
            ('clf', BernoulliNB(alpha=0.01))
        ])

        # SVC Linear Kernel
        clf3 = Pipeline([
            ('vect', self.tfidfVectorizer),
            ('clf', SVC(kernel='linear'))
        ])
        # SVC RBF Kernel
        clf4 = Pipeline([
            ('vect', self.tfidfVectorizer),
            ('clf', SVC(kernel='rbf'))
        ])
        # SVC Poly Kernel
        clf5 = Pipeline([
            ('vect', self.tfidfVectorizer),
            ('clf', SVC(kernel='poly'))
        ])

        return clf1, clf2, clf3, clf4, clf5

    def trainAndEvaluate(self, clf, xTrain, xTest, yTrain, yTest):
        clf.fit(xTrain, yTrain)
        yPred = clf.predict(xTest)
        print("Accuracy on Testing Set : ", clf.score(xTest, yTest))
        ''' --- START TEMPORARY ---'''
        # print(str(xTest[0], encoding='utf-8'))
        print("Prediction for 1st sample test")
        print(xTest[119])

        print('Predicted Target ', clf.predict([xTest[119]])[0])
        print('Actual Target ', yTest[119])
        print('Predicted Target Name ', self.target_names[clf.predict([xTest[119]])[0]])
        print('Actual Target Name ', self.target_names[yTest[119]])

        # print(str(xTest[600], encoding='utf-8'))
        print("#############################################################################################")

        print("Prediction for 2nd sample test")

        print(xTest[654])

        print('Predicted Target ', clf.predict([xTest[654]])[0])
        print('Actual Target ', yTest[654])
        print('Predicted Target Name ', self.target_names[clf.predict([xTest[654]])[0]])
        print('Actual Target Name ', self.target_names[yTest[654]])

        # print(str(xTest[1100], encoding='utf-8'))

        print("#############################################################################################")

        print("Prediction for 3rd sample test")
        print(xTest[19])

        print('Predicted Target ', clf.predict([xTest[19]])[0])
        print('Actual Target ', yTest[19])
        print('Predicted Target Name ', self.target_names[clf.predict([xTest[19]])[0]])
        print('Actual Target Name ', self.target_names[yTest[19]])
        ''' --- END TEMPORARY ---'''
        print("#############################################################################################")

        print("Classification Report : ")
        print(metrics.classification_report(yTest, yPred))
        print("Confusion Matrix : ")
        print(metrics.confusion_matrix(yTest, yPred))

    def start(self):
        clf1, clf2, clf3, clf4, clf5 = self.pipline_for_all_algorithm()
        print('Multinominal Naive Bayes')
        self.trainAndEvaluate(clf1, self.trainNews_df['data'].tolist(), self.testNews_df['data'].tolist(),
                              self.trainNews_df['target'].tolist(), self.testNews_df['target'].tolist())
        print('Bernoulli Naive Bayes \n')
        self.trainAndEvaluate(clf2, self.trainNews_df['data'].tolist(), self.testNews_df['data'].tolist(),
                              self.trainNews_df['target'].tolist(), self.testNews_df['target'].tolist())
        print('Linear Kernel SVC \n')
        self.trainAndEvaluate(clf3,self.trainNews_df['data'].tolist(), self.testNews_df['data'].tolist(),
                              self.trainNews_df['target'].tolist(), self.testNews_df['target'].tolist())
        print('RBF Kernel SVC \n')
        self.trainAndEvaluate(clf4, self.trainNews_df['data'].tolist(), self.testNews_df['data'].tolist(),
                              self.trainNews_df['target'].tolist(), self.testNews_df['target'].tolist())
        print('Poly Kernel SVC \n')
        self.trainAndEvaluate(clf5, self.trainNews_df['data'].tolist(), self.testNews_df['data'].tolist(),
                              self.trainNews_df['target'].tolist(), self.testNews_df['target'].tolist())
