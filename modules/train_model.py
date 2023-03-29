import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import auc, precision_recall_curve
from modules.feature_calculator import FeatureCalculator

class TrainModel:
    def __init__(self):
          
        self.lr_param_grid = {"solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                              "penalty": ['l2'],
                              "C": [100, 10, 1.0, 0.1, 0.01]}
    
        self.svm_param_grid = {"kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
                               "degree": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                               "C": [100, 10, 1.0, 0.1, 0.01]}

    def get_targets(self, datasets):
        return np.asarray(([1] * len(datasets['coding'])) + ([0] * len(datasets['noncoding'])))
    
    def hyperparameter_tuning(self, model, X, y, param_grid, scoring='f1', cv=10):
        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring)
        grid.fit(X, y)
        return grid.best_estimator_
    
    def train(self, model, X, y, threshold, splits=10):
        bin_preds, preds = [], []
        kf = KFold(n_splits=splits, shuffle=False)
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model.fit(X_train, y_train)
            pred = np.array([p[1] for p in model.predict_proba(X_test)])
            bin_pred = (pred > threshold).astype(int)
            bin_preds.append(bin_pred), preds.append(pred)
        return np.hstack((bin_preds)), np.hstack((preds))

    def evaluate(self, y, preds):
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        return {'accuracy': (tn+tp)/(tn+fp+fn+tp),
                'precision': tp/(tp+fp),
                'recall': tp/(tp+fn),
                'accuracy': (tn+tp)/(tn+fp+fn+tp),
                'specificity': tn/(tn+fp),
                'f1': 2*((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))}

    def optimize_thresh(self, y, preds):
        thresholds = [i/200 for i in range(201)]
        best_threshold = 1
        best_diff = float('inf')
        for threshold in thresholds:
            bin_preds = (preds > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, bin_preds).ravel()
            sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
            diff = abs(specificity - sensitivity)
            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold
        return best_threshold

    def pr_curve(self, y_true, y_proba):
        kf = KFold(n_splits=10)
        fig, ax = plt.subplots()
        mean_precision = 0.0
        mean_recall = np.linspace(0, 1, 100)
        for train_index, test_index in kf.split(y_true):
            y_true_train, y_true_test = y_true[train_index], y_true[test_index]
            y_proba_train, y_proba_test = y_proba[train_index], y_proba[test_index]
            precision, recall, _ = precision_recall_curve(y_true_test, y_proba_test)
            mean_precision += np.interp(mean_recall, np.flip(recall), np.flip(precision))
            plt.scatter(recall, precision, color='grey', s=0.5)
        mean_precision /= 10
        mean_auc = auc(mean_recall, mean_precision)
        plt.plot(np.flip(mean_recall), np.flip(mean_precision), color='black', lw=1)
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.set_ylim([0.4, 1.0])
        ax.set_xlim([0.0, 1.0])
        ax.set_title('precision-recall curve')
        plt.legend(['PR AUC: {}'.format(round(mean_auc, 3))])
        plt.show()

    def accuracy_vs_cutoff_value(self, y_true, y_proba):
        kf = KFold(n_splits=10)
        fig, ax = plt.subplots()
        mean_accuracy = 0.0
        cutoff_values = np.linspace(0, 1, 100)
        for train_index, test_index in kf.split(y_true):
            y_true_train, y_true_test = y_true[train_index], y_true[test_index]
            y_proba_train, y_proba_test = y_proba[train_index], y_proba[test_index]
            accuracies = []
            for cutoff in cutoff_values:
                y_pred = np.where(y_proba_test > cutoff, 1, 0)
                accuracy = accuracy_score(y_true_test, y_pred)
                accuracies.append(accuracy)
            mean_accuracy += np.interp(0.5, cutoff_values, accuracies)
            plt.scatter(cutoff_values, accuracies, color='grey', s=0.5)
        mean_accuracy /= 10
        mean_accuracies = []
        for cutoff in cutoff_values:
            y_pred = np.where(y_proba > cutoff, 1, 0)
            accuracy = accuracy_score(y_true, y_pred)
            mean_accuracies.append(accuracy)
        plt.plot(cutoff_values, mean_accuracies, color='black', lw=1)
        ax.set_xlabel('cutoff value')
        ax.set_ylabel('accuracy')
        ax.set_ylim([0.4, 1.0])
        ax.set_xlim([0.0, 1.0])
        ax.set_title('Accuracy vs cutoff value curve')
        plt.show()

    def roc_curve(self, y_true, y_proba):
        fpr, tpr, _ = metrics.roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='black', lw=1)
        for i in range(0, len(y_true), len(y_true)//10):
            end_index = min(i+len(y_true)//10, len(y_true))
            fpr, tpr, _ = metrics.roc_curve(y_true[i:end_index], y_proba[i:end_index])
            plt.scatter(fpr, tpr, color='grey', s=0.5)
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.legend(['ROC AUC: {}'.format(round(roc_auc, 3))])
        plt.title('Receiving operating characteristic curve')
        plt.show()

    def two_graph_roc_curve(self, y_true, y_proba):
        sensitivity_all, specificity_all = [], []
        thresh = np.arange(0, 1, 0.01)
        for i in range(0, len(y_true), len(y_true)//10):
            sensitivity, specificity = [], []
            end_index = min(i+len(y_true)//10, len(y_true))
            for t in range(len(thresh)):
                y_pred = np.where(y_proba[i:end_index] >= thresh[t], 1, 0)
                tn, fp, fn, tp = confusion_matrix(y_true[i:end_index], y_pred).ravel()
                sensitivity.append(tp/(tp+fn)), specificity.append(tn/(tn+fp))
            plt.scatter(thresh, sensitivity, s=1, color='blue')
            plt.scatter(thresh, specificity, s=1, color='red')
            sensitivity_all.append(sensitivity), specificity_all.append(specificity)
        plt.plot(thresh, np.mean(sensitivity_all, axis=0), color='blue', label='Sensitivity')
        plt.plot(thresh, np.mean(specificity_all, axis=0), color='red', label='Specificity')
        plt.xlabel("Coding probability cutoff")
        plt.ylabel("Performance")
        plt.legend()
        plt.title('Two-graph ROC curve')
        plt.show()

    def get_stacking(self, base_model1, base_model2, meta_model):
        level0 = [('model1', base_model1), ('model2', base_model2)]
        level1 = meta_model
        model = StackingClassifier(estimators=level0, final_estimator=level1)
        return model

    def CoP3E_predict(self, ensembl_pseudogene, X_unk, model):
        trxps = [x[0] for x in ensembl_pseudogene.items() if x[1]['coding'] == 'uncertain']
        preds = [x[1] for x in model.predict_proba(X_unk)]
        return list(zip(trxps, preds))