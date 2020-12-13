import numpy as np
import sys


import DecisionTree as Tree


def printStats(filename, data_len, test_len, val_len, train_len, depth, precision,
               conf_matrix, F1, precision_rates, recall_rates, class_rates, description):
    """
    Prints results to a training and evaluation round to a file.
    :param filename: File for saving results
    :param data_len: Len of complete dataset
    :param test_len: Len of test set
    :param val_len: Len of validation set
    :param train_len: Len of training set
    :param depth: Depth of the tree
    :param precision: Accuracy of the tree
    :param conf_matrix: Confusion matrix
    :param F1: F1 score
    :param precision_rates: Per-class precision rates
    :param recall_rates: Per-class recall rates
    :param class_rates: Per-class classification rates
    :param description: Description of the test
    """

    print("-------------", filename, "---------------")
    print(description, "\n")
    print("Dataset Length:", data_len, "\n")
    print("Length of -- X_test:", test_len,
          "X_val:", val_len, "X_train:", train_len, "\n")
    print("Depth of Tree:", depth, "\n")
    print('Precision Score:', precision, "\n")
    print("****Confusion Matrix****")
    print(conf_matrix, "\n")
    print("****F1 Score****")
    print(F1, "\n")
    print("****Precision Rates****")
    print(precision_rates, "\n")
    print("****Recall Rates****")
    print(recall_rates, "\n")
    print("****Classification Rates****")
    print(class_rates, "\n")
    print("-----------END OF CASE------------- \n")
    print("**************************************************** \n \n \n")


# Read command line arguments
if (len(sys.argv) < 2):
    print("Insufficient number of arguments. Required: \n input file, output file", )

# Open data file and output file
data_file = sys.argv[1]
output_file = sys.argv[2]

# Redirect stdout to the output file
orig_stdout = sys.stdout
out = open(output_file, 'w')
sys.stdout = out
dataset = np.loadtxt(data_file)
dataset[:, -1] = dataset[:, -1].astype(int)


# create instance of DecisionTree class
tree_inst = Tree.DecisionTree()

# split into training, validation and test data, train and evaluate
test, val, train = tree_inst.split_test_val_train(dataset, 0.2, 0.2)
tree, depth = tree_inst.decision_tree_learning(train, test)
Tree.plot_tree(
    tree, depth, "Examplary Decision Tree Before Pruning", "tree_no_pruning")
score_1 = tree_inst.evaluate(test, tree)

conf_matrix_1 = tree_inst.confusion_matrix(test, tree)

F1_1 = tree_inst.F1_score(test, tree, False)

precision_rates_1, recall_rates_1 = tree_inst.precision_recall_rates(test, tree, False)

class_rates1 = tree_inst.classification_rate(test, tree, False)

# print results to file
printStats(data_file, len(dataset), len(test), len(val), len(train), depth,
           score_1, conf_matrix_1, F1_1, precision_rates_1, recall_rates_1,
           class_rates1,
           "Case: 20% test, 20% val split, no pruning, no cross-validation")

# Prune tree and evaluate again
tree_inst.prune_tree(val, tree, test)
Tree.plot_tree(
    tree, depth, "Examplary Decision Tree After Pruning", "tree_after_pruning")
score_prune = tree_inst.evaluate(test, tree)

conf_matrix_prune = tree_inst.confusion_matrix(test, tree)

F1_prune = tree_inst.F1_score(test, tree, False)

precision_prune, recall_prune = tree_inst.precision_recall_rates(test, tree, False)

depth = tree_inst.get_depth(tree)

class_rates = tree_inst.classification_rate(test, tree, False)

# print results to file
printStats(data_file, len(dataset), len(test), len(val), len(train), depth,
           score_prune, conf_matrix_prune, F1_prune, precision_prune,
           recall_prune, class_rates,
           "Case: 20% test, 20% val split, after pruning, no cross-validation")


# 10 fold cross-validation without pruning
max_score, avg_precision, avg_F1_scores, avg_recall_rates,\
    avg_precision_rates, avg_conf_matrix, avg_depth, avg_class_rate = Tree.k_fold_evaluation(
        dataset, 10, False)

# print results to file
printStats(data_file, len(dataset), len(dataset)//10, len(dataset)//10,
           (len(dataset)//10) * 8, avg_depth,
           avg_precision, avg_conf_matrix, avg_F1_scores,
           avg_precision_rates, avg_recall_rates, avg_class_rate,
           """Case: 10-fold cross-validation without pruning (all scores are
           evg. over the ten folds)""")

# 10 fold cross-validation with pruning
max_score, avg_precision, avg_F1_scores, avg_recall_rates,\
    avg_precision_rates, avg_conf_matrix, avg_depth, avg_class_rate = Tree.k_fold_evaluation(
        dataset, 10)

# print results to file
printStats(data_file, len(dataset), len(dataset)//10, len(dataset)//10,
           (len(dataset)//10) * 8, avg_depth,
           avg_precision, avg_conf_matrix, avg_F1_scores,
           avg_precision_rates, avg_recall_rates, avg_class_rate,
           """Case: 10-fold cross-validation with pruning (all scores are
           evg. over the ten folds)""")
print("Best score achieved over 10-fold cross-validation: ", max_score)

# reset stdout
sys.stdout = orig_stdout
out.close()
