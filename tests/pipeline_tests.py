import os
import co395_cbc_dt.DecisionTree as Tree

import numpy as np

# Test Data sets to evaluate the entropy and information gain functions
T_data = [[1, 2, 2],
          [1, 2, 3],
          [1, 2, 3],
          [1, 3, 2],
          [1, 3, 4],
          [1, 3, 4],
          [1, 3, 3], ]

T_left = [[1, 2, 2],
          [1, 2, 2],
          [1, 2, 4],
          ]

T_right = [[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3],
           [1, 2, 4]]

T_data_2 = [[4, 3, 2],
            [5, 2, 3],
            [5, 2, 3],
            [4, 3, 2],
            [2, 3, 4],
            [2, 3, 4],
            [1, 3, 3]]

T_predict_row = [5, 2, 3]

T_predict_set = [[5, 2, 3],
                 [4, 3, 2],
                 [5, 5, 4]]

T_prune_set = [[5, 2, 3],
               [5, 2, 3]]


direc = os.path.dirname(__file__)
file = os.path.join(direc, '../wifi_db/clean_dataset.txt')

dataset = np.loadtxt(
    "/home/lol_l/Documents/ImperialMaster/machine_learning/co395_cbc_dt/wifi_db/clean_dataset.txt")
#dataset = np.loadtxt(file, delimiter="\t")


# def testEntropy():
#     assert Tree.entropy(T_data) == 1.5566567074628228


# def testInfoGain():
#     assert Tree.informationGain(T_data, T_left, T_right) \
#         == 0.6995138503199657


# def testtestSplit():
#     dataset_np = np.array(T_data)
#     left, right = Tree.testSplit(1, 2, dataset_np)
#     assert (left == [[1, 2, 2],
#                      [1, 2, 3],
#                      [1, 2, 3]]).all()


# def testfindSplit():
#     assert (Tree.findSplit(T_data)['left'] == np.array([[1, 2, 2],
#                                                         [1, 2, 3],
#                                                         [1, 2, 3]])).all()


# def testmakeTerminal():
#     assert (Tree.makeTerminal(T_left) == 2)


# def test_decision_tree_learning():
#     tree, depth = Tree.decision_tree_learning(T_data_2)
#     # print('\n')
#     # Tree.print_tree(tree)
#     assert (tree['left']['left'] == 3)


# def testPredict():
#     tree, depth = Tree.decision_tree_learning(T_data_2)
#     label = Tree.predict(tree, T_predict_row)
#     assert (label == 3)


# def testPredictDataset():
#     tree, depth = Tree.decision_tree_learning(T_data_2)
#     label = Tree.predict_dataset(tree, T_predict_set)
#     assert (label[2] == 3)


# # def test_k_fold_split():
# #     sets = Tree.k_fold_split(T_data_2, 3)
# #     assert (sets != None)


# def test_evaluate():
#     tree, depth = Tree.decision_tree_learning(T_data_2)
#     score = Tree.evaluate(T_predict_set, tree)
#     assert (score == 2/3)


# def test_conf_matrix():
#     tree, depth = Tree.decision_tree_learning(T_data_2)
#     matrix = Tree.confusion_matrix(T_predict_set, tree)
#     print(matrix)
#     assert (matrix[2][3] == 1)


# def test_rel_conf():
#     tree, depth = Tree.decision_tree_learning(T_data_2)
#     rel_matrix = Tree.relative_confusion_matrix(T_predict_set, tree)
#     print(rel_matrix)


# def test_get_depth():
#     tree, depth = Tree.decision_tree_learning(T_data_2)
#     Tree.print_tree(tree)
#     print("clculated depth:", Tree.get_depth(tree))
#     print("actual depth:", depth)
#     test, val, train = Tree.split_test_val_train(dataset, 0.2, 0.2)
#     tree1, depth1 = Tree.decision_tree_learning(train)
#     print("clculated depth:", Tree.get_depth(tree1))
#     print("actual depth:", depth1)


# def test_precision_rates():
#     tree, depth = Tree.decision_tree_learning(T_data_2)
#     rates, recall = Tree.precision_recall_rates(T_predict_set, tree)
#     print(rates, recall)
#     assert (rates[2] == 0.5 and recall[2] == 1)


# def test_F1_score():
#     tree, depth = Tree.decision_tree_learning(T_data_2)
#     scores = Tree.F1_score(T_predict_set, tree)
#     print("Scores: ", scores)
#     print("F1 Score:", scores[1])
#     assert(scores[1] == 1)


# def test_prune_tree():
#     tree, depth = Tree.decision_tree_learning(T_data_2)
#     # Tree.print_tree(tree)
#     Tree.prune_tree(T_prune_set, tree, tree)
#     # Tree.print_tree(tree)
#     #assert (tree['left'] == tree['right'])


# def test_splitting():
#     test, val, train = Tree.split_test_val_train(dataset, 0.2, 0.2)
#     print("total len:", len(dataset))
#     print("len test:", len(test))
#     print("len val:", len(val))
#     print("len train", len(train))


# def test_evaluate_overall_precision():
#     test, val, train = Tree.split_test_val_train(dataset, 0.2, 0.2)
#     tree, depth = Tree.decision_tree_learning(train)
#     score_train = Tree.evaluate(train, tree)
#     print("Training evaluation score (shoul dbe 100%)", score_train)
#     score_1 = Tree.evaluate(val, tree)
#     print("Before pruning:", score_1)
#     Tree.prune_tree(val, tree, tree)
#     score = Tree.evaluate(val, tree)
#     print("Precision score is:", score)
#     Tree.print_tree(tree)
# # def test_k_fold_evaluation():
# #     best_tree, score = Tree.k_fold_evaluation(dataset, 10)
# #     #Tree.print_tree(best_tree)
#     print("best score:", score)

# def test_plotting():
#     tree, depth = Tree.decision_tree_learning(T_data_2)
#     Tree.plot_tree(tree, depth)


# def test_plotting_advanced():
#     tree, depth = Tree.decision_tree_learning(dataset)
#     Tree.plot_tree(tree, depth, "Test Tree", "test_tree")

def test_classfification_rates():
    test, val, train = Tree.split_test_val_train(dataset, 0.2, 0.2)
    tree, depth = Tree.decision_tree_learning(train)
    conf_mat = Tree.confusion_matrix(test, tree)
    print(conf_mat)
    scores = Tree.classification_rate(test, tree)
    print("Scores: ", scores)
