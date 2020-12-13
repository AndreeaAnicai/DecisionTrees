import numpy as np
import math
import matplotlib.pyplot as plt

class DecisionTree:
    def __init__(self):
        self.conf_matrix = None
        self.tree = None

    @staticmethod
    def _entropy(data):
        """
        Helper function:
        Calculates entropy in a dataset.
        """

        np_data = np.array(data)
        values, counts = np.unique(np_data[:, -1], return_counts=True)
        ent = 0
        for p in counts:
            p_k = p / len(data)
            ent += (-p_k * math.log2(p_k))
        return ent

    def _information_gain(self, data, data_left, data_right):
        """
        Helper function:
        Calculates the information gain from splitting datasets.
        """

        s_left = len(data_left)
        s_right = len(data_right)

        remainder = (s_left / (s_left + s_right)) * self._entropy(data_left) + \
                    (s_right / (s_left + s_right)) * self._entropy(data_right)

        gain = self._entropy(data) - remainder
        return gain

    @staticmethod
    def _test_split(attr, value, dataset):
        """
        Helper function:
        Splits the dataset, according to some attribute value.
        """

        left = dataset[dataset[:, attr] <= value]
        right = dataset[dataset[:, attr] > value]
        return left, right


    def _find_split(self, dataset):
        """
        Helper function:
        Finds the best splitting point in a dataset to max information gain.
        """

        b_index, b_score, b_value = 0, 0, 0
        b_left, b_right = np.array([]), np.array([])
        dataset = np.array(dataset)

        for index in range(len(dataset[0])-1):
            # optimization by sorting the dataset by the attribute
            dataset = dataset[dataset[:, index].argsort(kind='mergesort')]
            for row in dataset:
                # only split at differing values
                if(row[index] != row[index + 1]):
                    left, right = self._test_split(index, row[index], dataset)
                    gain = self._information_gain(dataset, left, right)
                    if gain > b_score:
                        b_index, b_score, b_value = index, gain, row[index]
                        b_left, b_right = left, right
        return {'attribute': b_index, 'value': b_value,
                'left': b_left, 'right': b_right}

    @staticmethod
    def _make_terminal(dataset):
        """Helper function:
        Determines decision tree leaf value."""

        outcomes = [row[-1] for row in dataset]
        return max(set(outcomes), key=outcomes.count)


    def decision_tree_learning(self, train, test=None):
        """
        Trains a decision tree from a dataset. If testset is specified,
        the confusion matrix is calculates and stored as a class value.
        :param train: Training dataset
        :param test: Optional Test Dataset
        :return: Tree (dict), depth of tree (int)
        """

        tree, depth = self._decision_tree_learning_util(train)
        self.tree = tree

        if test is not None:
            self.conf_matrix = self.confusion_matrix(test, tree)

        return tree, depth

    def _decision_tree_learning_util(self, dataset, depth=0):
        """Helper function:
        Trains a decision tree from a dataset.
        """

        if len(set([row[-1] for row in dataset])) <= 1:
            return self._make_terminal(dataset), depth

        node = self._find_split(dataset)
        left, right = node['left'], node['right']
        node['left'], l_depth = self._decision_tree_learning_util(left, depth + 1)
        node['right'], r_depth = self._decision_tree_learning_util(right, depth + 1)
        return node, max(l_depth, r_depth)

    def predict(self, node, row):
        """
        Predicts label from a row of data, given a decision tree.
        :param node: Decicsion tree (dict)
        :param row: Array of features
        :return: Label
        """

        if row[node['attribute']] <= node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']


    def predict_dataset(self, tree, dataset):
        """
        Predicts labels for a dataset using a trained decision tree.
        :param tree: Decision tree (dict)
        :param dataset: Feature dataset to use for prediction
        :return: Labels (list)
        """

        return [self.predict(tree, row) for row in dataset]


    def evaluate(self, test_db, tree):
        """
        Function to calculate the accuracy of the tree.
        :param test_db: Test Dataset
        :param tree: Decision Tree (dict)
        :return: Accuracy (int)
        """

        correct_labels = [row[-1] for row in test_db]
        predicted_labels = self.predict_dataset(tree, test_db)
        valid = sum(1 for i, j in zip(correct_labels, predicted_labels)if i == j)
        return valid/len(test_db)


    def confusion_matrix(self, test_db, tree):
        """
        Function to construct the confusion matrix for a tree from a test dataset.
        :param test_db: Test Dataset
        :param tree: Decision Tree (dict)
        :return: Confusion Matrix (2D np.array)
        """

        actual_labels = [row[-1] for row in test_db]
        predicted_labels = self.predict_dataset(tree, test_db)
        dim = int(max(max(actual_labels), max(predicted_labels), 4))
        conf_matrix = np.zeros(shape=(dim, dim))

        for i in range(len(test_db)):
            row = int(actual_labels[i] - 1)
            col = int(predicted_labels[i] - 1)
            conf_matrix[row][col] += 1
        return conf_matrix


    def relative_confusion_matrix(self, test_db, tree, recalc=True):
        """
        Function to calculate therelative confusion matrix, i.e. in percent.
        :param test_db: Test Dataset
        :param tree: Decision Tree (dict)
        :param recalc: Recalcuation flag default=True (use if you want to
        calculate underlying conf. matrix again)
        :return: Confusion Matrix (2D np.array)
        """

        if recalc:
            abs_conf_matrix = self.confusion_matrix(test_db, tree)
        else:
            abs_conf_matrix = self.conf_matrix

        num_observations = len(test_db)
        rel_conf_matrix = abs_conf_matrix.copy()

        for i in range(len(abs_conf_matrix)):
            for j in range(len(abs_conf_matrix)):
                rel_conf_matrix[i][j] = abs_conf_matrix[i][j] / num_observations
        return rel_conf_matrix


    def precision_recall_rates(self, test_db, tree, recalc=True):
        """
        Function to calculate precision and recall rates on a per-class basis.
        :param test_db: Test Dataset
        :param tree: Decision Tree (dict)
        :param recalc: Recalcuation flag default=True (use if you want to
        calculate underlying conf. matrix again)
        :return: precision rates (list), recall rates (list)
        """

        if recalc:
            conf_matrix = self.confusion_matrix(test_db, tree)
        else:
            conf_matrix = self.conf_matrix

        prec_rates = []
        recall = []
        for i in range(max(len(conf_matrix), 4)):
            if np.count_nonzero(conf_matrix[i]) == 0:
                recall.append(np.nan)
                prec_rates.append(np.nan)
            elif (conf_matrix[i][i]) == 0:  # just included this
                recall.append(0)
                prec_rates.append(0)
            else:
                prec_rates.append(conf_matrix[i][i] / sum(conf_matrix[:, i]))
                recall.append(conf_matrix[i][i] / sum(conf_matrix[i]))
        return prec_rates, recall


    def F1_score(self, test_db, tree, recalc=True):
        """
        Calculates the F1 score on a per-class basis.
        :param test_db: Test Dataset
        :param tree: Decision Tree (dict)
        :param recalc: Recalcuation flag default=True (use if you want to
        calculate underlying conf. matrix again)
        :return: F1 scores (list)
        """

        rates, recall = self.precision_recall_rates(test_db, tree, recalc)
        F1_scores = []
        for i in range(max(len(rates), 4)):

            if(np.isnan(recall[i]) or np.isnan(rates[i])):
                score = np.nan
            elif(recall[i] + rates[i] != 0):
                score = 2 * (rates[i] * recall[i] / (rates[i] + recall[i]))
            else:
                score = 0
            F1_scores.append(score)
        return F1_scores


    def classification_rate(self, test_db, tree, recalc=True):
        """
        Calculates the classification rate on a per-class basis.
        :param test_db: Test Dataset
        :param tree: Decision Tree (dict)
        :param recalc: Recalcuation flag default=True (use if you want to
        calculate underlying conf. matrix again)
        :return: classification rates (list)
        """

        if recalc:
            conf_matrix = self.confusion_matrix(test_db, tree)
        else:
            conf_matrix = self.conf_matrix

        classification_rates = []
        TP_TN_total = 0

        for i in range(max(len(conf_matrix), 4)):
            TP_TN_total += conf_matrix[i][i]

        for i in range(max(len(conf_matrix), 4)):
            FN = sum(conf_matrix[i]) - conf_matrix[i][i]
            FP = sum(conf_matrix[:, i]) - conf_matrix[i][i]
            CR = TP_TN_total / (TP_TN_total + FN + FP)
            classification_rates.append(CR)
        return classification_rates

    @staticmethod
    def _node_has_two_leaves(node):
        """
        Helper function:
        Determines is a node has two leaves.
        """

        if isinstance(node, dict) and not \
           isinstance(node['left'], dict) and not \
           isinstance(node['right'], dict):
            return True
        return False


    def _eval_replacement(self, prune_db, tree, parent, side, node):
        """
        Helper function:
        Evaluate the replacement of a node by a leaf.
        """

        # copies node so it can be reinserted later
        node_cpy = node.copy()
        # calc base score and store leave values
        before = self.evaluate(prune_db, tree)
        l_value = node['left']
        r_value = node['right']
        # replace node by left leaf and evaluate
        parent[side] = l_value
        l_score = self.evaluate(prune_db, tree)

        # replace node by right leaf and evaluate
        parent[side] = r_value
        r_score = self.evaluate(prune_db, tree)

        # if either of the leaf scores is better, replace the node
        if l_score >= before or r_score >= before:
            if r_score > l_score:
                return
            else:
                parent[side] = l_value
                return
        else:
            parent[side] = node_cpy
        return

    def prune_tree(self, prune_db, root, test=None):
        """
        Function to prune a decision tree.
        Modifies the tree in place (no value returned).
        :param prune_db: Dataset for pruning
        :param root: Deision tree (dict)
        :param test: Optional test dataset for
        calculating the confusion matrix
        """

        self._prune_tree_util(prune_db, root, root)
        if test is not None:
            self.conf_matrix = self.confusion_matrix(test, root)


    def _prune_tree_util(self, prune_db, root, node):
        """
        Helper function:
        Executes the pruning of a decision tree.
        """

        # check if the node is a leave
        if (self._node_has_two_leaves(node['left'])):
            self._eval_replacement(prune_db, root, node, 'left', node['left'])

        if (self._node_has_two_leaves(node['right'])):
            self._eval_replacement(prune_db, root, node, 'right', node['right'])

        # move left
        if (isinstance(node['left'], dict)):
            self._prune_tree_util(prune_db, root, node['left'])

        # move right
        if (isinstance(node['right'], dict)):
            self._prune_tree_util(prune_db, root, node['right'])
        return

    @staticmethod
    def split_test_val_train(dataset, prop_t, prop_v):
        """
        Splits a dataset into training, validation and
        test set, according to prop_t and prop_v.
        :param dataset: Dataset to split
        :param prop_t: Proportion to use for testing
        :param prop_v: Proportion to use for validation
        :return: datasets (list)
        """

        dataset_copy = np.copy(dataset)
        np.random.shuffle(dataset_copy)
        n = len(dataset_copy)
        sets = np.split(dataset_copy, [int(n*prop_t), int(n*(prop_v + prop_t))])
        return sets[0], sets[1], sets[2]


    def get_depth(self, tree):
        """
        Computes the depth of a tree.
        :param tree: Decision tree (dict)
        :return: depth of tree (int)
        """

        l_depth = self.get_depth(tree['left']) if isinstance(tree['left'], dict) else 0
        r_depth = self.get_depth(tree['right']) if isinstance(
            tree['right'], dict) else 0
        return max(l_depth, r_depth) + 1


def k_fold_evaluation(dataset, k, prune=True):
    """
    K-fold corss-validation, including evaluation of the resulting models.
    :param dataset: Dataset to use for cross-validation
    :param k: Number of iterations
    :param prune: Pruning flag
    :return: Max. precision score (int), avg. precision (int),
    avg. F1 scores (list), avg. recall rates (list),
    avg. precision rates (list), avg. conf. matrix (2D np.array),
    avg. depth (float), avg. classification rate (list)
    """

    dataset_copy = np.copy(dataset)
    np.random.shuffle(dataset_copy)
    test_sets = np.array_split(np.array(dataset_copy), k)

    max_score = 0
    # create score lists
    precisions = []
    F1_scores = []
    recall_rates = []
    precision_rates = []
    conf_matrices = []
    depths = []
    classification_rates = []

    for fold_n in range(len(test_sets)):

        # get the fold not used for testing
        tree_k = DecisionTree()
        training_list = [s for i, s in enumerate(test_sets) if i != fold_n]
        # split the remaining sets into training and validation sets
        X_val = training_list.pop()
        X_train = np.concatenate(training_list)
        # train the tree, prune and evaluate
        tree, depth = tree_k.decision_tree_learning(X_train, test_sets[fold_n])

        if prune:
            tree_k.prune_tree(X_val, tree, test_sets[fold_n])

        # generate all evaluations and scores
        score = tree_k.evaluate(test_sets[fold_n], tree)
        F_1 = tree_k.F1_score(test_sets[fold_n], tree, False)
        conf_matrix = tree_k.relative_confusion_matrix(test_sets[fold_n], tree, False)
        # print(conf_matrix)
        precision_r, recall_r = tree_k.precision_recall_rates(test_sets[fold_n], tree, False)
        class_rate = tree_k.classification_rate(test_sets[fold_n], tree, False)

        # append the scores etc. to lists
        precisions.append(np.array(score))
        F1_scores.append(np.array(F_1))
        conf_matrices.append(np.array(conf_matrix))
        recall_rates.append(np.array(recall_r))
        precision_rates.append(np.array(precision_r))
        depths.append(tree_k.get_depth(tree))
        classification_rates.append(class_rate)

        if score > max_score:
            max_score = score

    # calculate all average statistics
    avg_precision = np.nanmean(precisions)
    avg_F1_scores = np.nanmean(F1_scores, axis=0)
    avg_recall_rates = np.nanmean(recall_rates, axis=0)
    avg_precision_rates = np.nanmean(precision_rates, axis=0)
    avg_depth = np.mean(depths)
    avg_class_rate = np.mean(classification_rates, axis=0)

    # Calculate the average confusion matrix
    avg_conf_matrix = np.zeros(shape=(4, 4))
    for matrix in conf_matrices:
        avg_conf_matrix += matrix
    avg_conf_matrix = np.divide(avg_conf_matrix, k)

    # return the tree with the highest precision score
    return max_score, avg_precision, avg_F1_scores, avg_recall_rates, \
        avg_precision_rates, avg_conf_matrix, avg_depth, avg_class_rate


def plot_tree_util(tree, depth, x=0.5, y=1, p_x=0.5, p_y=1, c=2):
    """
    Helper function:
    Recursively plots a tree.
    """

    new_x_l = x - 1/c
    new_x_r = x + 1/c
    new_y = y - 1/depth

    # print a node
    if isinstance(tree, dict):
        text = "att:" + str(tree['attribute']) + "\n split:" + \
               str(tree['value'])

        plt.text(x, y, text, size=8, ha="center", va="center",
                 bbox=dict(boxstyle="round", ec='blanchedalmond', fc='lightsteelblue'))

    # print a leaf
    else:
        text = "leaf:" + str(tree)
        plt.text(x, y, text, size=8, ha="center", va="center",
                 bbox=dict(boxstyle="round", ec='blanchedalmond',
                           fc='seashell'))
        return

    # connect previos node and current one and move left / right
    plt.plot([p_x, new_x_l], [p_y, new_y], 'k-')
    plot_tree_util(tree['left'], depth,
                   new_x_l, new_y, new_x_l, new_y, c*2)

    plt.plot([p_x, new_x_r], [p_y, new_y], 'k-')
    plot_tree_util(tree['right'], depth,
                   new_x_r, new_y, new_x_r, new_y, c*2)


def plot_tree(tree, depth, title, file_name):
    """
    Plots a decision tree.
    :param tree: Decision tree (dict)
    :param depth: Depth of tree (int)
    :param title: Title for the plot (string)
    :param file_name: File name for saving plot (string)
    """

    plt.figure(figsize=(20, 10))
    plot_tree_util(tree, depth)
    plt.title(title, fontdict=dict(size=15))
    plt.axis('off')
    plt.savefig(file_name, bbox_inches='tight')
