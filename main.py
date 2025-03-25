import numpy as np
import matplotlib.pyplot as plt

# ======================== Impurity Measures ============================= #

def gini_impurity(labels):
    if len(labels) == 0:
        return 0
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return 1 - np.sum(fractions ** 2)

def entropy(labels):
    if len(labels) == 0:
        return 0
    counts = np.unique(labels, return_counts=True)[1]
    fractions = counts / float(len(labels))
    return -np.sum(fractions * np.log2(fractions))


# ======================== Splitting Utilities ============================= #

def weighted_impurity(groups, criterion):
    total = sum(len(group) for group in groups)
    if criterion == 'gini':
        return sum(len(group) / total * gini_impurity(group) for group in groups)
    else:
        return sum(len(group) / total * entropy(group) for group in groups)

def split_node(X, y, index, value):
    x_index = X[:, index]
    if X[0, index].dtype.kind in ['i', 'f']:
        mask = x_index >= value
    else:
        mask = x_index == value
    left = [X[~mask], y[~mask]]
    right = [X[mask], y[mask]]
    return left, right

def get_best_split(X, y, criterion):
    best_index, best_value, best_score, children = None, None, float('inf'), None
    for index in range(X.shape[1]):
        for value in np.unique(X[:, index]):
            groups = split_node(X, y, index, value)
            impurity = weighted_impurity([groups[0][1], groups[1][1]], criterion)
            if impurity < best_score:
                best_index, best_value, best_score, children = index, value, impurity, groups
    return {'index': best_index, 'value': best_value, 'children': children}

def get_leaf(labels):
    return np.bincount(labels).argmax()


# ======================== Recursive Split Function ============================= #

def split(node, max_depth, min_size, depth, criterion):
    left, right = node['children']
    del node['children']

    if left[1].size == 0:
        node['right'] = get_leaf(right[1])
        return
    if right[1].size == 0:
        node['left'] = get_leaf(left[1])
        return
    if depth >= max_depth:
        node['left'], node['right'] = get_leaf(left[1]), get_leaf(right[1])
        return
    if left[1].size <= min_size:
        node['left'] = get_leaf(left[1])
    else:
        result = get_best_split(left[0], left[1], criterion)
        if result['children'][0][1].size == 0:
            node['left'] = get_leaf(result['children'][1][1])
        elif result['children'][1][1].size == 0:
            node['left'] = get_leaf(result['children'][0][1])
        else:
            node['left'] = result
            split(node['left'], max_depth, min_size, depth + 1, criterion)

    if right[1].size <= min_size:
        node['right'] = get_leaf(right[1])
    else:
        result = get_best_split(right[0], right[1], criterion)
        if result['children'][0][1].size == 0:
            node['right'] = get_leaf(result['children'][1][1])
        elif result['children'][1][1].size == 0:
            node['right'] = get_leaf(result['children'][0][1])
        else:
            node['right'] = result
            split(node['right'], max_depth, min_size, depth + 1, criterion)


# ======================== Train Tree ============================= #

def train_tree(X_train, y_train, max_depth, min_size, criterion='gini'):
    X = np.array(X_train)
    y = np.array(y_train)
    root = get_best_split(X, y, criterion)
    split(root, max_depth, min_size, 1, criterion)
    return root


# ======================== Visualize Tree ============================= #

CONDITION = {
    'numerical': {'yes': '>=', 'no': '<'},
    'categorical': {'yes': 'is', 'no': 'is not'}
}

def visualize_tree(node, depth=0):
    if isinstance(node, dict):
        condition = CONDITION['numerical'] if isinstance(node['value'], (int, float)) else CONDITION['categorical']
        print('{}|- X{} {} {}'.format(depth * '  ', node['index'] + 1, condition['no'], node['value']))
        if 'left' in node:
            visualize_tree(node['left'], depth + 1)
        print('{}|- X{} {} {}'.format(depth * '  ', node['index'] + 1, condition['yes'], node['value']))
        if 'right' in node:
            visualize_tree(node['right'], depth + 1)
    else:
        print('{}Leaf: {}'.format(depth * '  ', node))


# ======================== Test Tree ============================= #

X_train = [['tech', 'professional'],
           ['fashion', 'student'],
           ['fashion', 'professional'],
           ['sports', 'student'],
           ['tech', 'student'],
           ['tech', 'retired'],
           ['sports', 'professional']]

y_train = [1, 0, 0, 0, 1, 0, 1]

tree = train_tree(X_train, y_train, max_depth=2, min_size=2, criterion='gini')
visualize_tree(tree)

# ======================== Gini Plot ============================= #

pos_fraction = np.linspace(0.00, 1.00, 1000)
gini = 1 - pos_fraction**2 - (1 - pos_fraction)**2

plt.plot(pos_fraction, gini)
plt.xlabel('Positive fraction')
plt.ylabel('Gini impurity')
plt.title('Gini Impurity Curve')
plt.show()
