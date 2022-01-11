import numpy as np


def conf(true, pred, label_list):
    true_int = np.argmax(true, axis=1)
    pred_int = np.argmax(pred, axis=1)

    matrix = np.zeros((len(label_list), len(label_list)), dtype=int)
    for i in range(len(true_int)):
        matrix[true_int[i]][pred_int[i]] += 1

    return matrix


def print_matrix(matrix,label_list):
    count = 0
    print("\t\t", end="\t")
    for l in label_list:
        count += 1
        if count != len(label_list):
            print(l, end="\t")
        else:
            print(l)
    for row_label, row in zip(label_list, matrix):
        print('%s \t %s' % (row_label, '\t'.join('\t%03s' % i for i in row)))
    validate(matrix, label_list)

def validate(matrix, label_list):
    # recall
    for i in range(len(label_list)):
        total_recall = 0
        total_prec = 0
        label = label_list[i]
        for j in range(len(label_list)):
            if i == j:
                sorat = matrix[i][j]
                total_recall = total_recall + matrix[i][j]
                total_prec = total_prec + matrix[j][i]
            else:
                total_recall = total_recall + matrix[i][j]
                total_prec = total_prec + matrix[j][i]

        print(label.upper(), "metric: ", end="\t")
        if sorat != 0:
            recall = sorat/total_recall
            precision = sorat/total_prec
            print("Recall= %.2f" % recall , end="\t")
            print("Precision= %.2f" % precision , end="\t")
        else:
            print("Recall= ", "NaN", end="\t")
            print("Precision= ", "NaN", end="\t")

        f1 = 2*((precision*recall)/(precision+recall))
        print("F1 Score= %.2f" % f1)


