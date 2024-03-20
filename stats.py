import sys
import numpy as np

def read_matrix(path, name, n, m):
    with open(path + name, 'rb') as file:
        matrix = np.fromfile(file, dtype=np.float32)
    return np.reshape(matrix, (n, m))

def amount_uniques(matrix):
    return np.unique(matrix.flatten()).shape[0]

def element_per_row(matrix):
    rows = [[] for _ in range(matrix.shape[0])]
    for row in range(matrix.shape[0]):
        rows[row] = np.unique(matrix[row])
    return rows

def freq_uniques(matrix):
    dicc = dict()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] not in dicc.keys():
                dicc[matrix[i][j]] = 0
            dicc[matrix[i][j]] += 1
    return dicc.items()

if len(sys.argv) == 1:
    print("# Missing arguments! `python3 stats.py path matrices sizes`")
    print("# Matrices needs to be separated by :")
    print("# Sizes needs to be separeted by : and format rsxcs")
else:
    path = sys.argv[1]
    matrices = sys.argv[2].split(":")
    sizes = [list(map(int, size.split("x"))) for size in sys.argv[3].split(":")]

    if len(sizes) != len(matrices):
        print("# Different amount sizes and matrices")
    else:
        # freq of unique elems
        for i in range(len(matrices)):
            matrix = read_matrix(path, matrices[i], sizes[i][0], sizes[i][1])
            freqs = freq_uniques(matrix)
            with open(path + matrices[i] + "_freq_plot", 'w') as file:
                file.write("\n".join([" ".join([str(freq[0]), str(freq[1])]) for freq in freqs]))

#        # elements per row
#        type_matrices = dict()
#        for i in range(len(matrices)):
#            matrix = matrices[i]
#            aux = matrix.split("-")
#            matrix = "-".join(aux[0:len(aux) - 1])
#            layer = aux[-1]
#            if matrix not in type_matrices.keys():
#                type_matrices[matrix] = []
#            type_matrices[matrix].append((layer, element_per_row(read_matrix(path, matrices[i],
#                sizes[i][0], sizes[i][1]))))
#
#        for key, values in type_matrices.items():
#            with open(path + key + "_toplot", 'w') as file:
#                file.write("value layer row\n")
#                for value in values:
#                    layer, rows = value
#                    for i in range(len(rows)):
#                        for ele in rows[i]:
#                            file.write(f"{ele} {layer} {i}\n")
#
#
#        # this is for unique elements
#        suma = 0
#        for i in range(len(matrices)):
#            matrix = read_matrix(path, matrices[i], sizes[i][0], sizes[i][1])
#            suma += amount_uniques(matrix)
#        average = suma / len(matrices)
#        print("Sum of uniques elements per matrix", suma)
#        print("Average amount of uniques elements per matrix", average)
