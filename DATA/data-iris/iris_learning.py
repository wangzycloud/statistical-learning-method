import pandas as pd

iris = pd.read_csv('E:\Statistical_learning_method\DATA\data-iris\iris.csv', usecols=[1, 2, 3, 4, 5])
iris.info()
head_5 = iris.head()
#print(head_5)
desc = iris.describe()
print(desc)
iris = iris.values
print(type(iris))
data = iris[0:100]
input_vecs = data[:, [0, 1]]
labels = data[:,[4]]

for i in range(labels.shape[0]):
    if labels[i][0] == 'Iris-setosa':
        labels[i][0] = 1
    else:
        labels[i][0] = -1
#print(labels)
#print(labels.shape)