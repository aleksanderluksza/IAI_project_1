from sklearn import tree

file_descriptor = open("dry_bean_train.csv","r")

# split data into a list of lists containing actual values
file_data = file_descriptor.read().splitlines()
for iter in range(len(file_data)):
    file_data[iter] = file_data[iter].split(",")
labels = file_data.pop(0)
print(labels)
for iter in range(5):
    print(file_data[iter])
