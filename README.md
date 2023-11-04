## Principal Component Analysis
In simple terms, the goal of PCA is to reduce dimensions or reduce the number of attributes in a dataset without reducing information. For example, in a dataset of house prices. In PCA, each attribute is called a principal component. If there are 10 attributes in the dataset, it means there are 10 principal components. In the image below there is a histogram of the 10 principal components and the variance of each principal component.

## Principal Component Analysis Coding
Import the required libraries. Then we input the iris dataset and divide the data into train set and test set.

- from sklearn import datasets
- from sklearn.model_selection import train_test_split
 
- iris = datasets.load_iris()
- atribut = iris.data
- label = iris.target

## Divide The Dataset
Divide the dataset into train set and test set.
- X_train, X_test, y_train, y_test = train_test_split(atribut, label, test_size=0.2, random_state=1)

## Decision Tree
We will use the Decision Tree model and calculate its accuracy without using PCA. Accuracy without PCA is 0.9666. The accuracy of your model may differ from the output below.
- from sklearn import tree
 
- decision_tree = tree.DecisionTreeClassifier()
- model_pertama = decision_tree.fit(X_train, y_train)
- model_pertama.score(X_test, y_test)

The display of accuracy results without PCA from the code above is as follows.
![image](https://github.com/diantyapitaloka/Principal-Analysis/assets/147487436/816e8db5-ef9b-46bb-ad7f-f096d2d02e0b)

## Variance Each Attributes
Then we will use PCA and calculate the variance of each attribute. The result is that 1 attribute has a variance of 0.922, which means that this attribute holds high information and is much more significant than other attributes.

- from sklearn.decomposition import PCA
 
Create a PCA object with 4 principal components.
- pca = PCA(n_components=4)
 
Applying PCA to the dataset.
- pca_attributes = pca.fit_transform(X_train)
 
View the variance of each attribute.
- pca.explained_variance_ratio_

The results for each attribute are as follows.
![image](https://github.com/diantyapitaloka/Principal-Analysis/assets/147487436/271aa3b0-acdb-47f3-9a71-a7314b57b1d0)

## Variance on Principal Component Analysis
Looking at the previous variance, we can take the 2 best principal components because the total variance is 0.976 which is quite high.

PCA with 2 principal components.
- pca = PCA(n_components = 2)
- X_train_pca = pca.fit_transform(X_train)
- X_test_pca = pca.fit_transform(X_test)

## Accuracy of The Classifier
We will test the accuracy of the classifier after using PCA. Classifier accuracy test.
- model2 = decision_tree.fit(X_train_pca, y_train)
- model2.score(X_test_pca, y_test)

## Final Look Principal Component Analysis Testing
The results of accuracy testing after using PCA are as below.
![image](https://github.com/diantyapitaloka/Principal-Analysis/assets/147487436/ffdc24f9-15e6-4bc1-9665-7e13041b1a0a)


