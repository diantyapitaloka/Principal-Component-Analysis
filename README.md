## 🍘🍜🍣 Principal Component Analysis 🍣🍜🍘
- In simple terms, the goal of PCA is to reduce dimensions or reduce the numbers of attributes in a dataset without reducing information. For example, in a dataset of house prices. In PCA, each attribute is called a principal component. If there are 10 attributes in the datasets, it means there are 10 principal components. In the image below there is a histogram of the 10 principal components and the variance of each principal component.
- The primary goal of the first principal component (PC1) is to captures the maximum possible variance (spread) of the data. By "stretching" the data along the axis where it varies the most, PCA ensures that the most significant patterns are highlighted first.
- If you have 10 original attributes, you start with 10 principal components. However, you might find that the first 3 components explain 95% of the variance. By keeping only those 3, you reduce your 10D problem to a 3D problem while losing only a tiny fraction of the original information.
- Each principal component is perpendicular (orthogonal) to the ones before it. This means PC2 is completely uncorrelated with PC1. In a dataset where many attributes are redundant (like "square footage" and "number of rooms"), PCA strips away that overlap, ensuring each new component provides unique information.
- PCA is extremely sensitive to the scale of your data. If one attribute is measured in "kilometers" and another in "millimeters," the one with the larger numbers will falsely appear to have higher variance. For PCA to work correctly, you must standardize your data (centering it so the mean is 0 and the standard deviation is 1) before running the algorithm.
- Behind the scenes, PCA uses linear algebra. Eigenvectors determine the direction of the new axes (the principal components), while Eigenvalues determine the magnitude or the amount of variance explained by that specific component. A high eigenvalue means that principal components is carrying a lot of "weight."
- By discarding the "tail end" principal components (those with very low variance), you aren't just saving space—you’re often cleaning your data. These minor components frequently represent random noise rather than actual patterns. Removing them can actually make machine learning models more accurate and less prone to overfitting.
- One of the most practical uses for PCA is turning "invisible" high-dimensional data into something we can actually see. Since humans can't visualize 10 dimensions, we use PCA to collapse those 10 attributes into PC1 and PC2. We can then plot these on a 2D scatter plot to see if there are any natural clusters or outliers in the data.

![image](https://github.com/diantyapitaloka/Principal-Analysis/assets/147487436/1729ddb9-bb3c-45ac-b526-5c6048b1a30a)


## 🍘🍜🍣 Principal Component Analysis Coding 🍣🍜🍘
- In a house price dataset, attributes like "number of rooms" and "square footage" are usually highly correlated. PCA creates new components that are orthogonal (at right angles) to each other. This means every principal component is mathematically independent, eliminating redundant information.
- Import the required libraries. Then we input the iris dataset and divide the data into train set and test set.
- Feature Scaling is Essential: PCA is highly sensitive to the scale of the original features because it maximizes variance. You must standardize your data so that features with larger numeric ranges don't unfairly dominate the principal components.
- Train-Test Separation Logic: It is crucial to split your data into training and testing sets before applying PCA. This prevents "data leakage," ensuring the principal components are identified based only on information available during training.

```
from sklearn import datasets
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
atribut = iris.data
label = iris.target
```

## 🍘🍜🍣 Divide The Dataset 🍣🍜🍘
Divide the dataset into train set and test set.
```
X_train, X_test, y_train, y_test = train_test_split(atribut, label, test_size=0.2, random_state=1)
```

## 🍘🍜🍣 Decision Tree 🍣🍜🍘
We will use the Decision Tree model and calculate its accuracy without using PCA. Accuracy without PCA is 0.9666. The accuracy of your model may differ from the output below.
```
from sklearn import tree
```
 
```
decision_tree = tree.DecisionTreeClassifier()
model_pertama = decision_tree.fit(X_train, y_train)
model_pertama.score(X_test, y_test)
```

The display of accuracy results without PCA from the code above is as follows.
![image](https://github.com/diantyapitaloka/Principal-Analysis/assets/147487436/816e8db5-ef9b-46bb-ad7f-f096d2d02e0b)

## 🍘🍜🍣 Variance Each Attributes 🍣🍜🍘
Then we will use PCA and calculate the variance of each attribute. The result is that 1 attribute has a variance of 0.922, which means that this attribute holds high information and is much more significant than other attributes.

```
from sklearn.decomposition import PCA
```
 
Create a PCA object with 4 principal components.
```
pca = PCA(n_components=4)
```
 
Applying PCA to the dataset.
```
pca_attributes = pca.fit_transform(X_train)
```
 
View the variance of each attribute.
```
pca.explained_variance_ratio_
```

The results for each attribute are as follows.
![image](https://github.com/diantyapitaloka/Principal-Analysis/assets/147487436/271aa3b0-acdb-47f3-9a71-a7314b57b1d0)

## 🍘🍜🍣 Variance on Principal Component Analysis 🍣🍜🍘
Looking at the previous variance, we can take the 2 best principal components because the total variance is 0.976 which is quite high.

PCA with 2 principal components.
```
pca = PCA(n_components = 2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)
```

## 🍘🍜🍣 Accuracy of The Classifier 🍣🍜🍘
We will test the accuracy of the classifier after using PCA. Classifier accuracy test.
```
model2 = decision_tree.fit(X_train_pca, y_train)
model2.score(X_test_pca, y_test)
```

## 🍘🍜🍣 Final Look Principal Component Analysis Testing 🍣🍜🍘
The results of accuracy testing after using PCA are as below.
The final visualization as below :
![image](https://github.com/diantyapitaloka/Principal-Analysis/assets/147487436/ffdc24f9-15e6-4bc1-9665-7e13041b1a0a)

## 🍘🍜🍣 License 🍣🍜🍘
- Copyright by Diantya Pitaloka


