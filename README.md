# Implementation of Principal Component Analysis (PCA) for Dimensionality Reduction on Energy Data
<H3>NAME: KEERTHANA S</H3>
<H3>REGISTER NO.: 212223240070</H3>
<H3>EX. NO.10</H3>
<H3>DATE:11.11.24</H3>

## AIM:
To implement Principal Component Analysis (PCA) to reduce the dimensionality of the energy data.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Load the Data**  
   Import the dataset to begin the dimensionality reduction process.

2. **Explore the Data**  
   Perform an initial analysis to understand data characteristics, distributions, and potential patterns.

3. **Preprocess the Data (Feature Scaling)**  
   Scale features to ensure consistency, preparing the data for principal component analysis (PCA).

4. **Apply PCA for Dimensionality Reduction**  
   Use PCA to reduce the dataset’s dimensionality while retaining the most significant features.

5. **Analyze Explained Variance**  
   Assess the variance explained by each principal component to determine the effectiveness of dimensionality reduction.

6. **Visualize Principal Components**  
   Create visualizations of the principal components to interpret patterns and clusters in reduced dimensions.

## Program:
```py
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Load the dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0187EN-SkillsNetwork/labs/module%203/data/HeightsWeights.csv"
data = pd.read_csv(url)

# Step 2: Preprocess the data (Feature Scaling)
X = data[['Height(Inches)', 'Weight(Pounds)']]  # Select features for PCA

# Scale the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA for dimensionality reduction
pca = PCA(n_components=2)  # Reduce to 2 principal components
X_pca = pca.fit_transform(X_scaled)

# Step 4: Analyze the explained variance
explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance Ratio: {explained_variance}")
print(f"Total Explained Variance: {sum(explained_variance)}")

# Step 5: Visualize the principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)  # Scatter plot of the 2 principal components
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - Heights and Weights')
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/7f0f3226-8c9a-45e7-a318-19e681ac7eb2)


## Result:
Thus, Principal Component Analysis (PCA) was successfully implemented to reduce the dimensionality of the energy dataset.

