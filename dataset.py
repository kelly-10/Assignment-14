# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    # Load the Iris dataset
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame
    print("Dataset loaded successfully!\n")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# No missing values in the Iris dataset, but here’s how to handle them
# df = df.dropna()  # or df.fillna(method='ffill', inplace=True)

# Task 2: Basic Data Analysis
print("\nBasic Statistics:")
print(df.describe())

# Group by species and calculate mean of numeric columns
grouped_means = df.groupby('target').mean()
print("\nMean of numerical columns by species (target):")
print(grouped_means)

# Mapping target to species names for clarity
df['species'] = df['target'].map(dict(zip(range(3), iris_data.target_names)))

# Task 3: Data Visualization
# Set a clean visual style
sns.set(style="whitegrid")

# 1. Line Chart: Mean sepal length per species
mean_sepal = df.groupby('species')['sepal length (cm)'].mean()
mean_sepal.plot(kind='line', marker='o', title='Mean Sepal Length per Species')
plt.ylabel('Sepal Length (cm)')
plt.xlabel('Species')
plt.grid(True)
plt.show()

# 2. Bar Chart: Average petal length per species
avg_petal = df.groupby('species')['petal length (cm)'].mean()
avg_petal.plot(kind='bar', color='skyblue', title='Average Petal Length per Species')
plt.ylabel('Petal Length (cm)')
plt.xlabel('Species')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 3. Histogram: Distribution of sepal width
plt.hist(df['sepal width (cm)'], bins=10, color='lightgreen', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 4. Scatter Plot: Sepal length vs Petal length
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# Observations
print("\nFindings:")
print("- Iris-setosa tends to have shorter petal lengths compared to other species.")
print("- Sepal width is mostly distributed around 3.0 cm.")
print("- Clear visual distinction in petal length and sepal length among species, useful for classification.")
