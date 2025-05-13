# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Enable inline plotting for Jupyter Notebook
# %matplotlib inline (Uncomment this if using Jupyter Notebook)

# -------------------------------
# Task 1: Load and Explore Dataset
# -------------------------------

try:
    # Load the Iris dataset
    iris_data = load_iris()
    df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

    print("✅ Dataset loaded successfully!\n")
    
    # Display the first few rows
    print("🔍 First 5 rows of the dataset:")
    print(df.head())

    # Check data types and missing values
    print("\n📊 Data Types and Missing Values:")
    print(df.info())
    print("\nMissing values:\n", df.isnull().sum())

    # Clean missing data if any (not expected in this dataset)
    df = df.dropna()

except FileNotFoundError:
    print("❌ File not found.")
except Exception as e:
    print(f"❌ An error occurred: {e}")

# -------------------------------
# Task 2: Basic Data Analysis
# -------------------------------

# Basic statistics
print("\n📈 Basic Statistics:")
print(df.describe())

# Grouping by species and computing mean
grouped = df.groupby("species").mean()
print("\n📊 Mean values grouped by species:")
print(grouped)

# Observation
print("\n🔎 Observation:")
print("From the grouped data, we can see that 'setosa' generally has lower petal lengths and widths compared to 'versicolor' and 'virginica'.")

# -------------------------------
# Task 3: Data Visualization
# -------------------------------

# Set seaborn style
sns.set(style="whitegrid")

# 1. Line Chart: Mean petal length per species (not time-series, but a trend-style example)
plt.figure(figsize=(8, 5))
grouped["petal length (cm)"].plot(marker='o')
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Bar Chart: Average sepal width per species
plt.figure(figsize=(8, 5))
sns.barplot(x=grouped.index, y=grouped["sepal width (cm)"], palette="viridis")
plt.title("Average Sepal Width by Species")
plt.xlabel("Species")
plt.ylabel("Sepal Width (cm)")
plt.tight_layout()
plt.show()

# 3. Histogram: Distribution of petal length
plt.figure(figsize=(8, 5))
plt.hist(df["petal length (cm)"], bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Scatter Plot: Sepal length vs Petal length
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="sepal length (cm)", y="petal length (cm)", hue="species", palette="deep")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.tight_layout()
plt.show()
