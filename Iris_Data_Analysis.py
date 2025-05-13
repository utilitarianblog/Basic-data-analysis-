# =============================
# Iris Dataset Analysis Script
# =============================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set plot styles for better visuals
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# =============================
# Task 1: Load and Explore the Dataset
# =============================

try:
    # Load the Iris dataset
    df = sns.load_dataset("iris")
    print("✅ Dataset loaded successfully!\n")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# Inspect the first few rows
print("📝 First 5 rows of the dataset:")
print(df.head())

# Check data types and missing values
print("\n🔍 Data Types:")
print(df.dtypes)

print("\n🚫 Missing Values:")
print(df.isnull().sum())

# Drop rows with missing values (if any)
df_cleaned = df.dropna()

# Confirm that no rows were dropped
print("\n✔️ Missing Values After Cleaning:")
print(df_cleaned.isnull().sum())

# =============================
# Task 2: Basic Data Analysis
# =============================

# Basic statistics
print("\n📊 Basic Statistics:")
print(df_cleaned.describe())

# Average measurements per species
species_means = df_cleaned.groupby("species").mean()
print("\n📌 Average Measurements per Species:")
print(species_means)

# =============================
# Task 3: Data Visualization
# =============================

# 3.1 Line Chart (Trends)
plt.figure(figsize=(10, 6))
species_means.plot(marker='o', title="Average Flower Measurements by Species")
plt.xlabel("Species")
plt.ylabel("Measurement (cm)")
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# 3.2 Bar Chart (Comparison)
plt.figure(figsize=(10, 6))
species_means.plot(kind="bar", colormap="viridis", title="Average Measurements per Species")
plt.xlabel("Species")
plt.ylabel("Average Measurement (cm)")
plt.xticks(rotation=0)
plt.show()

# 3.3 Histogram (Distribution)
plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned["petal_length"], bins=20, kde=True, color="skyblue")
plt.title("Distribution of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 3.4 Scatter Plot (Relationships)
plt.figure(figsize=(10, 6))
sns.scatterplot(x="sepal_length", y="petal_length", hue="species", data=df_cleaned, palette="deep")
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()

# =============================
# Conclusion
# =============================
print("\n### **Conclusion:**")
print("1. The Iris dataset is clean and complete, with no missing values.")
print("2. Clear distinctions in measurements exist between species, particularly in petal length and width.")
print("3. These differences make the dataset suitable for classification problems.")
print("4. Visualizations clearly highlight these distinctions, providing insight into the relationships between features.")

print("\n✅ Analysis Complete!")
