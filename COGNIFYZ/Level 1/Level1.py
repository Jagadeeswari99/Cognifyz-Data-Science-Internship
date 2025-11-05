
# LEVEL 1: Data Exploration, Descriptive Analysis, Geospatial Analysis

# 📦 Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Change the file path if needed
file_path = r"C:\JAGA\COGNIFYZ\Dataset .csv"   # your uploaded dataset
df = pd.read_csv(file_path)

# Show first few rows
print("✅ Dataset Loaded Successfully!")
print(f"Shape of dataset: {df.shape}")
print(df.head())

# 🔹 Task 1: Data Exploration & Preprocessing

print("\n--- Task 1: Data Exploration & Preprocessing ---")

# Check column types
print("\nColumn information:")
print(df.dtypes)

# Check missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Target variable
target_col = "Aggregate rating"

# Basic info about the target
print("\nTarget Column Summary:")
print(df[target_col].describe())

# Histogram of ratings
plt.figure(figsize=(7,4))
plt.hist(df[target_col], bins=20, edgecolor='black')
plt.title("Distribution of Aggregate Rating")
plt.xlabel("Aggregate Rating")
plt.ylabel("Count")
plt.show()

# Check for class imbalance
rating_counts = df[target_col].value_counts().sort_index()
print("\nRatings distribution:\n", rating_counts)

# Handle missing values
df = df.copy()
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include=[object]).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values handled successfully!")

# 🔹 Task 2: Descriptive Analysis

print("\n--- Task 2: Descriptive Analysis ---")

# Basic statistics for numerical columns
num_stats = df.describe().T
print("\nDescriptive statistics for numerical columns:\n")
print(num_stats)

# Explore categorical columns
categorical_cols = ['Country Code', 'City', 'Cuisines']

# Top cuisines
if 'Cuisines' in df.columns:
    cuisine_counts = (
        df['Cuisines'].astype(str)
        .str.split(',')
        .explode()
        .str.strip()
        .value_counts()
    )
    print("\nTop 10 Cuisines:\n", cuisine_counts.head(10))
    plt.figure(figsize=(8,4))
    cuisine_counts.head(10).plot(kind='bar', color='skyblue')
    plt.title("Top 10 Cuisines")
    plt.xlabel("Cuisine Type")
    plt.ylabel("Number of Restaurants")
    plt.show()

# Top cities
if 'City' in df.columns:
    city_counts = df['City'].value_counts().head(10)
    print("\nTop 10 Cities by Number of Restaurants:\n", city_counts)
    plt.figure(figsize=(8,4))
    city_counts.plot(kind='bar', color='lightgreen')
    plt.title("Top 10 Cities by Restaurant Count")
    plt.xlabel("City")
    plt.ylabel("Number of Restaurants")
    plt.show()

# 🔹 Task 3: Geospatial Analysis

print("\n--- Task 3: Geospatial Analysis ---")

lat_col, lon_col = 'Latitude', 'Longitude'

# Scatter plot of restaurant locations
plt.figure(figsize=(7,6))
plt.scatter(df[lon_col], df[lat_col],
            s=(df[target_col] + 1) * 5,
            alpha=0.5)
plt.title("Restaurant Locations (Size ~ Rating)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Correlation between location and rating
corr_lat = df[lat_col].corr(df[target_col])
corr_lon = df[lon_col].corr(df[target_col])
print(f"\nCorrelation between Latitude and Rating: {corr_lat:.4f}")
print(f"Correlation between Longitude and Rating: {corr_lon:.4f}")


print("\n✅ LEVEL 1 COMPLETED SUCCESSFULLY!")
print("Tasks covered:")
print("1️⃣ Data Exploration & Preprocessing")
print("2️⃣ Descriptive Analysis")
print("3️⃣ Geospatial Analysis")
