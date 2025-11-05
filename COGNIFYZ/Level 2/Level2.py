
# LEVEL 2: Business Analysis & Feature Engineering


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 🔹 Load Dataset
file_path = r"C:\JAGA\COGNIFYZ\Dataset .csv"       # adjust if filename differs
df = pd.read_csv(file_path)

print("✅ Dataset Loaded!")
print(f"Shape: {df.shape}")
print("Columns:", df.columns.tolist())

# Fill small missing values
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include=[object]).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

target_col = "Aggregate rating"

# 🔹 TASK 1: Table Booking & Online Delivery Analysis

print("\n--- Task 1: Table Booking & Online Delivery ---")

# Normalize yes/no text
df["Has Table booking"] = df["Has Table booking"].str.strip().str.lower()
df["Has Online delivery"] = df["Has Online delivery"].str.strip().str.lower()

# Percentages
table_booking_pct = (df["Has Table booking"].eq("yes").mean()) * 100
online_delivery_pct = (df["Has Online delivery"].eq("yes").mean()) * 100

print(f"➡️ % Restaurants offering Table Booking: {table_booking_pct:.2f}%")
print(f"➡️ % Restaurants offering Online Delivery: {online_delivery_pct:.2f}%")

# Compare ratings
avg_rating_table_yes = df.loc[df["Has Table booking"] == "yes", target_col].mean()
avg_rating_table_no = df.loc[df["Has Table booking"] == "no", target_col].mean()
print(f"\nAverage Rating (Table Booking = Yes): {avg_rating_table_yes:.2f}")
print(f"Average Rating (Table Booking = No): {avg_rating_table_no:.2f}")

plt.bar(["Has Booking", "No Booking"],
        [avg_rating_table_yes, avg_rating_table_no],
        color=["green","red"])
plt.title("Average Rating vs Table Booking Availability")
plt.ylabel("Average Rating")
plt.show()

# Online Delivery vs Price Range
price_delivery = (
    df.groupby("Price range")["Has Online delivery"]
    .apply(lambda x: (x == "yes").mean() * 100)
)
print("\n% of restaurants with Online Delivery by Price Range:")
print(price_delivery)
price_delivery.plot(kind="bar", color="skyblue")
plt.title("Online Delivery Availability by Price Range")
plt.xlabel("Price Range")
plt.ylabel("% Restaurants with Online Delivery")
plt.show()

# 🔹 TASK 2: Price Range Analysis

print("\n--- Task 2: Price Range Analysis ---")

most_common_price = df["Price range"].mode()[0]
print(f"Most common price range: {most_common_price}")

avg_rating_per_price = (
    df.groupby("Price range")[target_col].mean().round(2)
)
print("\nAverage rating for each price range:")
print(avg_rating_per_price)

plt.figure(figsize=(6,4))
avg_rating_per_price.plot(kind="bar", color="orange")
plt.title("Average Rating by Price Range")
plt.xlabel("Price Range")
plt.ylabel("Average Rating")
plt.show()

# Color representing highest rating
color_rating = df.groupby("Rating color")[target_col].mean().round(2)
top_color = color_rating.idxmax()
print("\nAverage rating by Rating Color:")
print(color_rating)
print(f"\nColor with highest average rating: {top_color}")

# 🔹 TASK 3: Feature Engineering

print("\n--- Task 3: Feature Engineering ---")

# Derived features
df["Name_Length"] = df["Restaurant Name"].astype(str).str.len()
df["Address_Length"] = df["Address"].astype(str).str.len()

# Encode categorical Yes/No to binary
df["Has_Table_Booking_Binary"] = df["Has Table booking"].map({"yes":1, "no":0})
df["Has_Online_Delivery_Binary"] = df["Has Online delivery"].map({"yes":1, "no":0})

print("\nNew Feature Columns Created:")
print(["Name_Length", "Address_Length",
       "Has_Table_Booking_Binary", "Has_Online_Delivery_Binary"])

# Show sample rows with new features
print("\nSample data after feature engineering:")
print(df[["Restaurant Name","Name_Length","Address_Length",
          "Has_Table_Booking_Binary","Has_Online_Delivery_Binary"]].head())


print("\n✅ LEVEL 2 COMPLETED SUCCESSFULLY!")
print("Tasks done:")
print("1️⃣ Table Booking & Online Delivery Analysis")
print("2️⃣ Price Range Analysis")
print("3️⃣ Feature Engineering")
