
# LEVEL 3: Predictive Modeling, Customer Preference Analysis, Visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset

file_path = r"C:\JAGA\COGNIFYZ\Dataset .csv"   # change if needed
df = pd.read_csv(file_path)
print("Loaded dataset:", df.shape)

# Basic safety / column checks

required_cols = ["Aggregate rating"]
if "Aggregate rating" not in df.columns:
    # try to find similar column
    for col in df.columns:
        if "rating" in col.lower() or "aggregate" in col.lower():
            df.rename(columns={col: "Aggregate rating"}, inplace=True)
            print(f"Renamed column {col} -> 'Aggregate rating'")
            break
if "Aggregate rating" not in df.columns:
    print("ERROR: No target column 'Aggregate rating' found. Exiting.")
    sys.exit(1)

# fill small missing values
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include=[object]).columns:
    df[col].fillna("", inplace=True)

# Task 1: Predictive Modeling

print("\n--- Predictive Modeling ---")

# Feature engineering reuse: name/address length (if present)
if "Restaurant Name" in df.columns:
    df["Name_Length"] = df["Restaurant Name"].astype(str).str.len()
else:
    df["Name_Length"] = 0

if "Address" in df.columns:
    df["Address_Length"] = df["Address"].astype(str).str.len()
else:
    df["Address_Length"] = 0

# Binary encode Has Table booking / Has Online delivery (many datasets use slightly different column names)
def map_yes_no(colname):
    if colname in df.columns:
        return df[colname].astype(str).str.strip().str.lower().map(lambda x: 1 if x in ["yes","y","true","1"] else 0)
    else:
        return pd.Series(0, index=df.index)

df["Has_Table_Booking_Binary"] = map_yes_no("Has Table booking")
df["Has_Online_Delivery_Binary"] = map_yes_no("Has Online delivery")
df["Has_Online_Delivery_Binary"] = df["Has_Online_Delivery_Binary"].fillna(0).astype(int)

# Numeric features we will use if present
num_feats = []
for col in ["Average Cost for two", "Votes"]:
    if col in df.columns:
        num_feats.append(col)
    else:
        # create zeros to keep consistent shape
        df[col] = 0
        num_feats.append(col)

# Price range: keep numeric price range or map if strings
if "Price range" not in df.columns and "Price Range" in df.columns:
    df.rename(columns={"Price Range":"Price range"}, inplace=True)

if "Price range" in df.columns:
    # ensure numeric
    try:
        df["Price_range_num"] = pd.to_numeric(df["Price range"], errors='coerce').fillna(0)
    except Exception:
        df["Price_range_num"] = 0
else:
    df["Price_range_num"] = 0

# Cuisine features: create top-N cuisine binary columns (presence)
top_n = 10
if "Cuisines" in df.columns:
    # explode to find top cuisines
    cuisines_series = df["Cuisines"].astype(str).str.split(",", expand=False).explode().str.strip()
    top_cuisines = cuisines_series.value_counts().head(top_n).index.tolist()
    for cuisine in top_cuisines:
        safe_name = "cuisine_" + cuisine.replace(" ", "_").replace("&","and").replace("/","_").lower()
        df[safe_name] = df["Cuisines"].astype(str).str.contains(cuisine, na=False).astype(int)
else:
    top_cuisines = []
    # create dummy cuisine cols to avoid errors
    for i in range(top_n):
        df[f"cuisine_dummy_{i}"] = 0

# Final feature list
feature_cols = ["Name_Length", "Address_Length",
                "Has_Table_Booking_Binary", "Has_Online_Delivery_Binary",
                "Price_range_num"] + num_feats

# add cuisine binary features
cuisine_cols = [c for c in df.columns if c.startswith("cuisine_")]
feature_cols += cuisine_cols

# ensure no duplicate
feature_cols = list(dict.fromkeys(feature_cols))

print("Using features:", feature_cols)

# Prepare X, y
X = df[feature_cols].astype(float)
y = pd.to_numeric(df["Aggregate rating"], errors='coerce').fillna(0).astype(float)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Standardize numeric features (not strictly necessary for tree-based models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models to try
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42, max_depth=8),
    "RandomForest": RandomForestRegressor(random_state=42, n_estimators=100, max_depth=12, n_jobs=-1)
}

results = {}
for name, model in models.items():
    if name == "LinearRegression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))   # ✅ fixed line
    r2 = r2_score(y_test, preds)
    results[name] = {"rmse": rmse, "r2": r2}
    print(f"\nModel: {name}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")


# Save predictions (optional)
pred_df = pd.DataFrame({
    "Actual": y_test,
    "LR_Pred": models["LinearRegression"].predict(X_test_scaled),
    "DT_Pred": models["DecisionTree"].predict(X_test),
    "RF_Pred": models["RandomForest"].predict(X_test)
}, index=y_test.index)
pred_df.to_csv("level3_predictions_sample.csv", index=False)
print("\nPredictions sample saved to level3_predictions_sample.csv")


# Task 2: Customer Preference Analysis

print("\n--- Customer Preference Analysis ---")

# Relationship between cuisine and rating: average rating per cuisine
if "Cuisines" in df.columns:
    # explode cuisines
    exploded = df[["Cuisines","Aggregate rating","Votes"]].copy()
    exploded["Cuisines"] = exploded["Cuisines"].astype(str).str.split(",", expand=False)
    exploded = exploded.explode("Cuisines")
    exploded["Cuisines"] = exploded["Cuisines"].str.strip()
    avg_rating_by_cuisine = exploded.groupby("Cuisines")["Aggregate rating"].mean().sort_values(ascending=False)
    count_by_cuisine = exploded["Cuisines"].value_counts()
    votes_by_cuisine = exploded.groupby("Cuisines")["Votes"].sum().sort_values(ascending=False)

    # Show top 15 by count
    top15 = count_by_cuisine.head(15).index.tolist()
    print("\nTop 15 cuisines by restaurant count:")
    print(count_by_cuisine.head(15))

    print("\nTop 15 cuisines by total votes:")
    print(votes_by_cuisine.head(15))

    print("\nTop 15 cuisines by average rating (min 10 restaurants filter):")
    # apply a minimum count filter for reliability
    cuisine_counts = count_by_cuisine[count_by_cuisine >= 10].index
    reliable_avg = avg_rating_by_cuisine.loc[cuisine_counts].sort_values(ascending=False).head(15)
    print(reliable_avg)
else:
    print("Cuisines column not found. Skipping cuisine analysis.")

# Most popular cuisines among customers by votes (already printed top of votes_by_cuisine)

# Task 3: Visualizations

print("\n--- Visualizations ---")

# Rating distribution
plt.figure(figsize=(7,4))
plt.hist(df["Aggregate rating"].dropna(), bins=20)
plt.title("Distribution of Aggregate Rating")
plt.xlabel("Aggregate Rating")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Average rating for top cuisines (bar)
if "Cuisines" in df.columns:
    top_counts = count_by_cuisine.head(10).index
    top_avg = avg_rating_by_cuisine.loc[top_counts]
    plt.figure(figsize=(8,4))
    top_avg.plot(kind="bar")
    plt.title("Average Rating for Top 10 Cuisines")
    plt.xlabel("Cuisine")
    plt.ylabel("Average Rating")
    plt.tight_layout()
    plt.show()

# Feature importance-ish: for random forest, show feature importances if model exists
if "RandomForest" in models:
    rf = models["RandomForest"]
    importances = rf.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(15)
    print("\nRandom Forest feature importances (top 15):")
    print(feat_imp)
    plt.figure(figsize=(8,4))
    feat_imp.plot(kind="bar")
    plt.title("Top Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.show()

# ✅ Summary
print("\n✅ LEVEL 3 COMPLETE")
print("Model results:")
for m, res in results.items():
    print(f" {m}: RMSE={res['rmse']:.4f}, R2={res['r2']:.4f}")

print("\nCustomer preference outputs saved in memory (and printed above).")
print("Prediction CSV: level3_predictions_sample.csv")
