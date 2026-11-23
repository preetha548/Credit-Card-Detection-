
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("=" * 65)
print("       CREDIT CARD FRAUD DETECTION - DATA ANALYSIS")
print("=" * 65)

# 1. Load Dataset
df = pd.read_csv("creditcard.csv")

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# 2. Basic Dataset Information
print("\nFirst Five Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Description:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nClass Percentage:")
print(df["Class"].value_counts(normalize=True) * 100)

# 3. Create Output Folder
os.makedirs("CreditCard_Charts", exist_ok=True)

# 4. Chart Function
chart_number = 0

def save_chart(title):
    global chart_number
    chart_number += 1
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        f"CreditCard_Charts/Chart_{chart_number:02d}.png",
        dpi=150
    )
    plt.show()
    plt.close()

# Prepare Data
normal = df[df["Class"] == 0]
fraud = df[df["Class"] == 1]

# Sample for faster scatter plots
sample_df = df.sample(
    min(5000, len(df)),
    random_state=42
)

# ==========================================================
# 30 DATA VISUALIZATION CHARTS
# ==========================================================

# CHART 1 - Transaction Class Distribution
plt.figure(figsize=(8, 5))
df["Class"].value_counts().plot(
    kind="bar", rot=0
)
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")
save_chart("1. Transaction Class Distribution")

# CHART 2 - Normal vs Fraud Pie Chart
plt.figure(figsize=(7, 7))
plt.pie(
    df["Class"].value_counts(),
    labels=["Normal", "Fraud"],
    autopct="%1.2f%%",
    startangle=90
)
save_chart("2. Normal vs Fraud Transactions")

# CHART 3 - Transaction Amount Distribution
plt.figure(figsize=(9, 5))
plt.hist(df["Amount"], bins=50)
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
save_chart("3. Transaction Amount Distribution")

# CHART 4 - Fraud Transaction Amount
plt.figure(figsize=(9, 5))
plt.hist(fraud["Amount"], bins=30)
plt.xlabel("Amount")
plt.ylabel("Frequency")
save_chart("4. Fraud Transaction Amount Distribution")

# CHART 5 - Normal Transaction Amount
plt.figure(figsize=(9, 5))
plt.hist(normal["Amount"], bins=50)
plt.xlabel("Amount")
plt.ylabel("Frequency")
save_chart("5. Normal Transaction Amount Distribution")

# CHART 6 - Amount Comparison
plt.figure(figsize=(8, 5))
plt.boxplot(
    [normal["Amount"], fraud["Amount"]],
    tick_labels=["Normal", "Fraud"]
)
plt.ylabel("Amount")
save_chart("6. Transaction Amount Comparison")

# CHART 7 - Transaction Time Distribution
plt.figure(figsize=(9, 5))
plt.hist(df["Time"], bins=50)
plt.xlabel("Time")
plt.ylabel("Frequency")
save_chart("7. Transaction Time Distribution")

# CHART 8 - Fraud Transaction Time
plt.figure(figsize=(9, 5))
plt.hist(fraud["Time"], bins=30)
plt.xlabel("Time")
plt.ylabel("Fraud Count")
save_chart("8. Fraud Transaction Time Distribution")

# CHART 9 - Normal Transaction Time
plt.figure(figsize=(9, 5))
plt.hist(normal["Time"], bins=50)
plt.xlabel("Time")
plt.ylabel("Normal Count")
save_chart("9. Normal Transaction Time Distribution")

# CHART 10 - Time vs Amount
plt.figure(figsize=(9, 5))
plt.scatter(
    sample_df["Time"],
    sample_df["Amount"],
    alpha=0.3,
    s=8
)
plt.xlabel("Time")
plt.ylabel("Amount")
save_chart("10. Transaction Time vs Amount")

# CHART 11 - Normal Transaction Time vs Amount
plt.figure(figsize=(9, 5))
plt.scatter(
    normal["Time"].sample(
        min(3000, len(normal)),
        random_state=42
    ),
    normal["Amount"].sample(
        min(3000, len(normal)),
        random_state=42
    ),
    alpha=0.3,
    s=8
)
plt.xlabel("Time")
plt.ylabel("Amount")
save_chart("11. Normal Transaction Analysis")

# CHART 12 - Fraud Transaction Time vs Amount
plt.figure(figsize=(9, 5))
plt.scatter(
    fraud["Time"],
    fraud["Amount"],
    alpha=0.5,
    s=15
)
plt.xlabel("Time")
plt.ylabel("Amount")
save_chart("12. Fraud Transaction Analysis")

# CHART 13 - Feature V1 Distribution
plt.figure(figsize=(9, 5))
plt.hist(df["V1"], bins=50)
plt.xlabel("V1")
plt.ylabel("Frequency")
save_chart("13. V1 Feature Distribution")

# CHART 14 - Feature V2 Distribution
plt.figure(figsize=(9, 5))
plt.hist(df["V2"], bins=50)
plt.xlabel("V2")
plt.ylabel("Frequency")
save_chart("14. V2 Feature Distribution")

# CHART 15 - Feature V3 Distribution
plt.figure(figsize=(9, 5))
plt.hist(df["V3"], bins=50)
plt.xlabel("V3")
plt.ylabel("Frequency")
save_chart("15. V3 Feature Distribution")

# CHART 16 - Feature V4 Distribution
plt.figure(figsize=(9, 5))
plt.hist(df["V4"], bins=50)
plt.xlabel("V4")
plt.ylabel("Frequency")
save_chart("16. V4 Feature Distribution")

# CHART 17 - Feature V5 Distribution
plt.figure(figsize=(9, 5))
plt.hist(df["V5"], bins=50)
plt.xlabel("V5")
plt.ylabel("Frequency")
save_chart("17. V5 Feature Distribution")

# CHART 18 - Feature V6 Distribution
plt.figure(figsize=(9, 5))
plt.hist(df["V6"], bins=50)
plt.xlabel("V6")
plt.ylabel("Frequency")
save_chart("18. V6 Feature Distribution")

# CHART 19 - Feature V7 Distribution
plt.figure(figsize=(9, 5))
plt.hist(df["V7"], bins=50)
plt.xlabel("V7")
plt.ylabel("Frequency")
save_chart("19. V7 Feature Distribution")

# CHART 20 - Feature V8 Distribution
plt.figure(figsize=(9, 5))
plt.hist(df["V8"], bins=50)
plt.xlabel("V8")
plt.ylabel("Frequency")
save_chart("20. V8 Feature Distribution")

# CHART 21 - Feature V9 Distribution
plt.figure(figsize=(9, 5))
plt.hist(df["V9"], bins=50)
plt.xlabel("V9")
plt.ylabel("Frequency")
save_chart("21. V9 Feature Distribution")

# CHART 22 - Feature V10 Distribution
plt.figure(figsize=(9, 5))
plt.hist(df["V10"], bins=50)
plt.xlabel("V10")
plt.ylabel("Frequency")
save_chart("22. V10 Feature Distribution")

# CHART 23 - Feature Correlation Heatmap
plt.figure(figsize=(13, 10))
corr = df.corr(numeric_only=True)
plt.imshow(corr, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)
plt.yticks(
    range(len(corr.columns)),
    corr.columns
)
save_chart("23. Feature Correlation Heatmap")

# CHART 24 - Correlation with Class
plt.figure(figsize=(12, 6))
class_corr = corr["Class"].drop("Class")
class_corr.sort_values().plot(kind="bar")
plt.xlabel("Features")
plt.ylabel("Correlation")
save_chart("24. Feature Correlation with Fraud Class")

# CHART 25 - Amount by Transaction Class
plt.figure(figsize=(9, 5))
plt.boxplot(
    [normal["Amount"], fraud["Amount"]],
    tick_labels=["Normal", "Fraud"],
    showfliers=False
)
plt.ylabel("Transaction Amount")
save_chart("25. Amount Distribution by Class")

# CHART 26 - Feature V1 vs V2
plt.figure(figsize=(9, 5))
plt.scatter(
    sample_df["V1"],
    sample_df["V2"],
    c=sample_df["Class"],
    cmap="coolwarm",
    alpha=0.5,
    s=10
)
plt.xlabel("V1")
plt.ylabel("V2")
plt.colorbar(label="Class")
save_chart("26. V1 vs V2 Transaction Analysis")

# CHART 27 - Feature V3 vs V4
plt.figure(figsize=(9, 5))
plt.scatter(
    sample_df["V3"],
    sample_df["V4"],
    c=sample_df["Class"],
    cmap="coolwarm",
    alpha=0.5,
    s=10
)
plt.xlabel("V3")
plt.ylabel("V4")
plt.colorbar(label="Class")
save_chart("27. V3 vs V4 Transaction Analysis")

# CHART 28 - Feature V10 vs V11
plt.figure(figsize=(9, 5))
plt.scatter(
    sample_df["V10"],
    sample_df["V11"],
    c=sample_df["Class"],
    cmap="coolwarm",
    alpha=0.5,
    s=10
)
plt.xlabel("V10")
plt.ylabel("V11")
plt.colorbar(label="Class")
save_chart("28. V10 vs V11 Transaction Analysis")

# CHART 29 - Average Transaction Amount by Class
plt.figure(figsize=(8, 5))
df.groupby("Class")["Amount"].mean().plot(
    kind="bar",
    rot=0
)
plt.xlabel("Class (0=Normal, 1=Fraud)")
plt.ylabel("Average Amount")
save_chart("29. Average Transaction Amount by Class")

# CHART 30 - Transaction Count by Time Interval
df["Time_Interval"] = (
    df["Time"] // 3600
).astype(int)

hourly = df.groupby(
    "Time_Interval"
).size()

plt.figure(figsize=(10, 5))
hourly.plot(kind="line", marker="o")
plt.xlabel("Time Interval (Hours)")
plt.ylabel("Transaction Count")
save_chart("30. Transaction Count by Time Interval")

# ==========================================================
# FINAL ANALYSIS
# ==========================================================

print("\n" + "=" * 65)
print("             FINAL DATA ANALYSIS REPORT")
print("=" * 65)

print("\nTotal Transactions:", len(df))
print("Normal Transactions:", len(normal))
print("Fraud Transactions:", len(fraud))

print(
    "Fraud Percentage:",
    round((len(fraud) / len(df)) * 100, 4),
    "%"
)

print(
    "Normal Percentage:",
    round((len(normal) / len(df)) * 100, 4),
    "%"
)

print(
    "\nAverage Transaction Amount:",
    round(df["Amount"].mean(), 2)
)

print(
    "Average Fraud Amount:",
    round(fraud["Amount"].mean(), 2)
)

print(
    "Average Normal Amount:",
    round(normal["Amount"].mean(), 2)
)

print("\nHighest Transaction Amount:", df["Amount"].max())
print("Lowest Transaction Amount:", df["Amount"].min())

print("\nTotal Missing Values:", df.isnull().sum().sum())

print("\n30 Charts Generated Successfully!")
print("Chart Location: CreditCard_Charts")

print("\nCREDIT CARD FRAUD ANALYSIS COMPLETED!")
