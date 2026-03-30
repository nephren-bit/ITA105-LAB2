import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\ITA105_Lab_2_Housing.csv")

# 3 cột numeric
numeric_cols = ['dien_tich', 'gia', 'so_phong']

# 1. Shape + missing
print("Shape:", df.shape)
print("\nMissing:\n", df.isnull().sum())


# 2. Thống kê mô tả
print("\nDescribe:\n", df[numeric_cols].describe())


# 3. Boxplot
df[numeric_cols].boxplot()
plt.title("Boxplot Before")
plt.show()


# 4. Scatter (diện tích vs giá)
plt.scatter(df['dien_tich'], df['gia'])
plt.xlabel("Dien tich")
plt.ylabel("Gia")
plt.title("Dien tich vs Gia")
plt.show()


# 5. IQR
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

outlier_iqr = ((df[numeric_cols] < (Q1 - 1.5*IQR)) |
               (df[numeric_cols] > (Q3 + 1.5*IQR)))

print("\nIQR outliers:\n", outlier_iqr.sum())


# 6. Z-score
z = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
outlier_z = abs(z) > 3

print("\nZ-score outliers:\n", outlier_z.sum())


# 7. So sánh
print("\nTotal IQR:", outlier_iqr.sum().sum())
print("Total Z-score:", outlier_z.sum().sum())


# 9. Xử lý (clip)
df_clean = df.copy()
df_clean[numeric_cols] = df[numeric_cols].clip(
    lower=Q1 - 1.5*IQR,
    upper=Q3 + 1.5*IQR,
    axis=1
)

# 10. Boxplot sau xử lý
df_clean[numeric_cols].boxplot()
plt.title("Boxplot After")
plt.show()
# ------------------------

df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\ITA105_Lab_2_Iot.csv")

# Convert timestamp + index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Numeric columns
numeric_cols = ['temperature','pressure','humidity']

# 1. Missing
print(df.isnull().sum())


# 2. Line plot từng sensor
for sensor in df['sensor_id'].unique():
    df[df['sensor_id']==sensor][numeric_cols].plot(title=f"Sensor {sensor}")
    plt.show()


# 3. Rolling outlier
print("\n=== Rolling Outliers ===")
for sensor in df['sensor_id'].unique():
    data = df[df['sensor_id']==sensor]

    rm = data[numeric_cols].rolling(10).mean()
    rs = data[numeric_cols].rolling(10).std()

    outlier = abs(data[numeric_cols] - rm) > 3*rs

    print(sensor)
    print(outlier.sum())


# 4. Z-score
print("\n=== Z-score Outliers ===")
for sensor in df['sensor_id'].unique():
    data = df[df['sensor_id']==sensor]

    z = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()
    outlier = abs(z) > 3

    print(sensor)
    print(outlier.sum())


# 5. Scatter
plt.scatter(df['temperature'], df['pressure'])
plt.show()

plt.scatter(df['pressure'], df['humidity'])
plt.show()


# 7. Xử lý (interpolate)
df_clean = df.copy()
df_clean[numeric_cols] = df_clean[numeric_cols].interpolate()

df_clean[numeric_cols].plot(title="After Cleaning")
plt.show()

# ---------------------------

df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\ITA105_Lab_2_Ecommerce.csv")

# 1. Missing + stats
print(df.isnull().sum())
print(df.describe())

# 2. Boxplot
df[['price','quantity','rating']].boxplot()
plt.show()

# 3. IQR + Z
numeric_cols = ['price', 'quantity', 'rating']

Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

outlier = ((df[numeric_cols] < (Q1 - 1.5*IQR)) |
           (df[numeric_cols] > (Q3 + 1.5*IQR)))

print(outlier.sum())

z = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
outlier_z = abs(z) > 3

print("Z-score outliers:")
print(outlier_z.sum())

# 4. Scatter
plt.scatter(df['price'], df['quantity'])
plt.xlabel("Price")
plt.ylabel("Quantity")
plt.show()

# 6. Xử lý
df = df[df['price'] > 0]
df = df[df['rating'].between(0,5)]

df['price'] = np.log1p(df['price'])  # log transform

# 7. Vẽ lại
df[['price','quantity','rating']].boxplot()
plt.show()

plt.scatter(df['price'], df['quantity'])
plt.show()

# ---------------------------


def iqr_outlier(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return (series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)

df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\ITA105_Lab_2_Housing.csv")

cols_housing = ['dien_tich', 'gia']

# Univariate
out_uni = df[cols_housing].apply(iqr_outlier)
df['out_uni_housing'] = out_uni.any(axis=1)

# Multivariate
z = (df[cols_housing] - df[cols_housing].mean()) / df[cols_housing].std()
df['out_multi_housing'] = (abs(z) > 3).any(axis=1)

# Plot
plt.scatter(df['dien_tich'], df['gia'],
            c=df['out_multi_housing'], cmap='coolwarm')
plt.xlabel('Diện tích')
plt.ylabel('Giá')
plt.title('Housing Outliers')
plt.show()

df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\ITA105_Lab_2_Iot.csv")

cols_iot = ['temperature', 'pressure']

# Univariate
out_uni = df[cols_iot].apply(iqr_outlier)
df['out_uni_iot'] = out_uni.any(axis=1)

# Multivariate
z = (df[cols_iot] - df[cols_iot].mean()) / df[cols_iot].std()
df['out_multi_iot'] = (abs(z) > 3).any(axis=1)

# Plot
plt.scatter(df['temperature'], df['pressure'],
            c=df['out_multi_iot'], cmap='coolwarm')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.title('IoT Outliers')
plt.show()

df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\ITA105_Lab_2_Ecommerce.csv")

cols_ecom = ['price', 'quantity', 'rating']

# Univariate
out_uni = df[cols_ecom].apply(iqr_outlier)
df['out_uni_ecom'] = out_uni.any(axis=1)

# Multivariate
z = (df[cols_ecom] - df[cols_ecom].mean()) / df[cols_ecom].std()
df['out_multi_ecom'] = (abs(z) > 3).any(axis=1)

# Scatter 2D
plt.scatter(df['price'], df['quantity'],
            c=df['out_multi_ecom'], cmap='coolwarm')
plt.xlabel('Price')
plt.ylabel('Quantity')
plt.title('E-commerce Outliers')
plt.show()

# Scatter Matrix
scatter_matrix(df[cols_ecom], figsize=(8,8))
plt.show()