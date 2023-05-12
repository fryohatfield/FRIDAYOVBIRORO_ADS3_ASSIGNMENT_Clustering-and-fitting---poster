# Importing Libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import cluster_tools as ct
import scipy.optimize as opt
import numpy as np
import errors as err

# csv files into a dataframe
df_agric_forest = pd.read_csv('agric_forest_data.csv', skiprows=4)
print(df_agric_forest)

df_agric_forest = df_agric_forest.drop(['Indicator Code', 'Country Code', 'Indicator Name', 'Unnamed: 66'], axis=1)
df_agric_forest.set_index('Country Name', drop=True, inplace=True)
df_agric_forest = df_agric_forest.dropna()
print(df_agric_forest)

# countries list to remove not required
countries_to_remove = ['East Asia & Pacific (excluding high income)',
                       'East Asia & Pacific',
                       'IBRD only',
                       'IDA & IBRD total',
                       'Low & middle income',
                       'Late-demographic dividend',
                       'Middle income',
                       'South Asia',
                       'East Asia & Pacific (IDA & IBRD countries)',
                       'South Asia (IDA & IBRD)', 'Upper middle income']

# Remove countries from the DataFrame
df_selected = df_agric_forest[~df_agric_forest.index.isin(countries_to_remove)]
# Print the resulting DataFrame
df_selected
df_agric_forest_selected = df_selected[["1960", "1970", "1980", "1990", "2000", "2010", "2020"]]
print(df_agric_forest_selected)
df_agric_forest_selected = df_agric_forest_selected.drop(["1960"], axis=1)
df_agric_forest_selected
#plotting a scatter matrix to show corellation
pd.plotting.scatter_matrix(df_agric_forest_selected, figsize=(9, 9), s=5, alpha=0.8)
plt.show()


# extract columns for fitting. 
# .copy() prevents changes in df_fit to affect df_fish.
df_fit_agricfor = df_agric_forest_selected[["1980", "2020"]].copy()

# normalise dataframe and inspect result
# normalisation is done only on the extract columns .copy() prevents
# changes in df_fit to affect df_fish. This make the plots with the 
# original measurements
df_fit_agricfor, df_min, df_max = ct.scaler(df_fit_agricfor)
print(df_fit_agricfor.describe())
print()

print("n   score")
# loop over trial numbers of clusters calculating the silhouette
for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit_agricfor)     

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_fit_agricfor, labels))

# Fit k-means with 4 clusters
kmeans = cluster.KMeans(n_clusters=4)
kmeans.fit(df_fit_agricfor)

# Add cluster label column to the original dataframe
df_agric_forest_selected["cluster_label"] = kmeans.labels_

# Group countries by cluster label
grouped = df_agric_forest_selected.groupby("cluster_label")

# Print countries in each cluster
for label, group in grouped:
    print("Cluster", label)
    print(group.index.tolist())
    print()

# Plot clusters with labels
plt.scatter(df_fit_agricfor["1980"], df_fit_agricfor["2020"], c=kmeans.labels_, cmap="Set1")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="k", marker="d", s=80)
plt.xlabel("1980")
plt.ylabel("2020")
plt.title("4 clusters")
plt.show()


#Reading csv file into the dataframe for curve fit and err range plotting 
df_pop = pd.read_csv("population_wb_data.csv", skiprows=4)
df_pop = df_pop.drop(['Indicator Code', 'Country Code', 'Indicator Name'], axis=1)
df_pop.set_index('Country Name', drop=True, inplace=True)
print(df_pop)

# comparison country from each cluster
countries = ['Singapore', 'Kenya', 'Ghana', 'Niger']
pop_countries = df_pop.loc[countries]
pop_countries = pop_countries.transpose()
pop_countries = pop_countries.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
print(pop_countries)

#polynomial module for fitting
from numpy.polynomial.polynomial import Polynomial

#plotting a subplot for the curve fit for the four selected countries

fig, axs = plt.subplots(2, 2, figsize=(12, 12))  # Create a 2x2 grid of subplots

# Iterate over each country and each subplot
for ax, country in zip(axs.flatten(), ['Singapore', 'Kenya', 'Ghana', 'Niger']):
    # Get the data for the current country
    x_data = pop_countries.index.values.astype(int)
    y_data = pop_countries[country].values

    # Fit the polynomial (degree 3 in this case) to the data
    p = Polynomial.fit(x_data, y_data, 3)

    # Generate x values for the fitted function
    x_fit = np.linspace(x_data.min(), x_data.max(), 1000)

    # Calculate the fitted y values
    y_fit = p(x_fit)

    # Plot the original data and the fitted function
    ax.plot(x_data, y_data, 'bo', label='Data')
    ax.plot(x_fit, y_fit, 'r-', label='Fit')
    ax.set_title(country)
    ax.set_xlabel('Year')
    ax.set_ylabel('Population')

    # Add a legend
    ax.legend()

plt.tight_layout()
plt.show()

# defining a function to calculate the upper and lower error ranges
def err_ranges(x, poly, coeffs, std_dev):
    """Calculate lower and upper error ranges for a polynomial fit.

    Parameters:
    x (np.ndarray): The x values to calculate error ranges for.
    poly (np.poly1d): The polynomial function.
    coeffs (np.ndarray): The coefficients of the polynomial.
    std_dev (np.ndarray): The standard deviations of the coefficients.

    Returns:
    np.ndarray, np.ndarray: The lower and upper error ranges.
    """
    # Calculate the derivative of the polynomial
    dp = poly.deriv()

    # Calculate the error at each point
    err = np.sqrt((dp(x) ** 2) * (std_dev ** 2).sum())

    # Calculate the lower and upper bounds
    lower = poly(x) - err
    upper = poly(x) + err

    return lower, upper
"""
plotting the error range and the predictions for the 4 countries from eac cluster and comparing them
"""

fig, axs = plt.subplots(2, 2, figsize=(12, 12))  # Create a 2x2 grid of subplots

# Iterate over each country and each subplot
for ax, country in zip(axs.flatten(), ['Singapore', 'Kenya', 'Ghana', 'Niger']):
    # Get the data for the current country
    x_data = pop_countries.index.values.astype(int)
    y_data = pop_countries[country].values

    # Fit the polynomial (degree 3 in this case) to the data
    p = Polynomial.fit(x_data, y_data, 3)

    # Generate x values for the fitted function and for the future prediction
    x_fit = np.linspace(x_data.min(), x_data.max() + 20, 1000)  # extending the range for 20 more years

    # Calculate the fitted y values
    y_fit = p(x_fit)

    # Plot the original data and the fitted function
    ax.plot(x_data, y_data, 'bo', label='Data')
    ax.plot(x_fit, y_fit, 'r-', label='Fit')
    ax.set_title(country)
    ax.set_xlabel('Year')
    ax.set_ylabel('Population')

    # Add a legend
    ax.legend()

plt.tight_layout()
plt.show()
