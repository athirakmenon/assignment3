import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def read_renewable_data(file_path):
    """Read the renewable energy consumption data from a CSV file and \
        return a cleaned DataFrame."""
    renewable_data = pd.read_csv(file_path, skiprows=4)
    return renewable_data


def read_greenhouse_data(file_path):
    """Read greenhouse emission data from a CSV file and \
        return a cleaned DataFrame."""
    greenhouse_data = pd.read_csv(file_path, skiprows=4)
    return greenhouse_data


def preprocess_renewable_data(renewable_data, country_name):
    """Preprocess renewable energy consumption data,\
        filter relevant columns, handle missing values."""
    filtered_data = renewable_data[renewable_data['Country Name'] == country_name]
    selected_data = filtered_data[['Country Name'] +\
                                  [str(year) for year in range(1991, 2020)]]
    selected_data = selected_data.dropna()
    return selected_data


def preprocess_greenhouse_data(greenhouse_data, country_name):
    """Preprocess greenhouse data, filter relevant columns, handle missing values."""
    filtered_data = greenhouse_data[greenhouse_data['Country Name']\
                                    == country_name]
    selected_data = filtered_data[['Country Name'] +\
                                  [str(year) for year in range(1991, 2020)]]
    selected_data = selected_data.dropna()
    return selected_data


def transpose_renewable_data(renewable_data):
    """Transpose  and reset the index."""
    transposed_data = renewable_data.transpose().reset_index()
    transposed_data['Country Name'] = renewable_data['Country Name'].values[0]
    transposed_data.columns = ['Year', 'Renewable Energy Consumption',\
                               'Country Name'] + list(transposed_data.iloc[0, 3:])
    transposed_data = transposed_data[1:].reset_index(drop=True)
    return transposed_data


def transpose_greenhouse_data(data):
    """Transpose  and reset the index."""
    transposed_data = data.transpose().reset_index()
    transposed_data['Country Name'] = data['Country Name'].values[0]
    transposed_data.columns = ['Year', 'Greenhouse Gas Emission',\
                            'Country Name'] + list(transposed_data.iloc[0, 3:])
    transposed_data = transposed_data[1:].reset_index(drop=True)
    return transposed_data


def merge_transposed_data(transposed_renewable_data, transposed_greenhouse_data):
    
    merged_data = pd.merge(transposed_renewable_data,\
                           transposed_greenhouse_data, on=['Country Name', 'Year'])
    return merged_data


def print_silhouette_scores(data, max_clusters=5):
    """Print silhouette scores for different numbers of clusters."""
    features = data[['Renewable Energy Consumption', 'Greenhouse Gas Emission']]

    for n_clusters in range(2, max_clusters + 1):
        km = KMeans(n_clusters=n_clusters, random_state=42) 
        data['Cluster'] = km.fit_predict(features)

        # Calculate average silhouette score
        silhouette_avg = silhouette_samples(features, data['Cluster']).mean()
        print(f"For n_clusters = {n_clusters}, the average silhouette score is: {silhouette_avg}")
        

def apply_kmeans(data, n_clusters):
    """Apply K-Means clustering and scale relevant columns."""
    km = KMeans(n_clusters=n_clusters, random_state=42)  
   
    
    features = data[['Renewable Energy Consumption', 'Greenhouse Gas Emission']]
    data['Cluster'] = km.fit_predict(features)

  
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)
   
    
    scaled_centers = scaler.transform(km.cluster_centers_)
    print("Scaled Cluster Centers:")
    for i, center in enumerate(scaled_centers):
        print(f"Cluster {i+1}: {center}")

   
    data[['Renewable Energy Consumption', 'Greenhouse Gas Emission']] = scaled_data

    return data


def separate_clusters(data):
    clusters = [data[data['Cluster'] == i] for i in range(data['Cluster'].nunique())]
    return clusters


def plot_clusters(data, clusters):
    colors = ['darkgreen','teal','gold']
    plt.figure(figsize=(10,8))
    for i, cluster in enumerate(clusters):
        plt.scatter(cluster['Greenhouse Gas Emission'],\
                    cluster['Renewable Energy Consumption'],\
                        color=colors[i], label=f'Cluster {i+1}')
   
    
    centers = data.groupby('Cluster').mean()[['Greenhouse Gas Emission', \
                                              'Renewable Energy Consumption']]
    plt.scatter(centers['Greenhouse Gas Emission'],\
                centers['Renewable Energy Consumption'],\
                    marker='d', color='black', label='Cluster Centers')
   
   
    plt.xlabel('Greenhouse Gas Emission',fontsize=16)
    plt.ylabel('Renewable Energy Consumption',fontsize=16)
    plt.title('Total Greenhouse Gas Emission vs Renewable Energy of India',\
              fontsize=20)
    plt.legend()
    plt.show()
    
    
def plot_line_graph(data, variable_name, title, x_interval=5):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Year'], data[variable_name],color='teal')

    plt.xlabel('Year',fontsize=16)
    plt.ylabel(variable_name,fontsize=16)
    plt.title(title,fontsize=20)
    plt.xticks(data['Year'][::x_interval])
    plt.show()
    

def exponential(t, n0, g):
    return n0 * np.exp(g * (t - 1990))


def plot_prediction_graph(data, variable_name, title, x_interval=5):
    data['Year'] = pd.to_numeric(data['Year'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['Year'], data[variable_name], label=variable_name,color='black')

    param, covar = curve_fit(exponential, data['Year'], data[variable_name])
    year_forecast = np.linspace(data['Year'].min(), 2030, 100)
    param_std_dev = np.sqrt(np.diag(covar))
    lower_bound = exponential(year_forecast, *(param - param_std_dev))
    upper_bound = exponential(year_forecast, *(param + param_std_dev))

    plt.fill_between(year_forecast, lower_bound, upper_bound,\
                     color='yellow', alpha=0.3)
   
    year_forecast = np.linspace(data['Year'].min(), 2030, 100)
    forecast_exp = exponential(year_forecast, *param)
    plt.plot(year_forecast, forecast_exp, label="Prediction")

    plt.xlabel("Year",fontsize=16)
    plt.ylabel(variable_name,fontsize=16)
    plt.title(title,fontsize=20)
    plt.xticks(np.arange(data['Year'].min(), 2031, x_interval))
    plt.legend()
    plt.show()    



if __name__ == "__main__":
   
    np.random.seed(42)
  
    renewable_file_path = 'renew.csv'
    greenhouse_file_path = 'green.csv'  
    country_name = 'India'
   
    renewable_data = read_renewable_data(renewable_file_path)
    selected_data = preprocess_renewable_data(renewable_data, country_name)

    # Read and preprocess unemployment data
    greenhouse_data = read_greenhouse_data(greenhouse_file_path)
    greenhouse_selected_data = preprocess_greenhouse_data(greenhouse_data,\
                                                          country_name)
    transposed_renewable_data = transpose_renewable_data(selected_data)
    transposed_greenhouse_data = transpose_greenhouse_data(greenhouse_selected_data)
    merged_data = merge_transposed_data(transposed_renewable_data,transposed_greenhouse_data)

    n_clusters = 3
    clustered_data = apply_kmeans(merged_data, n_clusters)
    clusters = separate_clusters(clustered_data)
    plot_clusters(clustered_data, clusters)
    
    plot_line_graph(transposed_renewable_data, 'Renewable Energy Consumption',\
                    'Renewable Energy Consumption Over Years')
    plot_line_graph(transposed_greenhouse_data, 'Greenhouse Gas Emission',\
                    'Greenhouse Gas Emission Over Years', x_interval=5)

    plot_prediction_graph(transposed_renewable_data, 'Renewable Energy Consumption',\
                          'Renewable Energy Consumption Forecast')
    plot_prediction_graph(transposed_greenhouse_data, 'Greenhouse Gas Emission', \
                          'Greenhouse Gas Emission Forecast', x_interval=5)
