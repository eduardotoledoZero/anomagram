import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import numpy as np
import os
import unicodedata
import nltk
import string
from nltk.corpus import stopwords
import tkinter as tk
from tkinter import filedialog, messagebox
import joblib
import threading

# Function to set joblib backend
def set_joblib_backend():
    try:
        joblib.parallel_backend('multiprocessing')
    except ValueError as e:
        print(f"Warning: {e}. Proceeding without setting parallel backend.")

set_joblib_backend()

# Function to process customer data
def process_customer_data(sector_filepath, data_directory, output_filepath=None):
    def normalize_text(text):
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return '_'.join(filtered_words)

    nltk.download('stopwords')
    stop_words = set(stopwords.words('spanish'))
    sector_df = pd.read_csv(sector_filepath, encoding='ISO-8859-1')
    sector_df['sector_economico_normalizado'] = sector_df['Sector Economico'].map(lambda x: normalize_text(x).lower())
    sector_mapping = sector_df.set_index('Cliente')['sector_economico_normalizado'].to_dict()

    dataframes = []
    for filename in os.listdir(data_directory):
        if filename.startswith("DATOSCLIENTE") and filename.endswith('.csv'):
            filepath = os.path.join(data_directory, filename)
            customer_id = int(filename.replace('DATOSCLIENTE', '').replace('.csv', ''))
            df = pd.read_csv(filepath)
            df['ClienteId'] = customer_id
            dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df['Sector_Economico'] = combined_df['ClienteId'].map(lambda x: sector_mapping.get(f'Cliente {x} '))

    if output_filepath:
        combined_df.to_csv(output_filepath, index=False)
        print(f'Combined dataset saved to {output_filepath}')

    return combined_df

# Function to add new features to the dataframe
def add_new_features(df, output_filepath):
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    epsilon = 1e-10
    denominador = np.sqrt(np.square(df['Active_energy']) + np.square(df['Reactive_energy'])) + epsilon
    df['Factor_Potencia'] = df['Active_energy'] / denominador
    df['DiaSemana'] = df['Fecha'].dt.dayofweek
    df['Hora'] = df['Fecha'].dt.hour
    df['Mes'] = df['Fecha'].dt.month
    df['Dia_Sin'] = np.sin(df['DiaSemana'] * (2. * np.pi / 7))
    df['Dia_Cos'] = np.cos(df['DiaSemana'] * (2. * np.pi / 7))
    df['Hora_Sin'] = np.sin(df['Hora'] * (2. * np.pi / 24))
    df['Hora_Cos'] = np.cos(df['Hora'] * (2. * np.pi / 24))
    df['Mes_Sin'] = np.sin((df['Mes'] - 1) * (2. * np.pi / 12))
    df['Mes_Cos'] = np.cos((df['Mes'] - 1) * (2. * np.pi / 12))
    df.to_csv(output_filepath, index=False)
    print(f'Processed dataset saved to {output_filepath}')
    return df

# Function to detect anomalies using IsolationForest
def get_anomalies(df, contamination=0.05):
    df_copy = df.copy()
    iso_forest = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=1)
    iso_forest.fit(df_copy)
    preds = iso_forest.predict(df_copy)
    anomaly_score = iso_forest.decision_function(df_copy)
    df_copy['Anomaly'] = preds
    df_copy['Anomaly_Score'] = anomaly_score
    return df_copy

# Function for clustering
def clustering(df):
    features_to_scale = ['Active_energy', 'Reactive_energy', 'Voltaje_FA', 'Voltaje_FC', 'Factor_Potencia']
    X = df[features_to_scale]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_
    df['Cluster'] = labels
    return df['Cluster']

# Function to detect global anomalies
def detect_global_anomalies(df):
    features = ['Active_energy', 'Reactive_energy', 'Voltaje_FA', 'Voltaje_FC', 'Factor_Potencia']
    time_features = ['Dia_Sin', 'Dia_Cos', 'Hora_Sin', 'Hora_Cos', 'Mes_Sin', 'Mes_Cos']
    final_features = features + time_features
    df['Anomaly'] = pd.NA
    df['Anomaly_Score_Global'] = pd.NA
    data = df[final_features]
    data[final_features + ['Anomaly', 'Anomaly_Score_Global']] = get_anomalies(data)
    df.update(data[['Anomaly', 'Anomaly_Score_Global']])
    df.rename(columns={'Anomaly': 'Anomaly_Global'}, inplace=True)
    return df

# Function to detect sector-specific anomalies
def detect_sector_anomalies(df):
    contamination_df = pd.read_csv('../data/output/contamination_factor_sector_isoforest.csv')
    results = []
    sectors = sorted(list(df['Sector_Economico'].unique()))
    features = ['Active_energy', 'Reactive_energy', 'Voltaje_FA', 'Voltaje_FC', 'Factor_Potencia']
    time_features = ['Dia_Sin', 'Dia_Cos', 'Hora_Sin', 'Hora_Cos', 'Mes_Sin', 'Mes_Cos']
    final_features = features + time_features
    df['Anomaly_Score_Sector'] = pd.NA
    df['Anomaly'] = pd.NA
    for sector in sectors:
        data_orig = df[df['Sector_Economico'] == sector]
        data = data_orig[final_features]
        dictionary = eval(contamination_df[contamination_df['sector'] == sector]['Best Parameters'].iloc[0])
        data_orig[final_features + ['Anomaly', 'Anomaly_Score_Sector']] = get_anomalies(data, contamination=dictionary['contamination'])
        df.update(data_orig[['Anomaly', 'Anomaly_Score_Sector']])
    df.rename(columns={'Anomaly': 'Anomaly_Sector'}, inplace=True)
    return df

# Function to detect cluster-specific anomalies
def detect_cluster_anomalies(df):
    results = []
    clusters = sorted(list(df['Cluster'].unique()))
    features = ['Active_energy', 'Reactive_energy', 'Voltaje_FA', 'Voltaje_FC', 'Factor_Potencia']
    time_features = ['Dia_Sin', 'Dia_Cos', 'Hora_Sin', 'Hora_Cos', 'Mes_Sin', 'Mes_Cos']
    final_features = features + time_features
    df['Anomaly_Score_Cluster'] = pd.NA
    df['Anomaly'] = pd.NA
    for cluster in clusters:
        data_orig = df[df['Cluster'] == cluster]
        data = data_orig[final_features]
        data_orig[final_features + ['Anomaly', 'Anomaly_Score_Cluster']] = get_anomalies(data)
        df.update(data_orig[['Anomaly', 'Anomaly_Score_Cluster']])
    df.rename(columns={'Anomaly': 'Anomaly_Cluster'}, inplace=True)
    return df

# Function to select directory using tkinter
def select_directory(prompt):
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title=prompt)
    root.update()
    return folder_path

# Main function to run in a thread and avoid GUI blocking
def threaded_main(progress_label, start_button):
    try:
        progress_label.config(text="Starting script...")
        data_directory = select_directory("Select the data directory")
        sector_filepath = os.path.join(data_directory, 'sector_economico_clientes.csv')
        output_directory = select_directory("Select the output directory")

        # Ensure output directory exists
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        output_filepath_bronze = os.path.join(output_directory, 'consumo_datamart_bronze.csv')
        output_filepath_silver = os.path.join(output_directory, 'consumo_datamart_silver.csv')
        output_filepath_gold = os.path.join(output_directory, 'consumo_datamart_gold.csv')

        progress_label.config(text="Preprocessing STEP...")
        # Preprocessing STEP
        combined_df = process_customer_data(sector_filepath, data_directory, output_filepath_bronze)

        progress_label.config(text="New Features Enrichment STEP...")
        # New Features Enrichment STEP (Factor_potencia)
        combined_df = add_new_features(combined_df, output_filepath_silver)

        progress_label.config(text="Clustering STEP...")
        # Clustering STEP (4 clusters)
        combined_df['Cluster'] = clustering(combined_df)

        progress_label.config(text="Anomalies Detection Model Running STEP...")
        # Anomalies Detection with a global unique model STEP
        combined_df = detect_global_anomalies(combined_df)

        progress_label.config(text="Saving results...")
        # Save results
        combined_df.to_csv(output_filepath_gold, index=False)
        progress_label.config(text=f'Combined dataset saved to {output_filepath_gold}')
        print(combined_df.head())
        
        messagebox.showinfo("Success", "Data processing completed successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
    finally:
        start_button.config(state=tk.NORMAL)
        progress_label.config(text="")

# Function to create the main GUI window
def create_gui():
    root = tk.Tk()
    root.title("Data Processing Pipeline")

     # Maximize the window
    root.state('zoomed')
    
    tk.Label(root, text="Data Processing Pipeline", font=("Helvetica", 16)).pack(pady=40)
    progress_label = tk.Label(root, text="", font=("Helvetica", 12))
    progress_label.pack(pady=10)

    def on_start():
        start_button.config(state=tk.DISABLED)
        threading.Thread(target=threaded_main, args=(progress_label, start_button)).start()

    start_button = tk.Button(root, text="Start Processing", command=on_start, font=("Helvetica", 12))
    start_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
