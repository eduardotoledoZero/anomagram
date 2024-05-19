# Anomaly Detection in Energy Consumption


Analytical model using historical energy consumption data from PowerElec's unregulated customers to detect anomalies in their consumption patterns, to conduct technical reviews on their internal facilities or the distribution network supplying them with energy, and implement technical measures to minimize technical and non-technical losses.

This code was developed in a Windows Machine with 7 cores and 32GB and using an virtual environment in VSCode 

# Tabla de Contenido

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Data Understanding](#data-understanding)
4. [Features Selection and Extraction](#features-selection-and-extraction)
5. [Clustering](#clustering)
6. [Anomalies Detection Models](#anomalies-detection-models)
7. [Model Selection](#model-selection)
8. [Integration Pipeline ](#integration-pipeline)

## Installation

To use this project, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using the following command:

   ```bash
   pip install -r requirements.txt

## Data Preparation 
[Notebook for this section](https://github.com/eduardotoledoZero/anomagram/blob/master/step_2%20eda.ipynb)<br>

During the initial preprocessing stage, an ETL process was executed to merge various CSV files containing historical data into a cohesive dataset named "consumo_datamart" in the 'bronze' state. This procedure is pivotal for streamlining analysis and conducting descriptive studies effectively. Furthermore, integration of economic sector details from a separate CSV file was conducted. To enhance clarity and uniformity in presenting economic sector information, text normalization utilizing the NLTK library was employed. This involved removing accents, extraneous words, and other non-essential elements, thus enhancing the visual coherence and quality of data associated with this attribute.

## Data Understanding
[Notebook for this section](https://github.com/eduardotoledoZero/anomagram/blob/master/step_1%20preproprecessing.ipynb)<br>
The database consists of 463,425 records, each with 7 distinct attributes characterizing energy consumption and customer information. These data are essential for understanding the consumption patterns of our users and the distribution of these patterns across different economic sectors.

The data catalog is as follow:

| Attribute          | Type     | Description                                                                                                             |
|--------------------|----------|-------------------------------------------------------------------------------------------------------------------------|
| Date               | Timestamp| Hourly record since 2021                                                                                                |
| Active_energy      | Float64  | Active energy is the part of electrical energy that performs work, such as lighting, heating, cooling, machinery operation, and other devices that consume electrical energy to perform their functions. For anomaly detection in this context, "Active_energy" can be an important indicator, as anomalies may manifest as unusual changes in active energy consumption. Measured in kilowatt-hours (kWh) |
| Reactive_energy    | Float64  | Form of electrical energy that does not perform useful work but is necessary to maintain voltage and current in electrical systems. "Reactive_energy" could be relevant if unusual changes in the amount of reactive energy consumed are observed. Measured in kilowatt-hours (kWARh) |
| Voltage_FA         | Float64  | Voltage in phase A of an electrical system. Phase voltage is the voltage found in a single phase of a three-phase electrical system. Measured in volts (V).                             |
| Voltage_FC         | Float64  | Voltage in frequency controlled measured in Volts                                                                       |
| CustomerId         | Int64    | Domain from 1 to 30 corresponding to 30 customers. Unique numerical identifier for each customer.                       |
| Economic_Sector    | Object   | Category describing the customer's economic sector.                                                                      |
### Data Quality Analysis:
**Completeness**
The integrity of the database is characterized by zero missing values in any of the attributes, indicating 100% completeness.

**Uniqueness and Redundancy**
No duplicate rows were found, reflecting the uniqueness of each record and ensuring there is no redundancy in the repository.

## Features Selection and Extraction
[Notebook for this section](https://github.com/eduardotoledoZero/anomagram/blob/master/step_3%20feature%20selection.ipynb)</br>
The Enrichment with new attributes is achieved through the following actions:

**Incorporation of Power Factor:**
The power factor is a metric that indicates how effectively energy is being used. A power factor closer to 1 suggests efficient energy usage, while a low value may indicate losses and possibly the need for power factor correction in facilities.

**Decomposition of Time Dimension (Date) and Cyclical Encoding of These Components**
Decomposing the date dimension into its components (month, day of the week, and hour) allows us to detect relevant temporal patterns for energy consumption. The utilization of cyclical encoding, where sine and cosine functions are used to transform these temporal components into formats that reflect the cyclical nature of time, creates advantages for the construction of machine learning models.


Correlation analysis and PCA (Principal Component Analysis) were conducted to assess the importance of attributes (consumption variables and time variables) in shaping the principal components. This information helps understand the significance of attributes in anomaly detection.


## Clustering

[Notebook for this section](https://github.com/eduardotoledoZero/anomagram/blob/master/step_4%20clustering_kmeans.ipynb)</br>

Observing a multimodal distribution in the histograms of Voltage_FA and Voltage_FC suggests the possibility of multiple subgroups within the same dataset, potentially reflecting different populations or consumption patterns. This phenomenon justifies the use of segmentation techniques to identify and differentiate these subpopulations. Given the considerable variability in consumption patterns across economic sectors, segmentation is a method to discover groupings with similar characteristics or behaviors. Initially, applying a K-Means segmentation method, focusing solely on energy consumption variables, is proposed, and we will assess the model's effectiveness by calculating the silhouette index, which measures the cohesion and separation of clusters.

Subsequently, time variables with their respective cyclic encoding are incorporated into the segmentation analysis to observe if this improves differentiation and grouping, comparing silhouette indices again. The method resulting in the highest silhouette index will be selected to segment our data, providing another perspective to explain consumption patterns and hence anomaly detection. Ultimately, it was found that the procedure with the highest index  selected the consumption variables Voltage_FC, Voltage_FA, Active_Energy, Reactive_energy, and Power_factor with 4 clusters

## Anomalies Detection Models

[Isolation Forest Notebook for this section](https://github.com/eduardotoledoZero/anomagram/blob/master/step_4.1%20anomalies_by_isolation.ipynb)</br>

[LOF Notebook for this section](https://github.com/eduardotoledoZero/anomagram/blob/master/step_4.1%20anomalies_by_lof.ipynb)</br>

[OneClassSVM Notebook for this section](https://github.com/eduardotoledoZero/anomagram/blob/master/step_4.1%20anomalies_by_ocsvm.ipynb)</br>

The approach adopted for anomaly detection, after segmenting the population into four clusters, consists of the following steps:

- Model Selection: Three unsupervised models—Isolation Forest, Local Outlier Factor, and OneClassSVM—were chosen due to their robustness and interpretability.
- Model Calibration: Each model was calibrated to determine its contamination factor using RandomizedSearchCV, preferred over GridSearchCV for its speed. Consumption and time variables were utilized. The Silhouette index was used as a metric to understand how these models differentiate between normal and anomalous data. The goal is to observe anomaly clusters and verify cohesion between outliers and inliers. 
- Outlier Detection and Anomaly Evaluation: Based on the established contamination factor, outliers were identified, and anomaly scores were calculated from various perspectives:
Global Unidimensional: A label (1 for inlier, -1 for outlier) and an anomaly score were generated for each observation using a model trained with all data. The contamination factor was set at 5%.
By Sector: A label (1 for inlier, -1 for outlier) and an anomaly score were determined for each observation using a model trained per sector. Parameter contamination calibration was applied for each sector to obtain the best silhouette index and calculate anomalies.
By Cluster: Similarly, a label and score were assigned to each observation using a model specific to each cluster. Parameter contamination calibration was applied for each cluster to obtain the best silhouette index and calculate anomalies.
Final Model Selection: To determine the most effective model, the labels resulting from the models trained by Cluster were evaluated, and the corresponding silhouette indices were calculated. Isolation Forest in the perspective global with a factor of contamination of 5%, presented the best silhouette index, and hence, it was selected as the definitive model. From this model, a 'gold' state datamart was generated, including all observations along with their anomaly scores. This datamart will be used in the application layer.

## Model Selection
[Notebook for this section](https://github.com/eduardotoledoZero/anomagram/blob/master/step_5.0%20anomalies_detection_benchmark.ipynb)</br>

In this section, different anomaly detection models are compared  based on their silhouette scores. The silhouette score is used as a metric to understand how well the models differentiate between normal and anomalous data. The model with the highest silhouette score is selected as the final model for anomaly detection.

<table>
<tr> 
  <th>Model</th>
  <th>Silhouette Score</th>
</tr>
<tr style="background-color:blue;">
  <td>Isolation Forest / Unique Global Model</td>
  <td>0.4317</td>
</tr>
<tr>
  <td>Isolation Forest / Models by Sector</td>
  <td>0.2743</td>
</tr>
<tr>
  <td>Isolation Forest / Models by Cluster</td>
  <td>0.3257</td>
</tr>
<tr>
  <td>Local Outlier Factor / Unique Global Model</td>
  <td>-0.021</td>
</tr>
<tr>
  <td>Local Outlier Factor / Models by Sector</td>
  <td>0.063</td>
</tr>
<tr>
  <td>Local Outlier Factor / Models by Cluster</td>
  <td>0.023</td>
</tr>
<tr>
  <td>OneClassSVM / Unique Global Model</td>
  <td>0.3926</td>
</tr>
<tr>
  <td>OneClassSVM / Models by Sector</td>
  <td>0.2527</td>
</tr>
<tr>
  <td>OneClassSVM / Models by Cluster</td>
  <td>0.2923</td>
</tr>
<tr>
  <td>HDBSCAN</td>
  <td>0.10099</td>
</tr>
</table>
The model with the best silhouette score is chosen, which in this case is the Isolation Forest with a score of 0.4317. Although this score is only moderately significant, it provides a valuable metric in the unsupervised context, allowing us to rely on it for effective anomaly detection.

## Integration Pipeline

[Notebook for this section](https://github.com/eduardotoledoZero/anomagram/blob/master/pipeline_script.py)</br>

This pipeline processes historical energy consumption data, enriches it with new features, segments it, and detects anomalies. It integrates preprocessing steps, feature extraction, clustering, and anomaly detection using advanced machine learning techniques.

Pipeline steps
1. Preprocessing
ETL Process: Consolidate multiple CSV files containing customer consumption data into a unified dataset.
Normalization: Standardize text data for economic sectors using NLTK.

2. Feature enrichment
Add power factor and decompose the date into cyclic components (month, day of the week, hour) to capture temporal patterns.

3. Clustering
K-Means Clustering: Segment data into clusters to identify distinct consumption patterns.

4. Anomaly Detection
Isolation Forest: Detect anomalies globally, by sector, and by cluster to ensure robust anomaly detection.