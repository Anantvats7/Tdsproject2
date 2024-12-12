# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scikit-learn"
# ]
# ///

import os
import sys
import httpx
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")

def load_data(filename):
    try:
        data = pd.read_csv(filename, encoding='ISO-8859-1')
        return data
    except UnicodeDecodeError:
        print("Error loading file: Unable to decode the file with 'ISO-8859-1'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def analyze_data(data):
    numeric_df = data.select_dtypes(include=['number'])
    analysis = {
        "shape": data.shape,
        "columns": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "summary_statistics": data.describe().to_dict(),
        'correlation': numeric_df.corr().to_dict(), 
    }
    num_columns = data.select_dtypes(include=[np.number]).columns
    outliers = {}
    for column in num_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[column] = data[(data[column] < lower_bound) | (data[column] > upper_bound)].shape[0]
    
    analysis["outliers"] = outliers
    return analysis




def advanced_analysis(data):
    """
    Advanced analysis methods such as clustering and regression
    """

    # Impute missing values before clustering
    imputer = SimpleImputer(strategy='most_frequent')  
    data_imputed = imputer.fit_transform(data.select_dtypes(include=['number']))
    data_imputed = pd.DataFrame(data_imputed, columns=data.select_dtypes(include=['number']).columns)
    
    # Clustering (KMeans)
    kmeans = KMeans(n_clusters=3, random_state=42)
    data_imputed['Cluster'] = kmeans.fit_predict(data_imputed)
    
    # Regression Analysis
    # For simplicity, we'll predict a target variable (assuming 'target' exists in data)
    if 'target' in data.columns:
        X = data.drop(columns=['target'])
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        return {
            "Clustering": data_imputed['Cluster'].value_counts().to_dict(),
            "Regression_MSE": mse
        }
    else:
        return {"Clustering": data_imputed['Cluster'].value_counts().to_dict()}





def visualize_data(data, output_prefix="chart"):
    plt.figure(figsize=(10, 10))
    num_df = data.select_dtypes(include=['number'])
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    filename_corr = f"{output_prefix}_correlation_matrix.png"
    plt.savefig(filename_corr, dpi=100, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 10))
    sns.boxplot(data=num_df)
    plt.title("Box Plot for Outlier Detection")
    filename_boxplot = f"{output_prefix}_boxplot.png"
    plt.savefig(filename_boxplot, dpi=100, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 10))
    sns.histplot(num_df.iloc[:, 0], kde=True)
    plt.title("Histogram with KDE")
    filename_histogram = f"{output_prefix}_histogram.png"
    plt.savefig(filename_histogram, dpi=100, bbox_inches="tight")
    plt.close()

    return filename_corr, filename_boxplot, filename_histogram

def query_llm(prompt):
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = httpx.post(url, headers=headers, json=data, timeout=30)  # Increased timeout
        response.raise_for_status()
        content = response.json().get("choices", [])[0].get("message", {}).get("content", "")
        return content if content else "No content received from the model."
    except httpx.TimeoutException as e:
        return f"Request timed out: {str(e)}"
    except httpx.RequestError as e:
        return f"Request failed: {str(e)}"
    except (KeyError, IndexError):
        return "Unexpected response structure received from the server."

def generate_story(analysis, advanced_results, chart_filenames):
    prompt = (
    f"Based on the following analysis results, provide a comprehensive and detailed narrative:\n\n"
    
    f"**Column Names & Types:** {analysis['columns']}\n\n"
    
    f"**Summary Statistics:** {analysis['summary_statistics']}\n\n"
    
    f"**Missing Values:** {analysis['missing_values']}\n\n"
    
    f"**Outliers and Anomalies:** {analysis['outliers']}\n\n"
    
    f"**Advanced Analysis Results:** {advanced_results}\n\n"
    
    f"**Visulation ** {chart_filenames}\n\n"
    
    "In your analysis, please focus on the following:\n"
    "- Identify and describe any **trends** or **patterns** within the dataset. What variables have the strongest relationships with each other?"
    "- Discuss any **outliers** or **anomalies** that stand out, especially those that might need further investigation."
    "- Analyze the **missing values** and suggest possible imputation strategies or next steps for handling missing data."
    "- Highlight any **correlations** that might provide actionable insights, especially with respect to the success or failure of the campaign.\n"
    
    "- Finally, propose potential **recommendations** for improving the dataset strategy based on the insights you uncover."
    )
    story = query_llm(prompt)
    with open("README.md", "w") as f:
        f.write(story)
        for chart in chart_filenames:
            f.write(f"\n![Chart]({chart})\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]
    data = load_data(filename)
    analysis = analyze_data(data)
    
    print("Running analysis...")
    advanced_results = advanced_analysis(data)

    chart_files = visualize_data(data)
    
    print("Generating story...")
    generate_story(analysis, advanced_results, chart_files)

    print("README.md and charts generated successfully.")



#old codes for optional use if given code not gives expected results.
# def load_data(filename):
#     try:
#         data = pd.read_csv(filename, encoding='ISO-8859-1')
#         return data
#     except UnicodeDecodeError:
#         print("Error loading file: Unable to decode the file with 'ISO-8859-1'.")
#         sys.exit(1)
#     except Exception as e:
#         print(f"Error loading file: {e}")
#         sys.exit(1)

# def analyze_data(data):
#     numeric_df = data.select_dtypes(include=['number'])
#     analysis = {
#         "shape": data.shape,
#         "columns": data.dtypes.to_dict(),
#         "missing_values": data.isnull().sum().to_dict(),
#         "summary_statistics": data.describe().to_dict(),
#         'correlation': numeric_df.corr().to_dict() 
#         }
#     num_columns = data.select_dtypes(include=[np.number]).columns
#     outliers = {}
#     for column in num_columns:
#         Q1 = data[column].quantile(0.25)
#         Q3 = data[column].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#         outliers[column] = data[(data[column] < lower_bound) | (data[column] > upper_bound)].shape[0]
    
#     analysis["outliers"] = outliers
#     return analysis

# def query_llm(prompt):
#     headers = {
#         'Authorization': f'Bearer {AIPROXY_TOKEN}',
#         'Content-Type': 'application/json'
#     }
#     data = {
#         "model": "gpt-4o-mini",
#         "messages": [{"role": "user", "content": prompt}]
#     }
#     try:
#         response = httpx.post(url, headers=headers, json=data, timeout=30)  # Increased timeout
#         response.raise_for_status()
#         content = response.json().get("choices", [])[0].get("message", {}).get("content", "")
#         return content if content else "No content received from the model."
#     except httpx.TimeoutException as e:
#         return f"Request timed out: {str(e)}"
#     except httpx.RequestError as e:
#         return f"Request failed: {str(e)}"
#     except (KeyError, IndexError):
#         return "Unexpected response structure received from the server."


# def visualize_data(data, output_prefix="chart"):
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     # Correlation matrix heatmap
#     plt.figure(figsize=(10, 10))  # Set figsize when creating the figure
#     num_df = data.select_dtypes(include=['number'])
#     sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
#     plt.title("Correlation Matrix")
#     filename_corr = f"{output_prefix}_correlation_matrix.png"
#     plt.savefig(filename_corr, dpi=100, bbox_inches="tight")  
#     plt.close()

#     # Box plot for outlier detection
#     plt.figure(figsize=(10, 10))  # Set figsize when creating the figure
#     sns.boxplot(data=num_df)
#     plt.title("Box Plot for Outlier Detection")
#     filename_boxplot = f"{output_prefix}_boxplot.png"
#     plt.savefig(filename_boxplot, dpi=100, bbox_inches="tight")  
#     plt.close()

#     # Histogram with KDE
#     plt.figure(figsize=(10, 10))  # Set figsize when creating the figure
#     sns.histplot(num_df.iloc[:, 0], kde=True)
#     plt.title("Histogram with KDE")
#     filename_histogram = f"{output_prefix}_histogram.png"
#     plt.savefig(filename_histogram, dpi=100, bbox_inches="tight")  
#     plt.close()

#     return filename_corr, filename_boxplot, filename_histogram

# def generate_story(analysis, chart_filenames):
#     prompt =  (
#     f"Based on the following analysis results, provide a comprehensive and detailed narrative:\n\n"
    
#     f"**Column Names & Types:** {analysis['columns']}\n\n"
    
#     f"**Summary Statistics:** {analysis['summary_statistics']}\n\n"
    
#     f"**Missing Values:** {analysis['missing_values']}\n\n"
    
#     "In your analysis, please focus on the following:\n"
#     "- Identify and describe any **trends** or **patterns** within the dataset. What variables have the strongest relationships with each other?"
#     "- Discuss any **outliers** or **anomalies** that stand out, especially those that might need further investigation."
#     "- Analyze the **missing values** and suggest possible imputation strategies or next steps for handling missing data."
#     "- Highlight any **correlations** that might provide actionable insights, especially with respect to the success or failure of the campaign.\n"
#     "- Based on your findings, suggest **additional analyses** that could uncover further insights, such as clustering to identify customer segments or anomaly detection for identifying unusual patterns.\n"
#     "- Finally, propose potential **recommendations** for improving the dataset strategy based on the insights you uncover."
#     )
#     story = query_llm(prompt)
#     with open("README.md", "w") as f:
#         f.write(story)
#         for chart in chart_filenames:
#             f.write(f"\n![Chart]({chart})\n")

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python autolysis.py <dataset.csv>")
#         sys.exit(1)

#     filename = sys.argv[1]
#     data = load_data(filename)
#     analysis = analyze_data(data)

#     print("Running analysis...")

#     chart_files = visualize_data(data)
    
#     print("Generating story...")
#     generate_story(analysis, chart_files)


    print("README.md and charts generated successfully.")

