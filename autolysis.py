# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scikit-learn",
#   "ipykernel",
#   "scipy",
# ]
# ///

import os
import sys
import httpx
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Switch to a non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import base64
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency

url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")

def load_data(filename):
    """Load CSV data from a file."""
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
    """Perform basic and enhanced data analysis with statistical insights."""
    numeric_df = data.select_dtypes(include=['number'])
    analysis = {
        "shape": data.shape,
        "columns": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "summary_statistics": data.describe().to_dict(),
        'correlation': numeric_df.corr().to_dict(),
        "skewness": numeric_df.skew().to_dict(),
        "kurtosis": numeric_df.kurt().to_dict(),
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

    # PCA for Dimensionality Reduction
    if numeric_df.shape[1] > 1:  # Ensure enough columns for PCA
        pca = PCA(n_components=min(2, numeric_df.shape[1]))
        pca_result = pca.fit_transform(numeric_df.fillna(0))
        analysis['pca_variance_ratio'] = pca.explained_variance_ratio_.tolist()
    return analysis
    
def feature_importance(data):
    """Calculate feature importance using Random Forest (if applicable)."""
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le

    if 'target' in data.columns:
        y = data['target']
        X = data.drop('target', axis=1)
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X.fillna(0), y)
        return dict(zip(X.columns, clf.feature_importances_))
    return {}

# Outlier detection using Isolation Forest
def detect_anomalies(data):
    """Detect outliers in numeric columns using Isolation Forest."""
    numeric_df = data.select_dtypes(include=['number'])
    if numeric_df.empty:
        logging.warning("No numeric data available for outlier detection.")
        return {}
    
    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(numeric_df)
    outliers = clf.predict(numeric_df)
    outlier_counts = {col: sum(outliers == -1) for col in numeric_df.columns}
    return outlier_counts


def visualize_data(data, output_prefix="chart"):
    """Generate visualizations for data analysis."""
    chart_files = []
    num_df = data.select_dtypes(include=['number'])

    if num_df.empty:
        print("No numeric columns available for visualization.")
        return chart_files
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.xlabel("Features", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    filename_corr = f"{output_prefix}_correlation_matrix.png"
    plt.savefig(filename_corr, dpi=100, bbox_inches="tight")
    plt.close()
    chart_files.append(filename_corr)

    # Box Plot
    plt.figure(figsize=(10, 10))
    sns.boxplot(data=num_df)
    plt.title("Box Plot for Outlier Detection")
    plt.xticks(rotation=45, fontsize=10)
    plt.ylabel("Values", fontsize=12)
    filename_boxplot = f"{output_prefix}_boxplot.png"
    plt.savefig(filename_boxplot, dpi=100, bbox_inches="tight")
    plt.close()
    chart_files.append(filename_boxplot)

    # Histogram with KDE
    for col in num_df.columns[:2]:  # Limit to first 2 columns for visualization
            plt.figure(figsize=(10, 8))
            sns.histplot(num_df[col], kde=True, color='blue')
            plt.title(f"Histogram for {col}", fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            filename_histogram = f"{output_prefix}_histogram_{col}.png"
            plt.savefig(filename_histogram, dpi=100, bbox_inches="tight")
            plt.close()
            chart_files.append(filename_histogram)
        
    # Scatter Plot
    cols_to_plot = num_df.columns[:3]  # Select the first 3 columns
    for i, col1 in enumerate(num_df.columns):
        for col2 in num_df.columns[i+1:]:
            plt.figure(figsize=(6, 6))
            plt.scatter(num_df[col1], num_df[col2], alpha=0.5, c=np.random.rand(len(num_df)), cmap='viridis')
            plt.title(f"Scatter Plot: {col1} vs {col2}")
            plt.xlabel(col1)
            plt.ylabel(col2)
            filename_scatter = f"{output_prefix}_scatter_{col1}_vs_{col2}.png"
            plt.savefig(filename_scatter, dpi=100, bbox_inches="tight")
            plt.close()
            chart_files.append((f"Scatter Plot: {col1} vs {col2}", filename_scatter))

    return chart_files

def cramers_v(confusion_matrix):
    """Calculate Cramér's V statistic for categorical variables."""
    chi2, p, dof, expected = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

def calculate_cramers_v(data, col1, col2):
    """Calculate Cramér's V for a pair of categorical columns."""
    contingency_table = pd.crosstab(data[col1], data[col2])
    return cramers_v(contingency_table)

# Example usage: Calculate Cramér's V for all pairs of categorical columns
def calculate_cramers_v_for_all(data):
    categorical_cols = data.select_dtypes(include=['object']).columns
    cramers_v_results = {}

    # Loop through all pairs of categorical columns
    for i, col1 in enumerate(categorical_cols):
        for col2 in categorical_cols[i+1:]:
            cramers_v_results[(col1, col2)] = calculate_cramers_v(data, col1, col2)

    return cramers_v_results

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

def generate_story(analysis, chart_filenames,anomalies,feature_importance,results):
    prompt = "Based on the following analysis results, provide a comprehensive and detailed narrative:\n\n"
    
    # Add missing values information if any are detected
    if analysis['missing_values']:
        prompt += f"**Missing values detected in**: {analysis['missing_values']}\n"
    
    # Add anomalies information if any are detected
    if anomalies:
        prompt += f"**Anomalies detected in**: {anomalies}\n"
    # Add feature importance information if there are any
    if feature_importance:
        prompt += f"**Feature_importance in the dataset is**: {feature_importance,}\n"
    prompt += (
    f"**Column Names & Types:** {analysis['columns']}\n\n"
    
    f"**Summary Statistics:** {analysis['summary_statistics']}\n\n"
    
    
    f"**Outliers and Anomalies:** {analysis['outliers']}\n\n"
    
    f"**Correlation Analysis Results:** {analysis['correlation']}\n\n"
    
    f"**Visulation text** {chart_filenames}\n\n"

    f"**Anomalies count** {anomalies}\n\n"
    
    "In your analysis, please focus on the following:\n"
    "- Identify and describe any **trends** or **patterns** within the dataset. What variables have the strongest relationships with each other?"
    "- Discuss any **outliers** or **anomalies** that stand out, especially those that might need further investigation."
    "- Analyze the **missing values** and suggest possible imputation strategies or next steps for handling missing data."
    "- Highlight any **correlations** that might provide actionable insights, especially with respect to the success or failure of the campaign.\n"
    "-Provide **insights gained ** and **implications**"
    "- Finally, propose potential **recommendations** for improving the dataset strategy based on the insights you uncover. and provide **conclusion**"
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
    anomalies = detect_anomalies(data)

    chart_files = visualize_data(data)

    feature_importance=feature_importance(data)    
    # Calculate Cramér's V for all categorical pairs
    results = calculate_cramers_v_for_all(data)
    
    print("Generating story...")
    generate_story(analysis,  chart_files,anomalies,feature_importance,results)

    print("README.md and charts generated successfully.")
