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
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency

# Constants
URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")

# Utility Functions
def load_data(filename: str) -> pd.DataFrame:
    """Load CSV data from a file."""
    try:
        return pd.read_csv(filename, encoding='ISO-8859-1')
    except UnicodeDecodeError:
        print("Error: Unable to decode file with 'ISO-8859-1'.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def analyze_data(data: pd.DataFrame) -> dict:
    """Perform basic and enhanced data analysis."""
    numeric_df = data.select_dtypes(include=['number'])
    analysis = {
        "shape": data.shape,
        "columns": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "summary_statistics": data.describe().to_dict(),
        "correlation": numeric_df.corr().to_dict(),
        "skewness": numeric_df.skew().to_dict(),
        "kurtosis": numeric_df.kurt().to_dict(),
    }

    # Outlier detection using IQR
    outliers = {}
    for column in numeric_df.columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[column] = data[(data[column] < lower_bound) | (data[column] > upper_bound)].shape[0]
    analysis["outliers"] = outliers

    # PCA for dimensionality reduction
    if numeric_df.shape[1] > 1:
        pca = PCA(n_components=min(2, numeric_df.shape[1]))
        pca_result = pca.fit_transform(numeric_df.fillna(0))
        analysis['pca_variance_ratio'] = pca.explained_variance_ratio_.tolist()

    return analysis

def detect_anomalies(data: pd.DataFrame) -> dict:
    """Detect outliers using Isolation Forest."""
    numeric_df = data.select_dtypes(include=['number'])
    if numeric_df.empty:
        print("Warning: No numeric data for anomaly detection.")
        return {}

    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(numeric_df)
    outliers = clf.predict(numeric_df)
    return {col: (outliers == -1).sum() for col in numeric_df.columns}

def visualize_data(data: pd.DataFrame, output_prefix: str = "chart") -> list:
    """Generate visualizations for data analysis."""
    chart_files = []
    num_df = data.select_dtypes(include=['number'])

    if num_df.empty:
        print("No numeric columns available for visualization.")
        return chart_files

    # Correlation Heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Correlation Coefficient'})
    plt.title("Correlation Matrix")
    plt.xlabel("Features")
    plt.ylabel("Features")
    filename_corr = f"{output_prefix}_correlation_matrix.png"
    plt.savefig(filename_corr, dpi=100, bbox_inches="tight")
    plt.close()
    chart_files.append(filename_corr)


    # Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=num_df)
    plt.title("Box Plot for Outlier Detection")
    plt.xlabel("Features")
    plt.ylabel("Values")
    plt.xticks(rotation=45)

    # Adding annotations for outliers
    for i, column in enumerate(num_df.columns):
        Q1 = num_df[column].quantile(0.25)
        Q3 = num_df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = num_df[(num_df[column] < lower_bound) | (num_df[column] > upper_bound)]
        for outlier_idx in outliers.index:
            plt.text(i, num_df.loc[outlier_idx, column], f"{outlier_idx}", fontsize=8, color="red")

    filename_boxplot = f"{output_prefix}_boxplot.png"
    plt.savefig(filename_boxplot, dpi=100, bbox_inches="tight")
    plt.close()
    chart_files.append(filename_boxplot)

    # Histograms with KDE
    for col in num_df.columns[:2]:
        plt.figure(figsize=(8, 6))
        sns.histplot(num_df[col], kde=True, color='blue', label=f"KDE and Histogram for {col}")
        plt.title(f"Histogram for {col}")
        filename_histogram = f"{output_prefix}_histogram_{col}.png"
        plt.savefig(filename_histogram, dpi=100, bbox_inches="tight")
        plt.close()
        chart_files.append(filename_histogram)

    return chart_files

def cramers_v(confusion_matrix: np.ndarray) -> float:
    """Calculate Cramér's V statistic."""
    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum()
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))

def calculate_cramers_v(data: pd.DataFrame, col1: str, col2: str) -> float:
    """Calculate Cramér's V for two categorical columns."""
    contingency_table = pd.crosstab(data[col1], data[col2])
    return cramers_v(contingency_table.to_numpy())

def calculate_cramers_v_for_all(data: pd.DataFrame) -> dict:
    """Calculate Cramér's V for all pairs of categorical columns."""
    categorical_cols = data.select_dtypes(include=['object']).columns
    results = {}
    for i, col1 in enumerate(categorical_cols):
        for col2 in categorical_cols[i+1:]:
            results[(col1, col2)] = calculate_cramers_v(data, col1, col2)
    return results

def query_llm(prompt: str) -> str:
    """Query the LLM with a given prompt."""
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = httpx.post(URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json().get("choices", [])[0].get("message", {}).get("content", "No content received.")
    except httpx.RequestError as e:
        return f"Request failed: {e}"

def generate_story(analysis: dict, chart_files: list, anomalies: dict, results: dict):
    """Generate a narrative based on the analysis results."""
    prompt = (
        f"Based on the following analysis results:\n\n"
        f"- General Analysis: {analysis}\n"
        f"- Anomalies: {anomalies}\n"
        f"- Cramér's V Results: {results}\n\n"
        f"- Visualizations :{chart_files}\n\n"
        "Provide a detailed narrative including trends, anomalies, correlations, and recommendations.\n"
        "Ensure the narrative is structured into sections for readability: \n"
        "### Data description \n"
        "### Trends \n"
        "### Anomalies \n"
        "### Correlations \n"
        "### Recommendations \n"
        "### Insights \n"
        "### Implications \n"
        "### Conclusion \n"
        "Use Markdown formatting for emphasis."
    )
    story = query_llm(prompt)
    with open("README.md", "w") as f:
        f.write(story)
        for chart in chart_files:
            f.write(f"\n![Chart]({chart})\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    filename = sys.argv[1]
    data = load_data(filename)

    print("Running analysis...")
    analysis = analyze_data(data)
    anomalies = detect_anomalies(data)
    chart_files = visualize_data(data)
    results = calculate_cramers_v_for_all(data)

    print("Generating story...")
    generate_story(analysis, chart_files, anomalies, results)

    print("README.md and charts generated successfully.")

