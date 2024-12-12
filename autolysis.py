import pandas as pd
#import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import requests

def load_data(filename):
    try:
        data = pd.read_csv(filename, encoding='ISO-8859-1')
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"Error loading file: Unable to decode the file with 'ISO-8859-1'.")
        sys.exit(1)

def analyze_data(data):
    analysis = {
        "shape": data.shape,
        "columns": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "summary_statistics": data.describe().to_dict()
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


url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

#AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDU1OTdAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.mGtFocaNamOEpoh3Y6WUB-xoAJJzW3EQntzLwbHUSXg"
def query_llm(prompt):
    headers = {
        'Authorization': f'Bearer {AIPROXY_TOKEN}',
        'Content-Type': 'application/json'
        }
    #prompt = f"Provide a detailed analysis based on the following data summary: {analysis}"
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        
        # Extract the content from the response
        content = response.json().get("choices", [])[0].get("message", {}).get("content", "")
        return content if content else "No content received from the model."
    
    except requests.exceptions.RequestException as e:
        # Handle any request-related exceptions
        return f"Request failed: {str(e)}"
    except (KeyError, IndexError):
        # Handle unexpected JSON structure
        return "Unexpected response structure received from the server."

# def visualize_data(data, output_prefix="chart"):
#     plt.figure(figsize=(8, 6))
#     num_df = data.select_dtypes(include=['number']) 
#     sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
#     plt.title("Correlation Matrix")
#     filename = f"{output_prefix}_correlation_matrix.png"
#     plt.savefig(filename)
#     plt.close()
#     return filename


def visualize_data(data, output_prefix="chart"):
    # Correlation matrix heatmap
    plt.figure(figsize=(10, 10))  # Set figsize when creating the figure
    num_df = data.select_dtypes(include=['number'])
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    filename_corr = f"{output_prefix}_correlation_matrix.png"
    plt.savefig(filename_corr, dpi=100, bbox_inches="tight")  
    plt.close()

    # Box plot for outlier detection
    plt.figure(figsize=(10, 10))  # Set figsize when creating the figure
    sns.boxplot(data=num_df)
    plt.title("Box Plot for Outlier Detection")
    filename_boxplot = f"{output_prefix}_boxplot.png"
    plt.savefig(filename_boxplot, dpi=100, bbox_inches="tight")  
    plt.close()

    # Histogram with KDE
    plt.figure(figsize=(10, 10))  # Set figsize when creating the figure
    sns.histplot(num_df.iloc[:, 0], kde=True)
    plt.title("Histogram with KDE")
    filename_histogram = f"{output_prefix}_histogram.png"
    plt.savefig(filename_histogram, dpi=100, bbox_inches="tight")  
    plt.close()

    return filename_corr, filename_boxplot, filename_histogram

def generate_story(analysis, chart_filenames):
    prompt = f"""
    The dataset contains the following summary statistics: {analysis}.
    Here are some visualizations: {chart_filenames}.
    Write a story about the dataset, its insights,  implications and conclusion.
    """
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
    #print(analysis)
    print("Running analysis...")
    #chart_files = [visualize_data(data)]
    chart_files = visualize_data(data)
    
    print("Generating story...")
    generate_story(analysis, chart_files)

    print("README.md and charts generated successfully.")

