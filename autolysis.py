import pandas as pd
import os
import sys
import openai
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filename):
    try:
        data = pd.read_csv(filename)
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def analyze_data(data):
    analysis = {
        "shape": data.shape,
        "columns": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "summary_statistics": data.describe().to_dict()
    }
    return analysis

import requests
url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
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

def visualize_data(data, output_prefix="chart"):
    plt.figure(figsize=(8, 6))
    num_df = data.select_dtypes(include=['number']) 
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    filename = f"{output_prefix}_correlation_matrix.png"
    plt.savefig(filename)
    plt.close()
    return filename

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
    chart_files = [visualize_data(data)]

    print("Generating story...")
    generate_story(analysis, chart_files)

    print("README.md and charts generated successfully.")

