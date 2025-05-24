<p align="center">
  <img src="https://via.placeholder.com/800x200/4682B4/FFFFFF?text=Pre-Analysis+Agent" alt="Pre-Analysis Agent Banner">
</p>

# Pre-Analysis Agent

## Overview

The Pre-Analysis Agent is an intelligent application designed to give data scientists and analysts deeper insights into their CSV datasets. It provides step-by-step guidance from data cleaning to feature engineering to advanced analysis, helping users efficiently reach their desired solutions. Powered by Cohere's language models and built with Streamlit, this tool evaluates your data, identifies issues, suggests best practices, and recommends optimal processes tailored to your specific analytical needs.

Whether you're struggling with messy data, unsure about which features to engineer, or looking for the most appropriate analytical approach, the Pre-Analysis Agent provides clear and actionable guidance through every stage of the data science workflow. It serves as your AI assistant that helps you make informed decisions about how to approach your data before diving into complex modeling.

## Features

- **Interactive Data Analysis Interface**: Upload CSV files and ask questions in natural language
- **AI-Powered Data Insights**: Get sophisticated analyses of your data through natural language queries
- **Comprehensive Data Profiling**: Automatic detection of missing values, outliers, duplicates, and data quality issues
- **Smart Feature Engineering Suggestions**: Receive recommendations for potential feature transformations
- **Interactive Visualizations**: Auto-generated charts and plots based on your data's characteristics
- **Step-by-Step Analysis Guides**: Receive practical steps to address your data analysis needs
- **Recommendation Generation**: Get actionable insights and next steps to guide your analysis
- **Detailed Reporting**: Generate PDF reports of your analysis sessions for documentation and sharing

## Architecture

The project consists of the following main components:

- **app.py**: The Streamlit web application interface
- **agent.py**: The AI agent that processes user queries and generates responses
- **data_processor.py**: Handles data analysis, feature engineering and statistical calculations
- **pdf_generator.py**: Creates PDF reports containing analysis results

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Get a Cohere API key from [Cohere](https://cohere.ai)

## Usage

1. Run the Streamlit application:

```bash
streamlit run app.py
```

2. Open the provided URL in your web browser
3. Enter your Cohere API key in the sidebar
4. Upload one or more CSV files
5. Ask questions about your data in natural language

### Example Queries

- "What does my dataset consist of?"
- "Are there any missing values in my data?"
- "What feature engineering should I apply to this dataset?"
- "Can you visualize the correlation between variables?"
- "What are the key insights from this dataset?"
- "How can I clean this dataset?"
- "What statistical analysis can you provide on this data?"

## Key Capabilities

### Data Analysis

The assistant can perform various types of analysis:
- Basic dataset information (rows, columns, data types)
- Missing value detection and visualization
- Statistical summaries and correlations
- Outlier detection
- Duplicate record identification
- Data type checking and recommendations

### Visualizations

The tool automatically generates relevant visualizations:
- Correlation matrices
- Missing value heatmaps
- Distribution plots for numeric variables
- Bar charts for categorical variables
- And more based on your data's characteristics

### Feature Engineering Suggestions

Receive intelligent feature engineering suggestions:
- Categorical encoding recommendations
- Numerical transformation suggestions
- Datetime feature extraction ideas
- Text feature processing options
- Interaction feature possibilities

### Detailed Reporting

Generate comprehensive PDF reports containing:
- Executive summaries
- Dataset overviews
- Detailed statistical analysis
- Step-by-step guides
- Actionable recommendations
- Implementation checklists

## Technical Implementation

- **Language Model**: Uses Cohere's language models for natural language processing
- **Data Processing**: Pandas and NumPy for efficient data manipulation
- **Visualization**: Matplotlib and Seaborn for data visualization
- **Web Interface**: Streamlit for the interactive web application
- **PDF Generation**: FPDF for creating analysis reports

## Requirements

- Python 3.8+
- Streamlit 1.32.0+
- Cohere API Key
- Other dependencies listed in requirements.txt

## License

[Include your license information here]

## Contributing

[Include contribution guidelines if applicable]

## Authors

[Your name/organization]