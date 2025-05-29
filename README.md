<p align="center">
  <img src="https://raw.githubusercontent.com/user-attachments/assets/main/data_analysis_banner.png" alt="Pre-Analysis Agent Banner">
</p>

# Pre-Analysis Agent

> **Project Information:**  
> This project is submitted as a final semester project for the M.Sc. Data Science program at Chandigarh University.  
> **Author:** Guruduth Rao H  
> **Submission Date:** May 2025

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

## Methodology

The Pre-Analysis Agent follows a systematic approach to data analysis:

1. **Data Ingestion and Inspection**: CSV files are uploaded through the Streamlit interface and loaded into Pandas DataFrames for initial inspection.

2. **Query Understanding**: Natural language queries from users are processed to determine what type of analysis is needed.

3. **Automated Data Analysis**: The system performs relevant analyses based on the query:
   - For dataset overview queries, it extracts basic dataset information
   - For data quality queries, it detects missing values, duplicates, and outliers
   - For statistical queries, it computes descriptive statistics and correlations
   - For feature engineering queries, it suggests transformations based on data types

4. **AI-Powered Interpretation**: The Cohere language model processes the analysis results and user query to generate:
   - Explanations of findings in natural language
   - Step-by-step guides for data preparation and analysis
   - Recommendations for further actions

5. **Visualization Generation**: The system automatically creates relevant visualizations based on data characteristics and the user's query.

6. **Documentation**: Results are presented interactively and can be compiled into comprehensive PDF reports.

## System Implementation

### Architecture

The Pre-Analysis Agent is built using a modular architecture:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│                  │     │                  │     │                  │
│  Web Interface   │────▶│  Data Processor  │────▶│  AI Agent        │
│  (Streamlit)     │     │  (Pandas/NumPy)  │     │  (Cohere API)    │
│                  │     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         │                        │                        │
         │                        │                        │
         ▼                        ▼                        ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│                  │     │                  │     │                  │
│  Visualization   │     │  Feature         │     │  PDF Report      │
│  Engine          │     │  Engineering     │     │  Generator       │
│                  │     │  Suggestions     │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

### Module Details

The project consists of the following main components:

1. **app.py**: 
   - Streamlit web application interface
   - Manages user interactions, file uploads, and chat history
   - Renders visualizations and formatted analysis results
   - Coordinates between user queries and the backend agent

2. **agent.py**: 
   - Core AI agent that processes user queries
   - Integrates with the Cohere language model API
   - Transforms data analysis results into natural language insights
   - Generates structured responses (summaries, steps, recommendations)

3. **data_processor.py**: 
   - Handles data analysis using Pandas and NumPy
   - Performs statistical calculations and quality checks
   - Detects patterns and issues in the data
   - Suggests feature engineering transformations
   - Generates visualizations using Matplotlib and Seaborn

4. **pdf_generator.py**: 
   - Creates comprehensive PDF reports of analysis sessions
   - Formats data tables, visualizations, and analysis findings
   - Incorporates step-by-step guides and recommendations
   - Provides exportable documentation for sharing

### Process Logic

#### Data Analysis Flow

1. **User Input Processing**:
   ```
   User Query → Query Analysis → Data Analysis Selection → Results Compilation
   ```

2. **Data Processing Workflow**:
   ```
   Data Loading → Basic Information Extraction → Quality Check → 
   Statistical Analysis → Feature Engineering Suggestions → Visualization
   ```

3. **AI Response Generation**:
   ```
   Data Analysis Results → Context Preparation → LLM Processing → 
   Response Formatting → UI Presentation
   ```

### Hardware & Software Requirements

#### Hardware Requirements
- **Processor**: Multi-core CPU (4+ cores recommended)
- **Memory**: 8GB RAM minimum, 16GB or more recommended
- **Storage**: 1GB for application, additional space for datasets
- **Network**: Internet connection for Cohere API access

#### Software Requirements
- **Operating System**: Cross-platform (Windows, macOS, Linux)
- **Python**: Version 3.8 or higher
- **Key Libraries**:
  - Streamlit 1.32.0+ (Web interface)
  - Pandas 2.2.0+ (Data processing)
  - NumPy 1.26.3+ (Numerical operations)
  - Matplotlib 3.8.2+ and Seaborn 0.13.1+ (Visualization)
  - FPDF 1.7.2+ (Report generation)
  - Cohere API Client 4.37+ (Language model integration)

## Resources

### Data Analysis Capabilities

The assistant can analyze various aspects of your data:

1. **Basic Dataset Information**
   - Row and column counts
   - Data types and samples
   - Summary statistics

2. **Data Quality Assessment**
   - Missing value detection and quantification
   - Duplicate record identification
   - Outlier detection using statistical methods
   - Data type consistency checks

3. **Statistical Analysis**
   - Descriptive statistics (mean, median, std, etc.)
   - Distribution analysis
   - Correlation analysis
   - Aggregations and grouping insights

4. **Feature Engineering Guidance**
   - **Categorical Data**: 
     - One-hot encoding recommendations
     - Label encoding suggestions
     - Category consolidation opportunities
     - Handling high cardinality features
   - **Numerical Data**:
     - Scaling and normalization needs
     - Handling skewed distributions
     - Binning suggestions
     - Interaction feature potential
   - **DateTime Features**:
     - Temporal feature extraction (year, month, day, etc.)
     - Cyclical encoding options
     - Time-based aggregation possibilities
   - **Text Features**:
     - Basic NLP processing steps
     - Word count and character statistics
     - Tokenization recommendations
     - Vectorization approaches

5. **Visualization Options**
   - Distribution plots
   - Correlation matrices
   - Missing value heatmaps
   - Category bar plots
   - Box plots for outlier detection

## Limitations

The Pre-Analysis Agent has some limitations to be aware of:

1. **Data Size Constraints**
   - Limited to datasets that can fit in memory
   - Performance degrades with very large CSV files (>1GB)
   - No streaming data support

2. **File Format Restrictions**
   - Currently only supports CSV file format
   - No direct database connection capabilities
   - Limited handling of complex nested data structures

3. **Analysis Depth**
   - Provides pre-modeling insights rather than building actual models
   - No automated hyperparameter tuning
   - Limited support for advanced statistical tests

4. **Visualization Limitations**
   - Static visualizations only
   - Limited customization options
   - No interactive dashboards

5. **Language Model Constraints**
   - Requires Cohere API access and valid API key
   - Subject to Cohere's usage limits and policies
   - May occasionally generate inaccurate advice

6. **Security Considerations**
   - No built-in data encryption
   - Data is processed in memory, not persistently stored
   - No user authentication system

## System Maintenance & Evaluation

### Maintenance

**Regular Updates**:
- Update dependencies regularly using `pip install -r requirements.txt --upgrade`
- Check for Cohere API changes and version compatibility

**Error Handling**:
- The system includes basic error handling for:
  - API failures
  - Data loading issues
  - Visualization errors
- Check logs for error messages if issues arise

**Resource Monitoring**:
- Monitor memory usage with large datasets
- Check API usage against Cohere limits

### Evaluation

**Performance Metrics**:
- Response time for different query types
- Memory usage with varying dataset sizes
- Accuracy of data quality assessments

**User Experience**:
- Clarity of natural language explanations
- Relevance of recommendations
- Usefulness of step-by-step guides

**Quality Assurance**:
- Test with diverse dataset types and sizes
- Validate statistical calculations against known tools
- Review feature engineering suggestions for accuracy

## System Analysis & Design vs. User Requirements

### Primary User Requirements

1. **Data Understanding**: Users need to quickly grasp what their data contains
   - **Solution**: Automated profiling and summary generation

2. **Quality Assessment**: Users need to identify data quality issues before analysis
   - **Solution**: Comprehensive data quality checks for missing values, outliers, and inconsistencies

3. **Guidance for Next Steps**: Users need clear direction on how to proceed with their data
   - **Solution**: Step-by-step guides and actionable recommendations

4. **Feature Engineering Help**: Users need suggestions for transforming raw variables
   - **Solution**: Intelligent feature engineering recommendations based on data characteristics

5. **Documentation**: Users need shareable documentation of their analysis process
   - **Solution**: Comprehensive PDF report generation

### Design Decisions

1. **Natural Language Interface**: Using a chat-based interface allows users to express their analysis needs conversationally without requiring specific commands.

2. **PDF Report Generation**: Exportable reports enable users to share insights easily with stakeholders who don't have access to the system.

3. **Streamlit Framework**: Streamlit was chosen for its simplicity in developing data applications and its built-in support for rendering Pandas DataFrames and visualizations.

4. **Cohere Language Model**: Using an advanced LLM provides contextual understanding of user queries and generates human-like explanations.

5. **Modular Architecture**: The separation of concerns between web interface, data processing, AI agent, and report generation allows for easier maintenance and future enhancements.

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

Guruduth Rao H