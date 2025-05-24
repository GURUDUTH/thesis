import os
import time
from fpdf import FPDF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import tempfile
from typing import Dict, List, Any, Optional
from langchain.schema import HumanMessage, AIMessage


class Report(FPDF):
    """Custom PDF report class extending FPDF."""
    
    def __init__(self):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297
        self.title_text = 'Data Science Analysis Report'  # Store the title as a property
        
    def header(self):
        """Create header with logo and title."""
        # Add created time to header
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}', 0, 0, 'R')
        self.ln(20)
        
    def footer(self):
        """Create footer with page numbers."""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def title(self, txt):
        """Add report title."""
        self.title_text = txt  # Store the title text
        self.set_font('Arial', 'B', 24)
        self.cell(0, 20, txt, ln=True, align='C')
        self.ln(10)
        
    def chapter_title(self, title):
        """Add a chapter title."""
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)
        
    def section_title(self, title):
        """Add a section title."""
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, title, 0, 1, 'L')
        self.ln(2)
        
    def subsection_title(self, title):
        """Add a subsection title."""
        self.set_font('Arial', 'B', 11)
        self.cell(0, 6, title, 0, 1, 'L')
        self.ln(1)
        
    def body_text(self, text):
        """Add body text."""
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, text)
        self.ln(4)
        
    def step_text(self, step_num, text):
        """Add step text with step number."""
        self.set_font('Arial', 'B', 11)
        self.cell(12, 6, f"Step {step_num}:", 0, 0)
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, text)
        self.ln(2)
    
    def add_image(self, img_path=None, img_data=None, w=None, caption=None):
        """Add an image to the report with optional caption."""
        max_width = 180
        width = min(w if w else max_width, max_width)
        
        if img_path:
            self.image(img_path, x=(self.WIDTH - width)/2, w=width)
        elif img_data:
            # Create a temporary file to save the image data
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                tmp.write(img_data)
                tmp_path = tmp.name
            
            # Add the image to the PDF
            self.image(tmp_path, x=(self.WIDTH - width)/2, w=width)
            
            # Clean up the temporary file
            os.unlink(tmp_path)
        
        # Add caption if provided
        if caption:
            self.ln(2)
            self.set_font('Arial', 'I', 10)
            self.cell(0, 6, caption, 0, 1, 'C')
            
        self.ln(6)
    
    def add_table(self, data, header=None, col_widths=None):
        """Add a table to the report."""
        # Set default column widths if not provided
        if not col_widths:
            col_widths = [30] * len(data[0])
            
        # Calculate total width
        total_width = sum(col_widths)
            
        # Add header if provided
        if header:
            self.set_font('Arial', 'B', 10)
            for i, col in enumerate(header):
                self.cell(col_widths[i], 7, str(col), 1, 0, 'C')
            self.ln()
            
        # Add data rows
        self.set_font('Arial', '', 10)
        for row in data:
            for i, col in enumerate(row):
                self.cell(col_widths[i], 6, str(col)[:20], 1, 0, 'L')
            self.ln()
        
        self.ln(5)
    
    def add_checklist_item(self, text, checked=False):
        """Add a checklist item with checkbox."""
        self.set_font('ZapfDingbats', '', 10)
        if checked:
            self.cell(6, 6, "4", 0, 0)  # Checked box in ZapfDingbats
        else:
            self.cell(6, 6, "£", 0, 0)  # Empty box in ZapfDingbats
            
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, text)
        self.ln(2)


def generate_pdf_report(dataframes: Dict[str, pd.DataFrame], messages: List[Any]) -> str:
    """
    Generate a PDF report summarizing the data analysis and chat conversation.
    
    Args:
        dataframes: Dictionary of dataframes with filenames as keys
        messages: List of chat messages between user and assistant
        
    Returns:
        Path to the generated PDF file
    """
    # Create PDF object
    pdf = Report()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Add title
    pdf.title('Data Science Analysis Report')
    
    # Add introduction
    pdf.chapter_title('Introduction')
    pdf.body_text('This report summarizes the analysis of the uploaded CSV data and the insights provided by the Data Science Assistant.')
    
    # Dataset overview section
    pdf.add_page()
    pdf.chapter_title('Dataset Overview')
    
    for name, df in dataframes.items():
        pdf.section_title(f'File: {name}')
        
        # Basic dataset info
        pdf.body_text(f'Shape: {df.shape[0]} rows × {df.shape[1]} columns')
        pdf.body_text(f'Columns: {", ".join(df.columns.tolist())}')
        
        # Add data types
        pdf.body_text('Data Types:')
        dtypes_text = '\n'.join([f'- {col}: {str(dtype)}' for col, dtype in df.dtypes.items()])
        pdf.body_text(dtypes_text)
        
        # Add sample data as a table
        pdf.body_text('Sample Data (First 5 rows):')
        
        # Convert sample data to a list of lists
        header = df.columns.tolist()
        data = df.head().values.tolist()
        
        # Calculate column widths based on content
        col_widths = [min(30, max(10, len(col)*2)) for col in header]
        pdf.add_table(data, header, col_widths)
        
        # Add missing values information
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            pdf.body_text('Missing Values:')
            missing_text = '\n'.join([f'- {col}: {count} ({count/len(df)*100:.2f}%)' 
                                    for col, count in missing_values.items() if count > 0])
            pdf.body_text(missing_text)
        
        # Add visualizations
        try:
            # Missing values heatmap
            if df.isnull().sum().sum() > 0:
                plt.figure(figsize=(10, 6))
                sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
                plt.title('Missing Values Heatmap')
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_data = buf.getvalue()
                plt.close()
                
                pdf.body_text('Missing Values Heatmap:')
                pdf.add_image(img_data=img_data, w=160)
            
            # Correlation matrix for numeric columns
            if len(df.select_dtypes(include=['number']).columns) > 1:
                plt.figure(figsize=(10, 8))
                numeric_df = df.select_dtypes(include=['number'])
                corr = numeric_df.corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm')
                plt.title('Correlation Matrix')
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_data = buf.getvalue()
                plt.close()
                
                pdf.body_text('Correlation Matrix:')
                pdf.add_image(img_data=img_data, w=160)
        except Exception as e:
            pdf.body_text(f"Error generating visualizations: {str(e)}")
    
    # Chat conversation section
    pdf.add_page()
    pdf.chapter_title('Analysis Q&A')
    pdf.body_text('This section contains the questions and answers from the chat conversation.')
    
    for i, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            pdf.section_title(f"Q: {message.content}")
        elif isinstance(message, AIMessage) and i > 0:  # Ensure there's a question before this answer
            pdf.body_text(f"A: {message.content}")
    
    # Create the output file path (in the temp directory)
    output_path = tempfile.mktemp(suffix='.pdf')
    pdf.output(output_path, 'F')
    
    return output_path


def generate_enhanced_pdf_report(
    prompt: str,
    dataframes: Dict[str, pd.DataFrame],
    messages: List[Any],
    data_processor,
    agent
) -> str:
    """
    Generate an enhanced PDF report with detailed analysis, visualizations, and step-by-step instructions.
    
    Args:
        prompt: The user's query/prompt
        dataframes: Dictionary of dataframes with filenames as keys
        messages: List of chat messages between user and assistant
        data_processor: DataProcessor instance for analysis
        agent: DataScienceAgent instance for generating additional insights
        
    Returns:
        Path to the generated PDF file
    """
    # Create PDF object
    pdf = Report()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Add title and query
    pdf.title('Data Science Analysis Report')
    pdf.body_text(f"Analysis Query: {prompt}")
    pdf.ln(5)
    
    # Add executive summary
    pdf.chapter_title('Executive Summary')
    
    # Generate an executive summary using the agent
    exec_summary_prompt = f"""
    Create a brief executive summary (3-5 paragraphs) based on the following query and available data:
    Query: {prompt}
    
    Focus on:
    1. The key insights from the data
    2. Major findings related to the query
    3. High-level recommendations
    """
    exec_summary = agent._generate_response(exec_summary_prompt, {})
    pdf.body_text(exec_summary)
    
    # Dataset overview section
    pdf.add_page()
    pdf.chapter_title('Dataset Overview')
    
    # Get dataset summaries
    dataset_summary = data_processor.get_dataset_summary()
    
    for name, df in dataframes.items():
        pdf.section_title(f'Dataset: {name}')
        
        # Basic dataset info
        summary = dataset_summary.get(name, {})
        pdf.body_text(f"Rows: {summary.get('rows', len(df))}")
        pdf.body_text(f"Columns: {summary.get('columns', len(df.columns))}")
        
        # Column information
        pdf.subsection_title("Column Information")
        dtypes_text = '\n'.join([f"- {col}: {str(dtype)}" for col, dtype in df.dtypes.items()])
        pdf.body_text(dtypes_text)
        
        # Add sample data
        pdf.subsection_title("Sample Data (First 5 rows)")
        header = df.columns.tolist()
        data = df.head().values.tolist()
        col_widths = [min(30, max(10, len(col)*2)) for col in header]
        pdf.add_table(data, header, col_widths)
        
        # Add data quality information
        pdf.subsection_title("Data Quality")
        
        # Missing values
        missing_values = df.isnull().sum()
        missing_percent = missing_values / len(df) * 100
        
        if missing_values.sum() > 0:
            pdf.body_text("Missing Values:")
            missing_text = '\n'.join([f"- {col}: {count} ({percent:.2f}%)" 
                                    for col, (count, percent) in 
                                    zip(missing_values.index, zip(missing_values, missing_percent)) 
                                    if count > 0])
            pdf.body_text(missing_text)
        else:
            pdf.body_text("No missing values found in this dataset.")
        
        # Duplicate rows
        duplicates = df.duplicated().sum()
        pdf.body_text(f"Duplicate Rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
        
        # Add basic visualizations
        pdf.subsection_title("Data Visualizations")
        
        # Generate and add visualizations
        try:
            # 1. Missing values heatmap if there are missing values
            if missing_values.sum() > 0:
                plt.figure(figsize=(10, 6))
                sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
                plt.title('Missing Values Heatmap')
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_data = buf.getvalue()
                plt.close()
                
                pdf.add_image(img_data=img_data, w=160, caption="Missing Values Heatmap")
            
            # 2. Distribution of numerical columns (first 3)
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]
            if len(numeric_cols) > 0:
                plt.figure(figsize=(12, 4 * len(numeric_cols)))
                
                for i, col in enumerate(numeric_cols):
                    plt.subplot(len(numeric_cols), 1, i+1)
                    sns.histplot(df[col], kde=True)
                    plt.title(f'Distribution of {col}')
                    plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_data = buf.getvalue()
                plt.close()
                
                pdf.add_image(img_data=img_data, w=160, caption="Distribution of Key Numerical Variables")
            
            # 3. Correlation matrix for numeric columns
            if len(df.select_dtypes(include=[np.number]).columns) > 1:
                plt.figure(figsize=(10, 8))
                numeric_df = df.select_dtypes(include=[np.number])
                corr = numeric_df.corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
                plt.title('Correlation Matrix')
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_data = buf.getvalue()
                plt.close()
                
                pdf.add_image(img_data=img_data, w=160, caption="Correlation Matrix of Numerical Variables")
            
            # 4. Categorical data visualization (top 3)
            categorical_cols = df.select_dtypes(include=['object']).columns[:3]
            if len(categorical_cols) > 0:
                plt.figure(figsize=(12, 4 * len(categorical_cols)))
                
                for i, col in enumerate(categorical_cols):
                    plt.subplot(len(categorical_cols), 1, i+1)
                    value_counts = df[col].value_counts().head(10)  # Top 10 categories
                    sns.barplot(x=value_counts.index, y=value_counts.values)
                    plt.title(f'Top Categories in {col}')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_data = buf.getvalue()
                plt.close()
                
                pdf.add_image(img_data=img_data, w=160, caption="Distribution of Top Categories in Categorical Variables")
                
        except Exception as e:
            pdf.body_text(f"Error generating visualizations: {str(e)}")
    
    # Add detailed analysis section
    pdf.add_page()
    pdf.chapter_title('Detailed Analysis')
    
    # Generate detailed analysis based on the prompt
    analysis_prompt = f"""
    Provide a detailed analysis of the data based on the following query:
    Query: {prompt}
    
    Include:
    1. Key insights from the data related to the query
    2. Notable patterns or trends
    3. Potential challenges or issues in the data
    4. Opportunities for deeper analysis
    """
    detailed_analysis = agent._generate_response(analysis_prompt, data_processor.analyze_query(prompt))
    pdf.body_text(detailed_analysis)
    
    # Add step-by-step guide section
    pdf.add_page()
    pdf.chapter_title('Step-by-Step Guide')
    
    # Generate step-by-step guide based on the prompt
    steps_prompt = f"""
    Create a detailed step-by-step guide to address the following query:
    Query: {prompt}
    
    Format your response as a numbered list with 5-10 clearly defined steps.
    For each step:
    1. Provide a clear title for what needs to be done
    2. Give detailed instructions on how to perform the step
    3. Explain why this step is important
    
    Make the steps practical and actionable.
    """
    steps_response = agent._generate_response(steps_prompt, data_processor.analyze_query(prompt))
    
    # Parse steps (simplified parsing - assumes steps are numbered)
    steps = []
    current_step = ""
    
    for line in steps_response.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Try to detect new steps by looking for numbers at start of line
        if line[0].isdigit() and '. ' in line[:5]:
            if current_step:
                steps.append(current_step)
            current_step = line
        else:
            current_step += "\n" + line
    
    # Add the last step
    if current_step:
        steps.append(current_step)
    
    # Add steps to the report
    for i, step in enumerate(steps):
        step_num = i + 1
        step_text = step.split('. ', 1)[1] if '. ' in step else step
        pdf.step_text(step_num, step_text)
    
    # Add recommendations section
    pdf.add_page()
    pdf.chapter_title('Recommendations')
    
    # Generate recommendations based on the prompt
    recommendations_prompt = f"""
    Provide specific recommendations based on the following query and data analysis:
    Query: {prompt}
    
    Include:
    1. Actionable recommendations for next steps
    2. Potential areas for further data collection or analysis
    3. Key metrics to track
    4. Implementation suggestions
    
    Format your response as clear recommendation points.
    """
    recommendations = agent._generate_response(recommendations_prompt, data_processor.analyze_query(prompt))
    pdf.body_text(recommendations)
    
    # Create a checklist for implementation
    pdf.section_title("Implementation Checklist")
    
    # Extract checklist items from recommendations (simplified)
    checklist_items = []
    for line in recommendations.split('\n'):
        line = line.strip()
        if line and (line.startswith('- ') or line.startswith('* ')):
            checklist_items.append(line[2:])
    
    # If no items were easily extracted, create some generic ones
    if not checklist_items:
        checklist_items = [
            "Review the analysis findings",
            "Validate results with domain experts",
            "Implement recommended actions",
            "Monitor key metrics",
            "Schedule follow-up analysis"
        ]
    
    # Add checklist to PDF
    for item in checklist_items:
        pdf.add_checklist_item(item)
    
    # Create the output file path
    output_path = tempfile.mktemp(suffix='.pdf')
    pdf.output(output_path, 'F')
    
    return output_path