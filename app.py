import os
import streamlit as st
import pandas as pd
import tempfile
import uuid
from langchain.schema import HumanMessage, AIMessage
from data_processor import DataProcessor
from agent import DataScienceAgent

st.set_page_config(page_title="Data Science Assistant", layout="wide")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}
if "file_names" not in st.session_state:
    st.session_state.file_names = []
if "cohere_api_key" not in st.session_state:
    st.session_state.cohere_api_key = ""
if "cohere_base_url" not in st.session_state:
    st.session_state.cohere_base_url = "https://api.cohere.ai"

def display_analysis_in_chat(analysis, data_processor):
    """Display comprehensive analysis output in the chat interface."""
    st.markdown("## Analysis Results")
    
    # Create tabs for different sections
    tabs = st.tabs(["Executive Summary", "Dataset Overview", "Detailed Analysis", "Step-by-Step Guide", "Recommendations"])
    
    with tabs[0]:  # Executive Summary
        st.markdown(analysis["summary"])
    
    with tabs[1]:  # Dataset Overview
        for name, df in st.session_state.dataframes.items():
            st.subheader(f"Dataset: {name}")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            
            # Display column information
            st.markdown("### Column Information")
            col_info = pd.DataFrame({
                "Data Type": df.dtypes.astype(str),
                "Missing Values": df.isnull().sum(),
                "Missing %": (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info)
            
            # Show sample data
            st.markdown("### Sample Data")
            st.dataframe(df.head())
            
            # Show visualizations
            st.markdown("### Visualizations")
            
            # Missing values chart
            if df.isnull().sum().sum() > 0:
                st.subheader("Missing Values")
                missing_data = pd.DataFrame({
                    'Column': df.columns,
                    'Missing Values': df.isnull().sum().values,
                    'Missing %': (df.isnull().sum() / len(df) * 100).values
                }).sort_values('Missing %', ascending=False)
                
                st.bar_chart(missing_data.set_index('Column')['Missing %'])
            
            # Correlation heatmap for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
                st.subheader("Correlation Matrix")
                corr = df[numeric_cols].corr()
                st.pyplot(data_processor.plot_correlation_matrix(corr))
    
    with tabs[2]:  # Detailed Analysis
        st.markdown(analysis["details"])
    
    with tabs[3]:  # Step-by-Step Guide
        for i, step in enumerate(analysis["steps"]):
            st.markdown(f"### Step {i+1}: {step['title']}")
            st.markdown(step['description'])
            
            # Display any additional information for the step
            if "note" in step and step["note"]:
                st.info(step["note"])
    
    with tabs[4]:  # Recommendations
        st.markdown(analysis["recommendations"])
        
        # Create a checklist
        st.subheader("Implementation Checklist")
        for i, item in enumerate(analysis["checklist"]):
            st.checkbox(item, key=f"check_{i}")

def main():
    # Sidebar for configuration and file upload
    with st.sidebar:
        st.title("Data Science Assistant")
        st.subheader("Configuration")
        
        # API Key input
        api_key = st.text_input("Cohere API Key", value=st.session_state.cohere_api_key, type="password")
        if api_key != st.session_state.cohere_api_key:
            st.session_state.cohere_api_key = api_key
        
        # Base URL input
        base_url = st.text_input("Cohere Base URL", value=st.session_state.cohere_base_url)
        if base_url != st.session_state.cohere_base_url:
            st.session_state.cohere_base_url = base_url
        
        # File uploader
        st.subheader("Upload Data")
        uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)
        
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in st.session_state.file_names:
                    # Save the file temporarily
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(file.getvalue())
                        temp_path = tmp_file.name
                    
                    # Load the dataframe
                    try:
                        df = pd.read_csv(temp_path)
                        st.session_state.dataframes[file.name] = df
                        st.session_state.file_names.append(file.name)
                        st.success(f"File {file.name} loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading {file.name}: {str(e)}")
                    finally:
                        os.unlink(temp_path)
        
        # Show loaded datasets
        if st.session_state.file_names:
            st.subheader("Loaded Datasets")
            for name in st.session_state.file_names:
                st.write(f"• {name}")
            
            # Clear data button
            if st.button("Clear All Data"):
                st.session_state.dataframes = {}
                st.session_state.file_names = []
                st.session_state.messages = []
                st.experimental_rerun()
    
    # Main chat interface
    st.title("Data Science Analysis Assistant")
    
    # Display API key warning if not provided
    if not st.session_state.cohere_api_key:
        st.warning("Please enter your Cohere API key in the sidebar.")
    
    # Display file upload warning if no files uploaded
    if not st.session_state.file_names:
        st.warning("Please upload at least one CSV file to begin analysis.")
    
    # Display chat messages
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                # Check if this is a detailed analysis message
                if hasattr(message, 'detailed_analysis') and message.detailed_analysis:
                    display_analysis_in_chat(message.detailed_analysis, message.data_processor)
                else:
                    # Regular message
                    st.write(message.content)
    
    # Chat input
    if st.session_state.cohere_api_key and st.session_state.file_names:
        if prompt := st.chat_input("Describe your analysis needs..."):
            # Add user message to chat
            st.session_state.messages.append(HumanMessage(content=prompt))
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Process with the agent
            with st.chat_message("assistant"):
                with st.spinner("Analyzing data..."):
                    # Initialize data processor and agent
                    data_processor = DataProcessor(st.session_state.dataframes)
                    agent = DataScienceAgent(
                        api_key=st.session_state.cohere_api_key,
                        base_url=st.session_state.cohere_base_url,
                        data_processor=data_processor
                    )
                    
                    # Get analysis from agent
                    analysis_response, detailed_analysis = agent.process_query_with_details(prompt)
                    
                    # Create a response message with detailed analysis
                    response_message = AIMessage(content=analysis_response)
                    response_message.detailed_analysis = detailed_analysis
                    response_message.data_processor = data_processor
                    
                    # Display response and add to messages
                    st.write(response_message.content)
                    display_analysis_in_chat(detailed_analysis, data_processor)
                    st.session_state.messages.append(response_message)

if __name__ == "__main__":
    main()