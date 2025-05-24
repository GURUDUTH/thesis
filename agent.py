from typing import Dict, Any, List, TypedDict, Tuple
import cohere
from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage
import json
import re
from data_processor import DataProcessor


class AgentState(TypedDict):
    """Type definition for the agent state."""
    messages: List[Any]  # HumanMessage or AIMessage
    context: dict
    data_analysis: dict
    query: str


def create_data_analysis_prompt(context: dict) -> str:
    """
    Create a prompt with context about the data for the LLM to analyze.
    
    Args:
        context: Dictionary with data analysis results
        
    Returns:
        String prompt with formatted data analysis
    """
    prompt = "Here is the analysis of the uploaded CSV data:\n\n"
    
    # Basic dataset info
    if "basic_info" in context:
        prompt += "## Dataset Information\n"
        for file_name, info in context["basic_info"].items():
            prompt += f"File: {file_name}\n"
            prompt += f"- Shape: {info['shape'][0]} rows × {info['shape'][1]} columns\n"
            prompt += f"- Columns: {', '.join(info['columns'])}\n"
            
            # Add sample data (first few rows)
            prompt += "- Sample data (first 5 rows):\n"
            for col, values in info["head"].items():
                prompt += f"  - {col}: {list(values.values())[:5]}\n"
            
            # Add missing value information
            prompt += "- Missing values:\n"
            for col, missing in info["missing_percent"].items():
                if missing > 0:
                    prompt += f"  - {col}: {missing:.2f}% missing\n"
            
            prompt += "\n"
    
    # Data quality information
    if "data_quality" in context:
        prompt += "## Data Quality Analysis\n"
        for file_name, quality in context["data_quality"].items():
            prompt += f"File: {file_name}\n"
            
            # Duplicates
            prompt += f"- Duplicate rows: {quality['duplicate_rows']}\n"
            
            # Outliers
            if quality["potential_outliers"]:
                prompt += "- Potential outliers detected in:\n"
                for col, outlier_info in quality["potential_outliers"].items():
                    if outlier_info["potential_outliers_count"] > 0:
                        prompt += f"  - {col}: {outlier_info['potential_outliers_count']} potential outliers\n"
            
            prompt += "\n"
    
    # Feature engineering suggestions
    if "feature_engineering" in context:
        prompt += "## Feature Engineering Suggestions\n"
        for file_name, suggestions in context["feature_engineering"].items():
            prompt += f"File: {file_name}\n"
            
            for category, items in suggestions.items():
                if items:  # Only add non-empty categories
                    category_name = category.replace("_", " ").title()
                    prompt += f"- {category_name}:\n"
                    for suggestion in items:
                        prompt += f"  - {suggestion}\n"
            
            prompt += "\n"
    
    # Statistical summary
    if "statistics" in context:
        prompt += "## Statistical Summary\n"
        for file_name, stats in context["statistics"].items():
            if isinstance(stats, dict) and not stats.get("error"):
                prompt += f"File: {file_name}\n"
                prompt += "- Numerical column statistics:\n"
                
                # For each column in the statistics
                for col, col_stats in stats.items():
                    prompt += f"  - {col}:\n"
                    for stat_name, value in col_stats.items():
                        prompt += f"    - {stat_name}: {value}\n"
                
                prompt += "\n"
    
    prompt += "\nBased on this analysis, please answer the user's question."
    return prompt


class DataScienceAgent:
    def __init__(self, api_key: str, base_url: str, data_processor: DataProcessor):
        """
        Initialize the Data Science Agent with Cohere API directly.
        
        Args:
            api_key: Cohere API key
            base_url: Cohere API base URL (optional)
            data_processor: DataProcessor instance for analyzing CSV data
        """
        self.api_key = api_key
        self.base_url = base_url
        self.data_processor = data_processor
        
        # Set up the Cohere client
        if base_url and base_url != "https://api.cohere.ai":
            self.client = cohere.Client(api_key=api_key, api_url=base_url)
        else:
            self.client = cohere.Client(api_key=api_key)
    
    def _analyze_query(self, query: str) -> dict:
        """
        Analyze the user query and run appropriate data analyses.
        
        Args:
            query: User's question
            
        Returns:
            Data analysis results
        """
        # Analyze the query and get relevant data
        data_analysis = self.data_processor.analyze_query(query)
        return data_analysis
    
    def _generate_response(self, query: str, data_analysis: dict = None) -> str:
        """
        Generate a response using the Cohere API based on the query and optional data analysis.
        
        Args:
            query: User's question
            data_analysis: Optional results from data analysis (if not provided, uses a general response)
            
        Returns:
            Generated response text
        """
        system_message = """You are a helpful data science assistant.
Your job is to answer questions about data that has been uploaded as CSV files.
Analyze the provided data details and answer the user's question in a clear and informative way.
Include specific insights from the data and suggest next steps when appropriate.
If you're unsure about something or the data is insufficient, say so clearly."""
        
        # Create context prompt with data analysis if provided
        if data_analysis:
            context_prompt = create_data_analysis_prompt(data_analysis)
            full_message = f"Here's my question about the data: {query}\n\n{context_prompt}"
        else:
            # If no data analysis provided, just use the query
            full_message = query
        
        # Call the Cohere API
        response = self.client.chat(
            message=full_message,
            model="command",
            preamble=system_message,
            temperature=0.1,
            max_tokens=2000
        )
        
        return response.text
    
    def process_query(self, query: str) -> str:
        """
        Process a user query and return a response.
        
        Args:
            query: User's question about the data
            
        Returns:
            Generated response
        """
        # Analyze the query
        data_analysis = self._analyze_query(query)
        
        # Generate response
        response = self._generate_response(query, data_analysis)
        
        return response
        
    def process_query_with_details(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a user query and return a detailed structured response for display in the UI.
        
        Args:
            query: User's question about the data
            
        Returns:
            Tuple of (summary_response, detailed_analysis_dict)
        """
        # Analyze the query to get data insights
        data_analysis = self._analyze_query(query)
        
        # Generate main response
        main_response = self._generate_response(query, data_analysis)
        
        # Generate an executive summary
        exec_summary_prompt = f"""
        Create a brief executive summary (3-5 paragraphs) based on the following query and available data:
        Query: {query}
        
        Focus on:
        1. The key insights from the data
        2. Major findings related to the query
        3. High-level recommendations
        """
        summary = self._generate_response(exec_summary_prompt, data_analysis)
        
        # Generate detailed analysis
        detailed_analysis_prompt = f"""
        Provide a detailed analysis of the data based on the following query:
        Query: {query}
        
        Include:
        1. Key insights from the data related to the query
        2. Notable patterns or trends
        3. Potential challenges or issues in the data
        4. Opportunities for deeper analysis
        """
        details = self._generate_response(detailed_analysis_prompt, data_analysis)
        
        # Generate recommendations
        recommendations_prompt = f"""
        Provide specific recommendations based on the following query and data analysis:
        Query: {query}
        
        Include:
        1. Actionable recommendations for next steps
        2. Potential areas for further data collection or analysis
        3. Key metrics to track
        4. Implementation suggestions
        
        Format your response as clear recommendation points.
        """
        recommendations = self._generate_response(recommendations_prompt, data_analysis)
        
        # Generate step-by-step guide
        steps_prompt = f"""
        Create a detailed step-by-step guide to address the following query:
        Query: {query}
        
        Format your response as a numbered list with 5-10 clearly defined steps.
        For each step:
        1. Provide a clear title for what needs to be done
        2. Give detailed instructions on how to perform the step
        3. Explain why this step is important
        
        Make the steps practical and actionable.
        """
        steps_text = self._generate_response(steps_prompt, data_analysis)
        
        # Parse steps (simplified parsing - assumes steps are numbered)
        parsed_steps = self._parse_steps(steps_text)
        
        # Extract checklist items from recommendations
        checklist_items = self._extract_checklist_items(recommendations)
        
        # Create detailed analysis structure
        detailed_analysis = {
            "summary": summary,
            "details": details,
            "recommendations": recommendations,
            "steps": parsed_steps,
            "checklist": checklist_items
        }
        
        return main_response, detailed_analysis
    
    def _parse_steps(self, steps_text: str) -> List[Dict[str, str]]:
        """Parse a step-by-step text into structured steps."""
        steps = []
        current_step = {}
        lines = steps_text.split('\n')
        
        # Simple parsing - find steps that start with numbers
        step_pattern = re.compile(r'^\s*(\d+)\.\s+(.*)')
        
        for i, line in enumerate(lines):
            step_match = step_pattern.match(line)
            if step_match:
                # If we were building a previous step, save it first
                if current_step and 'title' in current_step:
                    steps.append(current_step)
                
                # Start a new step
                step_number = step_match.group(1)
                step_title = step_match.group(2)
                current_step = {
                    'title': step_title,
                    'description': '',
                    'note': ''
                }
            elif current_step and 'title' in current_step:
                # Add this line to the current step's description
                if "Note:" in line or "Important:" in line:
                    current_step['note'] += line + "\n"
                else:
                    current_step['description'] += line + "\n"
        
        # Add the last step if there is one
        if current_step and 'title' in current_step:
            steps.append(current_step)
        
        # If no steps were found, create a generic one from the whole text
        if not steps:
            steps.append({
                'title': 'Analysis Steps',
                'description': steps_text,
                'note': ''
            })
        
        return steps
    
    def _extract_checklist_items(self, recommendations: str) -> List[str]:
        """Extract checklist items from recommendations text."""
        items = []
        
        # Look for bullet points or numbered items
        for line in recommendations.split('\n'):
            line = line.strip()
            if line and (line.startswith('- ') or line.startswith('* ') or re.match(r'^\d+\.\s+', line)):
                # Remove the bullet/number prefix
                cleaned_item = re.sub(r'^[-*]\s+|^\d+\.\s+', '', line)
                items.append(cleaned_item)
        
        # If no items found, check for sentences
        if not items:
            sentences = re.split(r'(?<=[.!?])\s+', recommendations)
            items = [s for s in sentences if len(s) > 10 and s[-1] in ['.', '!', '?']]
        
        # If still no items, add a default one
        if not items:
            items = ["Review the analysis findings and implement recommendations"]
        
        return items
        
    def generate_step_by_step_analysis(self, prompt: str) -> str:
        """
        Generate step-by-step analysis for a specific query.
        
        Args:
            prompt: The user's query
            
        Returns:
            Generated step-by-step guide
        """
        # Get data analysis
        data_analysis = self._analyze_query(prompt)
        
        # Create a step-by-step guide prompt
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
        
        # Generate the step-by-step guide
        return self._generate_response(steps_prompt, data_analysis)
        
    def generate_recommendations(self, prompt: str) -> str:
        """
        Generate recommendations for a specific query.
        
        Args:
            prompt: The user's query
            
        Returns:
            Generated recommendations
        """
        # Get data analysis
        data_analysis = self._analyze_query(prompt)
        
        # Create a recommendations prompt
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
        
        # Generate the recommendations
        return self._generate_response(recommendations_prompt, data_analysis)