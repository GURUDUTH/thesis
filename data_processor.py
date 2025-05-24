import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

class DataProcessor:
    def __init__(self, dataframes):
        """
        Initialize the DataProcessor with loaded dataframes.
        
        Args:
            dataframes: Dictionary of dataframes with filenames as keys
        """
        self.dataframes = dataframes
    
    def get_basic_info(self):
        """Get basic information about all loaded datasets."""
        info = {}
        for name, df in self.dataframes.items():
            info[name] = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "head": df.head().to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "missing_percent": (df.isnull().sum() / len(df) * 100).to_dict()
            }
        return info
    
    def get_dataset_summary(self):
        """Get a summary of all datasets."""
        summary = {}
        for name, df in self.dataframes.items():
            summary[name] = {
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "column_types": df.dtypes.value_counts().to_dict(),
                "missing_values_total": df.isnull().sum().sum(),
                "missing_values_percent": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
            }
        return summary
    
    def get_statistical_summary(self):
        """Get statistical summaries for numeric columns in all datasets."""
        stats = {}
        for name, df in self.dataframes.items():
            try:
                stats[name] = df.describe().to_dict()
            except:
                stats[name] = {"error": "Could not generate statistical summary"}
        return stats
    
    def check_data_quality(self):
        """Check data quality issues like missing values, outliers, etc."""
        quality_report = {}
        for name, df in self.dataframes.items():
            # Check missing values
            missing = df.isnull().sum()
            missing_percent = missing / len(df) * 100
            
            # Check potential outliers using IQR for numeric columns
            outliers = {}
            for col in df.select_dtypes(include=np.number).columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers[col] = {
                    "lower_bound": Q1 - 1.5 * IQR,
                    "upper_bound": Q3 + 1.5 * IQR,
                    "potential_outliers_count": len(df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)])
                }
            
            # Check duplicates
            duplicates = df.duplicated().sum()
            
            # Check data types
            dtypes = df.dtypes.astype(str).to_dict()
            
            # Compile quality report
            quality_report[name] = {
                "missing_values": missing.to_dict(),
                "missing_percent": missing_percent.to_dict(),
                "potential_outliers": outliers,
                "duplicate_rows": duplicates,
                "data_types": dtypes
            }
        
        return quality_report
    
    def suggest_feature_engineering(self):
        """Suggest potential feature engineering based on data types and values."""
        suggestions = {}
        
        for name, df in self.dataframes.items():
            file_suggestions = {
                "categorical_encoding": [],
                "numeric_transformations": [],
                "datetime_features": [],
                "text_features": [],
                "interactions": []
            }
            
            # Check categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                file_suggestions["categorical_encoding"] = [
                    f"Consider one-hot encoding for {col}" for col in categorical_cols
                ]
                
                # Check for columns that might benefit from label encoding
                for col in categorical_cols:
                    if df[col].nunique() < 10 and df[col].nunique() > 2:
                        file_suggestions["categorical_encoding"].append(
                            f"Consider ordinal/label encoding for {col}"
                        )
            
            # Check numerical columns
            numeric_cols = df.select_dtypes(include=np.number).columns
            if len(numeric_cols) > 0:
                # Check for skewed distributions
                for col in numeric_cols:
                    try:
                        skewness = df[col].skew()
                        if abs(skewness) > 1:
                            file_suggestions["numeric_transformations"].append(
                                f"Consider log or power transformation for {col} (skewness: {skewness:.2f})"
                            )
                    except:
                        pass
            
            # Check for potential datetime columns
            for col in categorical_cols:
                # Fix: Use pandas' vectorized & operator instead of Python's 'and'
                if df[col].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}').any():
                    file_suggestions["datetime_features"].append(
                        f"Convert {col} to datetime and extract features like year, month, day, weekday"
                    )
                # Check for date format with slashes (MM/DD/YYYY)
                elif (df[col].astype(str).str.contains('/') & 
                      (df[col].astype(str).str.count('/') == 2)).any():
                    file_suggestions["datetime_features"].append(
                        f"Convert {col} to datetime and extract features like year, month, day, weekday"
                    )
            
            # Check for potential text features
            for col in categorical_cols:
                if df[col].astype(str).str.len().mean() > 20:
                    file_suggestions["text_features"].append(
                        f"Consider text features extraction for {col} (TF-IDF, embeddings, etc.)"
                    )
            
            # Suggest interaction features for numeric columns
            if len(numeric_cols) >= 2:
                file_suggestions["interactions"].append(
                    f"Consider interaction features between numeric columns"
                )
            
            suggestions[name] = file_suggestions
        
        return suggestions
    
    def generate_plots(self, column_name=None, plot_type=None):
        """
        Generate visualizations for data analysis.
        
        Args:
            column_name: Optional column to plot
            plot_type: Type of plot to generate (histogram, scatter, etc.)
        
        Returns:
            Base64 encoded string of the plot image
        """
        plots = {}
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        for name, df in self.dataframes.items():
            file_plots = {}
            
            # If specific column and plot type are provided
            if column_name and column_name in df.columns and plot_type:
                plt.figure(figsize=(10, 6))
                
                if plot_type == 'histogram':
                    if pd.api.types.is_numeric_dtype(df[column_name]):
                        sns.histplot(df[column_name], kde=True)
                        plt.title(f'Distribution of {column_name}')
                        plt.xlabel(column_name)
                        plt.ylabel('Frequency')
                
                elif plot_type == 'boxplot':
                    if pd.api.types.is_numeric_dtype(df[column_name]):
                        sns.boxplot(y=df[column_name])
                        plt.title(f'Boxplot of {column_name}')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
                file_plots[f"{plot_type}_{column_name}"] = plot_base64
                plt.close()
            
            # Default plots if no specific column/type is provided
            else:
                # Missing values heatmap
                plt.figure(figsize=(12, 8))
                sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
                plt.title('Missing Values Heatmap')
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
                file_plots["missing_values_heatmap"] = plot_base64
                plt.close()
                
                # Correlation matrix for numeric columns
                if len(df.select_dtypes(include=np.number).columns) > 1:
                    plt.figure(figsize=(12, 10))
                    corr = df.select_dtypes(include=np.number).corr()
                    mask = np.triu(np.ones_like(corr, dtype=bool))
                    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt=".2f",
                                linewidths=0.5, vmin=-1, vmax=1)
                    plt.title('Correlation Matrix')
                    plt.tight_layout()
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    file_plots["correlation_matrix"] = plot_base64
                    plt.close()
            
            plots[name] = file_plots
        
        return plots
    
    def plot_correlation_matrix(self, corr_matrix):
        """
        Create a correlation matrix plot.
        
        Args:
            corr_matrix: Correlation matrix from pandas
            
        Returns:
            Matplotlib figure object
        """
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            cmap=cmap, 
            vmax=1, 
            vmin=-1, 
            center=0,
            square=True, 
            linewidths=.5, 
            annot=True, 
            fmt='.2f',
            ax=ax
        )
        
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        return fig
    
    def analyze_query(self, query):
        """
        Analyze the user query and determine what data analysis to perform.
        
        Args:
            query: User's question
        
        Returns:
            Dictionary with relevant data for answering the query
        """
        query = query.lower()
        result = {}
        
        if "dataset" in query or "data" in query or "consist" in query:
            result["basic_info"] = self.get_basic_info()
            result["summary"] = self.get_dataset_summary()
            
        if "clean" in query or "missing" in query or "quality" in query:
            result["data_quality"] = self.check_data_quality()
            
        if "stat" in query or "descriptive" in query:
            result["statistics"] = self.get_statistical_summary()
            
        if "feature" in query or "engineering" in query:
            result["feature_engineering"] = self.suggest_feature_engineering()
            
        if "visual" in query or "plot" in query or "chart" in query:
            result["plots"] = self.generate_plots()
        
        # If no specific analysis was triggered, return all info
        if not result:
            result["basic_info"] = self.get_basic_info()
            result["summary"] = self.get_dataset_summary()
            result["data_quality"] = self.check_data_quality()
            
        return result