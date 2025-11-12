"""
Data Loader for Enterprise Datasets
Handles loading enterprise datasets from Kaggle
"""

import pandas as pd
import os
from typing import Dict, List
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

class EnterpriseDataLoader:
    """
    Loads enterprise datasets from Kaggle
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.kaggle_api = KaggleApi()
        self.kaggle_api.authenticate()
        self.logger = logging.getLogger(__name__)
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

    def load_kaggle_dataset(self, dataset_name: str, custom_filename: str = None) -> pd.DataFrame:
        """
        Load dataset from Kaggle with custom filename support and download checks
        """
        try:
            # Create custom filename if provided
            if custom_filename:
                target_file = os.path.join(self.data_dir, custom_filename)
                if os.path.exists(target_file):
                    self.logger.info(f"File already exists: {target_file}")
                    return pd.read_csv(target_file)
            
            # Download dataset only if no custom file exists
            self.kaggle_api.dataset_download_files(
                dataset_name, 
                path=self.data_dir, 
                unzip=True
            )
            
            # Find the downloaded files
            files = os.listdir(self.data_dir)
            csv_files = [f for f in files if f.endswith('.csv')]
            
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
            
            # Load the first CSV file
            file_path = os.path.join(self.data_dir, csv_files[0])
            
            # Rename to custom filename if specified
            if custom_filename and not os.path.exists(os.path.join(self.data_dir, custom_filename)):
                new_file_path = os.path.join(self.data_dir, custom_filename)
                os.rename(file_path, new_file_path)
                file_path = new_file_path
            
            df = pd.read_csv(file_path)
            
            self.logger.info(f"Loaded dataset {dataset_name} with shape {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading Kaggle dataset {dataset_name}: {e}")
            raise
    
    def get_enterprise_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Get multiple enterprise datasets from Kaggle
        Downloads datasets and saves with custom filenames
        """
        datasets = {}
        
        # Kaggle datasets to download with custom filenames
        kaggle_datasets = [
            # ('pavansubhasht/ibm-hr-analytics-attrition-dataset', 'hr_analytics', 'hr_employee_attrition.csv'),
            # ('aslanahmedov/walmart-sales-forecast', 'sales_forecasting', 'warlmart_sales_forecasting_data.csv'), # this dataset contains 4 files: features.csv, stores.csv, test.csv, train.csv
            ('rohitsahoo/sales-forecasting', 'superstore_sales', 'global_superstore_sales_dataset.csv')
        ]
        
        for dataset_name, dataset_key, custom_filename in kaggle_datasets:
            try:
                df = self.load_kaggle_dataset(dataset_name, custom_filename)
                datasets[dataset_key] = df
                self.logger.info(f"Successfully loaded Kaggle dataset: {dataset_name} as {custom_filename}")
            except Exception as e:
                self.logger.warning(f"Could not load Kaggle dataset {dataset_name}: {e}")
        
        return datasets

# Example usage
if __name__ == "__main__":
    loader = EnterpriseDataLoader()
    
    # Load datasets
    datasets = loader.get_enterprise_datasets()
    
    print("Available datasets:")
    for name, df in datasets.items():
        print(f"  {name}: {df.shape}")
        print(f"    Columns: {list(df.columns)}")