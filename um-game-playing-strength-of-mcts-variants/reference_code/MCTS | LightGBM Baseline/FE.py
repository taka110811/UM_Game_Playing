import polars as pl
import pandas as pd

class FE:
    
    def __init__(self, batch_size, low_memory):
        self.batch_size = batch_size # Number of lines to read into the buffer at once
        self.low_memory = low_memory # Reduce memory pressure
        
    def clean_data(self, df):
        
        # Define columns to drop
        drop_cols = [
            'Id',
            'num_wins_agent1',
            'num_draws_agent1',
            'num_losses_agent1',
        ]
        
        # Drop columns
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(col)
        
        return df
    
    def set_datatypes(self, df):
        
        # Define categorical columns
        cat_cols = [
            'GameRulesetName',
            'agent1',
            'agent2', 
            'Behaviour', 
            'StateRepetition', 
            'Duration',
            'Complexity',
            'BoardCoverage',
            'GameOutcome',
            'StateEvaluation',
            'Clarity',
            'Decisiveness',
            'Drama',
            'MoveEvaluation',
            'StateEvaluationDifference',
            'BoardSitesOccupied',
            'BranchingFactor',
            'DecisionFactor',
            'MoveDistance',
            'PieceNumber',
            'ScoreDifference',
            'EnglishRules',
            'LudRules'
        ]
        
        # Define numeric columns
        num_cols = [col for col in df.columns if col not in cat_cols]
        
        # Set datatypes for categorical columns
        df = df.with_columns([pl.col(col).cast(pl.Categorical) for col in cat_cols if col in df.columns])            
        
        # Set datatypes for numeric columns
        df = df.with_columns([pl.col(col).cast(pl.Float32) for col in num_cols if col in df.columns])
        
        return df
    
    def extract_cat_cols(self, df):
        
        # Define a list of categorical columns
        cat_cols = []
        
        # Find categorical columns
        for col in df.columns:
            if df[col].dtype == pl.Categorical:
                cat_cols.append(col)
        
        return cat_cols
    
    def extract_cat_cols(self, df):
        
        # Define a list of categorical columns
        cat_cols = []
        
        # Find categorical columns
        for col in df.columns:
            if df[col].dtype == pl.Categorical:
                cat_cols.append(col)
        
        return cat_cols
    
    def display_info(self, df, for_eda):

        # Display information for EDA
        if for_eda:

            # Display the shape of the DataFrame
            print(f'Shape: {df.shape}')

            # Display the memory usage of the DataFrame
            mem = df.memory_usage().sum() / 1024**2
            print('Memory usage: {:.2f} MB\n'.format(mem))

            # Display first rows of the DataFrame
            display(df.head())

        # Display basic information for non-EDA processing
        else:

            # Display the shape of the DataFrame
            print(f'Shape: {df.shape}')

            # Display the memory usage of the DataFrame
            mem = df.estimated_size() / 1024**2
            print('Memory usage: {:.2f} MB\n'.format(mem))

    def process_data(self, path, for_eda=True): # Determines whether to convert to pandas for EDA or keep as polars for processing

        # Load data as polars DataFrame and drop the Id column
        df = pl.read_csv(path, low_memory=self.low_memory, batch_size=self.batch_size)

        # Drop redundant columns
        df = self.clean_data(df)

        # Set datatypes for each column
        df = self.set_datatypes(df)

        # Extract categorical columns
        cat_cols = self.extract_cat_cols(df)

        # Convert Polars to Pandas DataFrame
        if for_eda:
            df = df.to_pandas()

        # Show the shape and first few rows of the DataFrame
        self.display_info(df, for_eda)

        return df, cat_cols