import gc
import os
import kaggle_evaluation
from kaggle_evaluation.mcts_inference_server import MCTSInferenceServer

import polars as pl

from FE import FE
from MD import MD
from CFG import CFG
from EDA import EDA

def train_model(fe, md, CFG):
    
    global cat_cols, lgb_models
    
    # Load and process train data - extract categorical columns
    train, cat_cols = fe.process_data(CFG.train_path, for_eda=False)
    
    # Train LightGBM models
    lgb_models, _ = md.train_lgb(train, cat_cols, 'LightGBM')


# Define the predict function for the API
def predict(test, submission, fe, md):
    
    # Use the global counter variable
    global counter
    
    # If this is the first prediction call, train LightGBM models
    if counter == 0:
        
        # Train LightGBM models
        train_model(fe, md, CFG)
        
    # Increment the counter for each prediction call to avoid re-training
    counter += 1
    
    # Drop redundant columns
    test = fe.clean_data(test)

    # Set datatypes for each column
    test = fe.set_datatypes(test)
    
    # Generate test predictions and assign them to the submission DataFrame
    return submission.with_columns(pl.Series('utility_agent1', md.infer_lgb(test, cat_cols, lgb_models)))


def main():
    # Initialize class for feature engineering
    fe = FE(CFG.batch_size, CFG.low_memory)
    # Load and process train data
    train_data, _ = fe.process_data(CFG.train_path)

    # Initialize class for Exploratory Data Analysis (EDA)
    eda = EDA(train_data, CFG.color)
    eda.target_distribution()
    # Delete references to train data
    del train_data
    gc.collect()

    # Initialize class for model development# Initialize class for model development
    md = MD(CFG.early_stop,
            CFG.n_splits,
            CFG.color, 
            CFG.lgb_p)
    
    # Initialize a counter to keep track of prediction calls
    counter = 0

    # Load and process test datainference_server = kaggle_evaluation.mcts_inference_server.MCTSInferenceServer(predict)
    inference_server = MCTSInferenceServer(predict)

    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(
            (
                '/kaggle/input/um-game-playing-strength-of-mcts-variants/test.csv',
                '/kaggle/input/um-game-playing-strength-of-mcts-variants/sample_submission.csv'
            )
        )

if __name__ == '__main__':
    main()
