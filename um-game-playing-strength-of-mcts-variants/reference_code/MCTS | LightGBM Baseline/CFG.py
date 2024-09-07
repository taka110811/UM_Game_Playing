from pathlib import Path

class CFG:
    
    # Paths to competition data
    train_path = Path('um-game-playing-strength-of-mcts-variants/data/train.csv')
    test_path = Path('um-game-playing-strength-of-mcts-variants/data/test.csv')
    subm_path = Path('um-game-playing-strength-of-mcts-variants/data/sample_submission.csv')
    
    # Feature engineering (FE) arguments
    batch_size = 16384
    low_memory = True
    
    # Color for EDA and MD
    color = '#E0BFB8'
    
    # Model development (MD) arguments
    early_stop = 50
    n_splits = 5
    
    # LightGBM parameters
    lgb_p = {
        'objective': 'regression',
        'num_iterations': 400,
        'learning_rate': 0.03,
        'extra_trees': True,
        'reg_lambda': 0.8,
        'num_leaves': 64,
        'metric': 'rmse',
        'device': 'cpu',
        'max_depth': 4,
        'max_bin': 128,
        'verbose': -1,
        'seed': 42
    }