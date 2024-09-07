import lightgbm as lgb
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse

class MD:
    
    def __init__(self, 
                 early_stop, 
                 n_splits,
                 color,
                 lgb_p):
        
        self.early_stop = early_stop
        self.n_splits = n_splits
        self.color = color
        self.lgb_p = lgb_p
    
    def plot_cv(self, fold_scores, model_name):
        
        # Round the fold scores to 3 decimal places
        fold_scores = [round(score, 3) for score in fold_scores]
        mean_score = round(np.mean(fold_scores), 3)
        std_score = round(np.std(fold_scores), 3)

        # Create a new figure for plotting
        fig = go.Figure()

        # Add scatter plot for individual fold scores
        fig.add_trace(go.Scatter(
            x = list(range(1, len(fold_scores) + 1)),
            y = fold_scores,
            mode = 'markers', 
            name = 'Fold Scores',
            marker = dict(size = 24, color=self.color, symbol='diamond'), # Diamond shape marker
            text = [f'{score:.3f}' for score in fold_scores],
            hovertemplate = 'Fold %{x}: %{text}<extra></extra>',
            hoverlabel=dict(font=dict(size=16))  # Adjust the font size here
        ))

        # Add a horizontal line for the mean score
        fig.add_trace(go.Scatter(
            x = [1, len(fold_scores)],
            y = [mean_score, mean_score],
            mode = 'lines',
            name = f'Mean: {mean_score:.3f}',
            line = dict(dash = 'dash', color = '#FFBF00'), # Colored Amber
            hoverinfo = 'none'
        ))

        # Update the layout of the plot
        fig.update_layout(
            title = f'{model_name} | Cross-Validation RMSE Scores | Variation of CV scores: {mean_score} ± {std_score}',
            xaxis_title = 'Fold',
            yaxis_title = 'RMSE Score',
            plot_bgcolor = 'rgba(0,0,0,0)',
            paper_bgcolor = 'rgba(0,0,0,0)',
            xaxis = dict(
                gridcolor = 'lightgray',
                tickmode = 'linear',
                tick0 = 1,
                dtick = 1,
                range = [0.5, len(fold_scores) + 0.5]
            ),
            yaxis = dict(gridcolor = 'lightgray')
        )

        # Display the plot
        fig.show() 
        
    def train_lgb(self, data, cat_cols, title):
        
        # Convert data for pandas for training
        data = data.to_pandas()
        
        # Extract features columns and label
        X = data.drop(['utility_agent1'], axis=1)
        y = data['utility_agent1']
        
        # Convert categorical columns to category dtype
        for col in cat_cols:
            X[col] = X[col].astype('category')
        
        # Initialize cross-validation strategy
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        # Initialize lists to store models, CV scores, and OOF predictions
        models, scores = [], []
        oof_preds = np.zeros(len(X))
        
        # Perform cross-validation
        for fold, (train_index, valid_index) in enumerate(cv.split(X, y)):
            
            # Split the data into training and validation sets for the current fold
            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
            # Train the model
            model = lgb.LGBMRegressor(**self.lgb_p)
            model.fit(X_train, y_train,
                      eval_set=[(X_valid, y_valid)],
                      eval_metric='rmse',
                      callbacks=[lgb.early_stopping(self.early_stop, verbose=0), 
                                 lgb.log_evaluation(0)])
            
            # Append the trained model to the list
            models.append(model)
            
            # Make predictions on the validation set
            oof_preds[valid_index] = model.predict(X_valid)
            
            # Calculate and store the RMSE score for the current fold
            score = mse(y_valid, oof_preds[valid_index], squared=False)
            scores.append(score)
        
        # Plot the cross-validation results
        self.plot_cv(scores, title)
        
        return models, oof_preds

    def infer_lgb(self, data, cat_cols, models):
        
        # Convert data for pandas for inference
        data = data.to_pandas()

        # Convert categorical columns to category dtype
        for col in cat_cols:
            data[col] = data[col].astype('category')

        # Return the averaged predictions of LightGBM models
        return np.mean([model.predict(data) for model in models], axis=0)