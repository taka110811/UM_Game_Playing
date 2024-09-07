import plotly.express as px

class EDA:
    
    def __init__(self, df, color):
        self.df = df  
        self.color = color  

    def template(self, fig, title):
        
        # Set plot background and layout to match the user's theme
        fig.update_layout(
            title=title,
            title_x=0.5, 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',  
            font=dict(color='#7f7f7f'),
            margin=dict(l=90, r=90, t=90, b=90), 
            height=900  
        )
        
        return fig
    
    def target_distribution(self):
        
        # Calculate the distribution of the target variable (utility_agent1)
        target_distribution = self.df['utility_agent1'].value_counts().sort_index()

        # Create a histogram for the target distribution
        fig = px.histogram(
            self.df,
            x='utility_agent1',
            nbins=50,  # Granularity of the histogram
            title='Distribution of Agent 1 Utility',  
            color_discrete_sequence=[self.color]  
        )

        # Customize the histogram layout
        fig.update_layout(
            xaxis_title='Utility of Agent 1',
            yaxis_title='Count', 
            bargap=0.1  
        )

        # Customize hover text: round numbers to 3 decimal places, format large numbers with commas
        fig.update_traces(
            hovertemplate='Utility: %{x:.3f}<br>Count: %{y:,}'
        )

        # Apply the template to the histogram
        fig = self.template(fig, 'Distribution of Agent 1 Utility')

        # Display the histogram
        fig.show()