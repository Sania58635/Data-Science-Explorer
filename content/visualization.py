"""
Data Visualization Content Module
Communicating insights through visual storytelling
"""

def get_content():
    return {
        'summary': """
        <div class="topic-summary">
            <h3>📋 What is Data Visualization?</h3>
            <p>Data visualization transforms numbers into charts, graphs, and dashboards that reveal patterns and insights at a glance. Good visualizations make complex data understandable and help communicate findings to both technical and non-technical audiences.</p>
            <p><strong>Used in:</strong> Business Dashboards, Exploratory Data Analysis, Report Generation, Presentations, Web Analytics, and Interactive Data Apps.</p>
        </div>
        """,
        'topics': [
            {
                'id': 'chart-types',
                'title': 'Essential Chart Types',
                'description': 'Choosing the right visualization',
                'subtopics': [
                    {
                        'name': 'Distribution Plots',
                        'content': """
                        <h3>Visualizing Distributions</h3>
                        <p><strong>Histogram:</strong> Shows frequency distribution of continuous variable</p>
                        <ul>
                            <li>Reveals shape, central tendency, spread</li>
                            <li>Bin width affects interpretation</li>
                        </ul>
                        <p><strong>Box Plot:</strong> Displays quartiles, median, outliers</p>
                        <ul>
                            <li>Box spans IQR (Q1 to Q3)</li>
                            <li>Whiskers extend to 1.5×IQR</li>
                            <li>Points beyond whiskers are outliers</li>
                        </ul>
                        <p><strong>Violin Plot:</strong> Combines box plot with kernel density estimation</p>
                        <p><strong>KDE Plot:</strong> Smooth probability density curve</p>
                        <div class="visual">
                            <img src="/static/images/chart_types.png" alt="Essential Chart Types" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        <div class="example">
                            <h4>Python Example</h4>
                            <pre><code>import matplotlib.pyplot as plt
import seaborn as sns

# Histogram
plt.hist(data, bins=30, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Box plot
sns.boxplot(data=df, x='category', y='value')
plt.show()</code></pre>
                        </div>
                        """
                    },
                    {
                        'name': 'Comparison Charts',
                        'content': """
                        <h3>Comparing Values</h3>
                        <p><strong>Bar Chart:</strong> Compare categorical data</p>
                        <ul>
                            <li>Horizontal for long labels</li>
                            <li>Grouped for multiple categories</li>
                            <li>Stacked for part-to-whole relationships</li>
                        </ul>
                        <p><strong>Line Chart:</strong> Show trends over time</p>
                        <ul>
                            <li>Multiple lines for comparison</li>
                            <li>Ideal for continuous time series</li>
                        </ul>
                        <p><strong>Scatter Plot:</strong> Relationship between two variables</p>
                        <ul>
                            <li>Add color for third dimension</li>
                            <li>Size for fourth dimension (bubble chart)</li>
                            <li>Trend line shows correlation</li>
                        </ul>
                        <p><strong>Heatmap:</strong> Matrix of values with color encoding</p>
                        <ul>
                            <li>Correlation matrices</li>
                            <li>Confusion matrices</li>
                            <li>Temporal patterns</li>
                        </ul>
                        <div class="visual">
                            <img src="/static/images/time_series.png" alt="Time Series Visualization" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        """
                    },
                    {
                        'name': 'Composition Charts',
                        'content': """
                        <h3>Part-to-Whole Relationships</h3>
                        <p><strong>Pie Chart:</strong> Show proportions (use sparingly!)</p>
                        <ul>
                            <li>Limit to 5-7 slices maximum</li>
                            <li>Start at 12 o'clock</li>
                            <li>Consider donut chart variant</li>
                        </ul>
                        <p><strong>Treemap:</strong> Hierarchical data with nested rectangles</p>
                        <ul>
                            <li>Size represents value</li>
                            <li>Color adds additional dimension</li>
                        </ul>
                        <p><strong>Stacked Area Chart:</strong> Show cumulative totals over time</p>
                        <p><strong>Waterfall Chart:</strong> Visualize sequential changes</p>
                        """
                    }
                ]
            },
            {
                'id': 'dashboards',
                'title': 'Dashboard Design',
                'description': 'Creating effective analytical interfaces',
                'subtopics': [
                    {
                        'name': 'Dashboard Principles',
                        'content': """
                        <h3>Effective Dashboard Design</h3>
                        <p><strong>Key Principles:</strong></p>
                        <ul>
                            <li><strong>Clarity:</strong> Clear hierarchy, logical grouping</li>
                            <li><strong>Simplicity:</strong> Avoid clutter, focus on KPIs</li>
                            <li><strong>Context:</strong> Provide benchmarks, targets, trends</li>
                            <li><strong>Actionability:</strong> Enable decision-making</li>
                        </ul>
                        <p><strong>Layout Strategies:</strong></p>
                        <ul>
                            <li><strong>F-Pattern:</strong> Most important info top-left</li>
                            <li><strong>Z-Pattern:</strong> Guide eye across dashboard</li>
                            <li><strong>Grid System:</strong> Consistent alignment and spacing</li>
                        </ul>
                        <p><strong>Components:</strong></p>
                        <ul>
                            <li>KPI cards for key metrics</li>
                            <li>Trend charts for temporal patterns</li>
                            <li>Comparison charts for segments</li>
                            <li>Filters for interactivity</li>
                        </ul>
                        """
                    }
                ]
            },
            {
                'id': 'plotly-dash',
                'title': 'Interactive Visualizations',
                'description': 'Building dynamic plots with Plotly and Dash',
                'subtopics': [
                    {
                        'name': 'Plotly Basics',
                        'content': """
                        <h3>Plotly Interactive Charts</h3>
                        <p><strong>Features:</strong> Hover tooltips, zoom, pan, export, responsive design</p>
                        <div class="visual">
                            <iframe src="/static/images/plotly_interactive.html" width="100%" height="500" frameborder="0" style="border-radius: 12px;"></iframe>
                        </div>
                        <div class="example">
                            <h4>Interactive Scatter Plot</h4>
                            <pre><code>import plotly.express as px
import pandas as pd

df = px.data.iris()
fig = px.scatter(df, 
                 x='sepal_width', 
                 y='sepal_length',
                 color='species',
                 size='petal_length',
                 hover_data=['petal_width'],
                 title='Iris Dataset Analysis')
fig.show()</code></pre>
                        </div>
                        <p><strong>Chart Types:</strong></p>
                        <ul>
                            <li>Scatter, Line, Bar, Histogram</li>
                            <li>Box, Violin, Heatmap</li>
                            <li>3D plots, Maps, Animations</li>
                            <li>Subplots and facets</li>
                        </ul>
                        """
                    },
                    {
                        'name': 'Dash Applications',
                        'content': """
                        <h3>Building Dash Web Apps</h3>
                        <p><strong>Architecture:</strong> Python-based framework for analytical web applications</p>
                        <div class="example">
                            <h4>Simple Dash App</h4>
                            <pre><code>import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Data Science Dashboard'),
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': i, 'value': i} for i in df.columns],
        value=df.columns[0]
    ),
    dcc.Graph(id='graph')
])

@app.callback(
    Output('graph', 'figure'),
    Input('dropdown', 'value')
)
def update_graph(selected_column):
    fig = px.histogram(df, x=selected_column)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)</code></pre>
                        </div>
                        <p><strong>Components:</strong> Dropdowns, sliders, inputs, graphs, tables, markdown</p>
                        """
                    }
                ]
            },
            {
                'id': 'visualization-principles',
                'title': 'Visualization Best Practices',
                'description': 'Design principles for effective charts',
                'subtopics': [
                    {
                        'name': 'Design Principles',
                        'content': """
                        <h3>Effective Visualization Design</h3>
                        <p><strong>Color Usage:</strong></p>
                        <ul>
                            <li><strong>Sequential:</strong> Light to dark for ordered data</li>
                            <li><strong>Diverging:</strong> Two hues for data with critical midpoint</li>
                            <li><strong>Categorical:</strong> Distinct colors for categories</li>
                            <li><strong>Accessibility:</strong> Colorblind-friendly palettes</li>
                        </ul>
                        <p><strong>Gestalt Principles:</strong></p>
                        <ul>
                            <li><strong>Proximity:</strong> Group related items</li>
                            <li><strong>Similarity:</strong> Similar appearance implies relationship</li>
                            <li><strong>Enclosure:</strong> Borders create grouping</li>
                            <li><strong>Closure:</strong> Mind fills gaps</li>
                        </ul>
                        <p><strong>Data-Ink Ratio:</strong> Maximize data, minimize decoration (Tufte principle)</p>
                        """
                    },
                    {
                        'name': 'Common Pitfalls',
                        'content': """
                        <h3>Visualization Mistakes to Avoid</h3>
                        <p><strong>Chart Junk:</strong> Unnecessary decorations, 3D effects, excessive colors</p>
                        <p><strong>Misleading Axes:</strong></p>
                        <ul>
                            <li>Truncated y-axis exaggerates differences</li>
                            <li>Non-zero baseline distorts perception</li>
                            <li>Inconsistent scales between charts</li>
                        </ul>
                        <p><strong>Wrong Chart Type:</strong></p>
                        <ul>
                            <li>Pie chart with too many slices</li>
                            <li>3D pie charts (perspective distortion)</li>
                            <li>Dual axes without clear labels</li>
                        </ul>
                        <p><strong>Overplotting:</strong> Too many data points obscure patterns (use transparency, sampling, or aggregation)</p>
                        <p><strong>Missing Context:</strong> No title, labels, units, or legends</p>
                        """
                    }
                ]
            },
            {
                'id': 'case-studies',
                'title': 'Visualization Case Studies',
                'description': 'Real-world examples and success stories',
                'subtopics': [
                    {
                        'name': 'Business Analytics',
                        'content': """
                        <h3>Business Intelligence Dashboards</h3>
                        <p><strong>Sales Performance Dashboard:</strong></p>
                        <ul>
                            <li>KPI cards: Revenue, growth rate, conversion</li>
                            <li>Time series: Monthly sales trends</li>
                            <li>Bar charts: Sales by region/product</li>
                            <li>Funnel chart: Conversion pipeline</li>
                        </ul>
                        <p><strong>Customer Analytics:</strong></p>
                        <ul>
                            <li>Cohort analysis heatmaps</li>
                            <li>Retention curves over time</li>
                            <li>Segment comparison bar charts</li>
                            <li>Geographic maps for regional insights</li>
                        </ul>
                        """
                    },
                    {
                        'name': 'Exploratory Data Analysis',
                        'content': """
                        <h3>EDA Visualization Workflows</h3>
                        <p><strong>Univariate Analysis:</strong></p>
                        <ul>
                            <li>Histograms for distributions</li>
                            <li>Box plots for outliers</li>
                            <li>Bar charts for categorical frequencies</li>
                        </ul>
                        <p><strong>Bivariate Analysis:</strong></p>
                        <ul>
                            <li>Scatter plots with trend lines</li>
                            <li>Correlation heatmaps</li>
                            <li>Grouped bar charts</li>
                        </ul>
                        <p><strong>Multivariate Analysis:</strong></p>
                        <ul>
                            <li>Pair plots (scatterplot matrix)</li>
                            <li>Parallel coordinates</li>
                            <li>PCA biplots</li>
                            <li>t-SNE/UMAP embeddings</li>
                        </ul>
                        """
                    }
                ]
            }
        ]
    }

