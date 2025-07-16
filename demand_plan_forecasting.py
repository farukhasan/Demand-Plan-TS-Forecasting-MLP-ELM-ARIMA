"""
Demand Planning Forecasting Model
Using Time Series, MLP, ELM and ARIMA approaches
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# For ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    ARIMA_AVAILABLE = True
except ImportError:
    print("Installing required packages for ARIMA...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    ARIMA_AVAILABLE = True

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DemandForecastingModel:
    """
    A comprehensive demand forecasting model using multiple techniques:
    1. Time Series Analysis
    2. Multi-Layer Perceptron (MLP)
    3. Extreme Learning Machine (ELM)
    4. ARIMA Model
    """
    
    def __init__(self):
        self.data = None
        self.scaler = MinMaxScaler()
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
    def generate_synthetic_data(self, n_periods=365, start_date='2022-01-01'):
        """Generate realistic synthetic demand data"""
        
        # Create date range
        dates = pd.date_range(start=start_date, periods=n_periods, freq='D')
        
        # Base demand with trend and seasonality
        t = np.arange(n_periods)
        
        # Trend component
        trend = 100 + 0.1 * t
        
        # Seasonal components
        weekly_seasonality = 20 * np.sin(2 * np.pi * t / 7)  # Weekly pattern
        monthly_seasonality = 30 * np.sin(2 * np.pi * t / 30)  # Monthly pattern
        yearly_seasonality = 50 * np.sin(2 * np.pi * t / 365)  # Yearly pattern
        
        # Random noise
        noise = np.random.normal(0, 15, n_periods)
        
        # Special events (random spikes)
        special_events = np.zeros(n_periods)
        event_days = np.random.choice(n_periods, size=n_periods//20, replace=False)
        special_events[event_days] = np.random.normal(80, 20, len(event_days))
        
        # Combine all components
        demand = trend + weekly_seasonality + monthly_seasonality + yearly_seasonality + noise + special_events
        
        # Ensure non-negative demand
        demand = np.maximum(demand, 10)
        
        # Add some external factors
        temperature = 25 + 10 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 3, n_periods)
        promotion = np.random.binomial(1, 0.1, n_periods)  # 10% of days have promotion
        holiday = np.random.binomial(1, 0.05, n_periods)   # 5% of days are holidays
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'date': dates,
            'demand': demand,
            'temperature': temperature,
            'promotion': promotion,
            'holiday': holiday,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'day_of_year': dates.dayofyear
        })
        
        # Add lag features
        for lag in [1, 7, 30]:
            self.data[f'demand_lag_{lag}'] = self.data['demand'].shift(lag)
            
        # Add moving averages
        for window in [7, 30]:
            self.data[f'demand_ma_{window}'] = self.data['demand'].rolling(window=window).mean()
            
        self.data.dropna(inplace=True)
        
        print(f"Generated {len(self.data)} data points")
        print(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        
        return self.data
    
    def save_data_to_csv(self, filename='demand_data.csv'):
        """Save the generated data to CSV"""
        if self.data is not None:
            self.data.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
        else:
            print("No data to save. Generate data first.")
    
    def load_data_from_csv(self, filename='demand_data.csv'):
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(filename)
            self.data['date'] = pd.to_datetime(self.data['date'])
            print(f"Data loaded from {filename}")
            return self.data
        except FileNotFoundError:
            print(f"File {filename} not found. Generating synthetic data...")
            return self.generate_synthetic_data()
    
    def explore_data(self):
        """Explore and visualize the data"""
        if self.data is None:
            print("No data available. Generate or load data first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time series plot
        axes[0, 0].plot(self.data['date'], self.data['demand'])
        axes[0, 0].set_title('Demand Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Demand')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Distribution
        axes[0, 1].hist(self.data['demand'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Demand Distribution')
        axes[0, 1].set_xlabel('Demand')
        axes[0, 1].set_ylabel('Frequency')
        
        # Seasonal patterns
        monthly_avg = self.data.groupby('month')['demand'].mean()
        axes[1, 0].bar(monthly_avg.index, monthly_avg.values)
        axes[1, 0].set_title('Average Demand by Month')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Average Demand')
        
        # Weekly patterns
        weekly_avg = self.data.groupby('day_of_week')['demand'].mean()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1, 1].bar(range(7), weekly_avg.values)
        axes[1, 1].set_title('Average Demand by Day of Week')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Average Demand')
        axes[1, 1].set_xticks(range(7))
        axes[1, 1].set_xticklabels(days)
        
        plt.tight_layout()
        plt.show()
        
        # Correlation matrix
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_cols].corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.show()
        
        # Basic statistics
        print("\n=== Data Summary ===")
        print(self.data.describe())
    
    def prepare_features(self, target_col='demand'):
        """Prepare features for modeling"""
        if self.data is None:
            print("No data available.")
            return None, None
        
        # Select features (excluding date and target)
        feature_cols = [col for col in self.data.columns if col not in ['date', target_col]]
        
        X = self.data[feature_cols]
        y = self.data[target_col]
        
        return X, y
    
    def create_sequences(self, data, seq_length=30):
        """Create sequences for time series forecasting"""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    def train_mlp_model(self, X_train, X_test, y_train, y_test):
        """Train Multi-Layer Perceptron model"""
        print("\n=== Training MLP Model ===")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create and train MLP
        mlp = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        
        mlp.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred = mlp.predict(X_train_scaled)
        test_pred = mlp.predict(X_test_scaled)
        
        # Store model and predictions
        self.models['MLP'] = {'model': mlp, 'scaler': scaler}
        self.predictions['MLP'] = {
            'train': train_pred,
            'test': test_pred,
            'y_train': y_train,
            'y_test': y_test
        }
        
        # Calculate metrics
        self.metrics['MLP'] = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred)
        }
        
        print(f"MLP - Test MAE: {self.metrics['MLP']['test_mae']:.2f}")
        print(f"MLP - Test RMSE: {self.metrics['MLP']['test_rmse']:.2f}")
        print(f"MLP - Test R²: {self.metrics['MLP']['test_r2']:.4f}")
        
        return mlp
    
    def train_elm_model(self, X_train, X_test, y_train, y_test, n_hidden=100):
        """Train Extreme Learning Machine model"""
        print("\n=== Training ELM Model ===")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # ELM Implementation
        class ELM:
            def __init__(self, n_hidden):
                self.n_hidden = n_hidden
                self.W = None
                self.b = None
                self.beta = None
                
            def fit(self, X, y):
                n_samples, n_features = X.shape
                
                # Random input weights and biases
                self.W = np.random.randn(n_features, self.n_hidden)
                self.b = np.random.randn(self.n_hidden)
                
                # Calculate hidden layer output
                H = np.tanh(np.dot(X, self.W) + self.b)
                
                # Calculate output weights using pseudo-inverse
                self.beta = np.dot(np.linalg.pinv(H), y)
                
            def predict(self, X):
                H = np.tanh(np.dot(X, self.W) + self.b)
                return np.dot(H, self.beta)
        
        # Train ELM
        elm = ELM(n_hidden)
        elm.fit(X_train_scaled, y_train)
        
        # Make predictions
        train_pred = elm.predict(X_train_scaled)
        test_pred = elm.predict(X_test_scaled)
        
        # Store model and predictions
        self.models['ELM'] = {'model': elm, 'scaler': scaler}
        self.predictions['ELM'] = {
            'train': train_pred,
            'test': test_pred,
            'y_train': y_train,
            'y_test': y_test
        }
        
        # Calculate metrics
        self.metrics['ELM'] = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred)
        }
        
        print(f"ELM - Test MAE: {self.metrics['ELM']['test_mae']:.2f}")
        print(f"ELM - Test RMSE: {self.metrics['ELM']['test_rmse']:.2f}")
        print(f"ELM - Test R²: {self.metrics['ELM']['test_r2']:.4f}")
        
        return elm
    
    def train_arima_model(self, train_data, test_data, order=(1, 1, 1)):
        """Train ARIMA model"""
        print("\n=== Training ARIMA Model ===")
        
        # Check stationarity
        result = adfuller(train_data)
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        
        # Train ARIMA model
        try:
            model = ARIMA(train_data, order=order)
            fitted_model = model.fit()
            
            # Make predictions
            train_pred = fitted_model.fittedvalues
            test_pred = fitted_model.forecast(steps=len(test_data))
            
            # Store model and predictions
            self.models['ARIMA'] = {'model': fitted_model}
            self.predictions['ARIMA'] = {
                'train': train_pred,
                'test': test_pred,
                'y_train': train_data,
                'y_test': test_data
            }
            
            # Calculate metrics
            self.metrics['ARIMA'] = {
                'train_mae': mean_absolute_error(train_data[1:], train_pred[1:]),
                'test_mae': mean_absolute_error(test_data, test_pred),
                'train_rmse': np.sqrt(mean_squared_error(train_data[1:], train_pred[1:])),
                'test_rmse': np.sqrt(mean_squared_error(test_data, test_pred)),
                'train_r2': r2_score(train_data[1:], train_pred[1:]),
                'test_r2': r2_score(test_data, test_pred)
            }
            
            print(f"ARIMA - Test MAE: {self.metrics['ARIMA']['test_mae']:.2f}")
            print(f"ARIMA - Test RMSE: {self.metrics['ARIMA']['test_rmse']:.2f}")
            print(f"ARIMA - Test R²: {self.metrics['ARIMA']['test_r2']:.4f}")
            
            return fitted_model
            
        except Exception as e:
            print(f"Error training ARIMA model: {e}")
            return None
    
    def train_all_models(self, test_size=0.2):
        """Train all models and compare performance"""
        if self.data is None:
            print("No data available.")
            return
        
        # Prepare features
        X, y = self.prepare_features()
        
        # Split data for ML models
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False, random_state=42
        )
        
        # Train ML models
        self.train_mlp_model(X_train, X_test, y_train, y_test)
        self.train_elm_model(X_train, X_test, y_train, y_test)
        
        # Prepare data for ARIMA (time series only)
        ts_data = self.data['demand'].values
        split_idx = int(len(ts_data) * (1 - test_size))
        train_ts = ts_data[:split_idx]
        test_ts = ts_data[split_idx:]
        
        # Train ARIMA model
        self.train_arima_model(train_ts, test_ts)
        
        # Print comparison
        self.compare_models()
    
    def compare_models(self):
        """Compare performance of all models"""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        comparison_data = []
        for model_name, metrics in self.metrics.items():
            comparison_data.append({
                'Model': model_name,
                'Test MAE': metrics['test_mae'],
                'Test RMSE': metrics['test_rmse'],
                'Test R²': metrics['test_r2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_model = comparison_df.loc[comparison_df['Test MAE'].idxmin(), 'Model']
        print(f"\nBest Model (lowest MAE): {best_model}")
        
        return comparison_df
    
    def plot_predictions(self, model_name='all', n_points=100):
        """Plot predictions vs actual values"""
        if model_name == 'all':
            models_to_plot = list(self.predictions.keys())
        else:
            models_to_plot = [model_name] if model_name in self.predictions else []
        
        if not models_to_plot:
            print("No models to plot.")
            return
        
        fig, axes = plt.subplots(len(models_to_plot), 2, figsize=(15, 5*len(models_to_plot)))
        if len(models_to_plot) == 1:
            axes = axes.reshape(1, -1)
        
        for i, model in enumerate(models_to_plot):
            pred_data = self.predictions[model]
            
            # Plot training predictions
            y_train = pred_data['y_train']
            train_pred = pred_data['train']
            
            # Limit points for better visualization
            if len(y_train) > n_points:
                idx = np.linspace(0, len(y_train)-1, n_points, dtype=int)
                y_train_plot = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
                train_pred_plot = train_pred[idx]
            else:
                y_train_plot = y_train
                train_pred_plot = train_pred
            
            axes[i, 0].plot(y_train_plot, label='Actual', alpha=0.7)
            axes[i, 0].plot(train_pred_plot, label='Predicted', alpha=0.7)
            axes[i, 0].set_title(f'{model} - Training Set')
            axes[i, 0].set_xlabel('Time')
            axes[i, 0].set_ylabel('Demand')
            axes[i, 0].legend()
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot test predictions
            y_test = pred_data['y_test']
            test_pred = pred_data['test']
            
            axes[i, 1].plot(y_test, label='Actual', alpha=0.7)
            axes[i, 1].plot(test_pred, label='Predicted', alpha=0.7)
            axes[i, 1].set_title(f'{model} - Test Set')
            axes[i, 1].set_xlabel('Time')
            axes[i, 1].set_ylabel('Demand')
            axes[i, 1].legend()
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def forecast_future(self, days=30):
        """Make future predictions using all models"""
        if not self.models:
            print("No trained models available.")
            return
        
        print(f"\n=== Forecasting next {days} days ===")
        
        # Get last known values for feature engineering
        last_row = self.data.iloc[-1:].copy()
        future_predictions = {}
        
        for model_name, model_info in self.models.items():
            if model_name == 'ARIMA':
                # ARIMA forecasting
                arima_model = model_info['model']
                forecast = arima_model.forecast(steps=days)
                future_predictions[model_name] = forecast
                
            else:
                # ML models forecasting
                model = model_info['model']
                scaler = model_info['scaler']
                
                # For simplicity, we'll use the last row's features
                # In practice, you'd want to engineer features for future dates
                last_features = self.prepare_features()[0].iloc[-1:].values
                last_features_scaled = scaler.transform(last_features)
                
                # Simple approach: repeat prediction (in practice, you'd update features)
                predictions = []
                for _ in range(days):
                    pred = model.predict(last_features_scaled)[0]
                    predictions.append(pred)
                
                future_predictions[model_name] = np.array(predictions)
        
        # Plot future predictions
        future_dates = pd.date_range(
            start=self.data['date'].iloc[-1] + pd.Timedelta(days=1),
            periods=days,
            freq='D'
        )
        
        plt.figure(figsize=(12, 6))
        
        # Plot historical data (last 60 days)
        historical_data = self.data.tail(60)
        plt.plot(historical_data['date'], historical_data['demand'], 
                label='Historical', color='black', linewidth=2)
        
        # Plot future predictions
        colors = ['red', 'blue', 'green', 'orange']
        for i, (model_name, predictions) in enumerate(future_predictions.items()):
            plt.plot(future_dates, predictions, 
                    label=f'{model_name} Forecast', 
                    color=colors[i % len(colors)], 
                    linestyle='--', linewidth=2)
        
        plt.axvline(x=self.data['date'].iloc[-1], color='gray', linestyle=':', alpha=0.7)
        plt.title(f'Demand Forecast - Next {days} Days')
        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Print forecast summary
        forecast_df = pd.DataFrame(future_predictions, index=future_dates)
        print("\nForecast Summary (First 10 days):")
        print(forecast_df.head(10))
        
        return forecast_df

# Example usage
if __name__ == "__main__":
    # Initialize the forecasting model
    forecaster = DemandForecastingModel()
    
    # Generate synthetic data
    print("Generating synthetic demand data...")
    data = forecaster.generate_synthetic_data(n_periods=500)
    
    # Save data to CSV
    forecaster.save_data_to_csv('demand_forecasting_data.csv')
    
    # Explore the data
    print("\nExploring the data...")
    forecaster.explore_data()
    
    # Train all models
    print("\nTraining all models...")
    forecaster.train_all_models()
    
    # Plot predictions
    print("\nPlotting predictions...")
    forecaster.plot_predictions()
    
    # Make future forecasts
    print("\nMaking future forecasts...")
    future_forecast = forecaster.forecast_future(days=30)
    
    print("\n" + "="*50)
    print("DEMAND FORECASTING COMPLETE!")
    print("="*50)