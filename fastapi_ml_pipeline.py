from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import io
import os
import tempfile
import uuid
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path
import base64
import requests
from supabase import create_client, Client
import json

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Supabase configuration
SUPABASE_URL = "https://hyzttvamqpiilykxjjvl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imh5enR0dmFtcXBpaWx5a3hqanZsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE0ODIzNTksImV4cCI6MjA2NzA1ODM1OX0.tUI83upn7zzBJU7ys0FA7dHLMcBUrV9KfrO8nanHnlg"
BUCKET_NAME = "dashboard-images"

# Initialize FastAPI app
app = FastAPI(title="SKU Demand Predictor API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_season(month):
    """Helper function to get season from month"""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

class SKUDemandPredictor:
    def __init__(self, top_n=10, lookback_months=3):
        """
        Initialize the SKU Demand Predictor
        
        Args:
            top_n (int): Number of top SKUs to predict
            lookback_months (int): Number of historical months to use for features
        """
        self.top_n = top_n
        self.lookback_months = lookback_months
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.monthly_data = None
        self.historical_top_skus = {}
        
    def convert_google_drive_url(url: str) -> str:
      """Convert Google Drive sharing URL to direct download URL"""
      if "drive.google.com" in url:
          # Extract file ID from the URL
          if "/file/d/" in url:
            file_id = url.split("/file/d/")[1].split("/")[0]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
          elif "id=" in url:
              file_id = url.split("id=")[1].split("&")[0]
              return f"https://drive.google.com/uc?export=download&id={file_id}"
      return url
    def load_and_preprocess_data(self, file_path_or_df):
       """
    Load and preprocess the CSV data
    
    Args:
        file_path_or_df: Path to CSV file or pandas DataFrame
       """
    # Load data
       if isinstance(file_path_or_df, str):
          df = pd.read_csv(file_path_or_df)
       else:
           df = file_path_or_df.copy()
    
    # Convert date column to datetime with error handling
       try:
           df['date'] = pd.to_datetime(df['date'])
       except Exception as e:
           logger.error(f"Error converting date column: {str(e)}")
        # Try different date formats
           try:
               df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
           except:
               try:
                   df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
               except:
                   try:
                       df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
                   except:
                       # If all else fails, use infer_datetime_format
                       df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    
    # Create time-based features
       df['year'] = df['date'].dt.year
       df['month'] = df['date'].dt.month
       df['day'] = df['date'].dt.day
       df['day_of_week'] = df['date'].dt.dayofweek
       df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
       df['quarter'] = df['date'].dt.quarter
    
    # Create season feature
       df['season'] = df['month'].apply(get_season)
    
    # Create month-year identifier
       df['month_year'] = df['date'].dt.to_period('M')
    
    # Group by SKU and month-year, aggregate quantities
       monthly_agg = df.groupby(['sku', 'month_year', 'item', 'category']).agg({
         'quantity': 'sum',
         'location': 'count',  # number of dispatches
         'year': 'first',
         'month': 'first',
         'quarter': 'first',
         'season': 'first'
        }).reset_index()
    
       monthly_agg.rename(columns={'location': 'dispatch_count'}, inplace=True)
    
       self.monthly_data = monthly_agg
       return self.monthly_data
    
    def create_features(self):
        """Create features for machine learning model"""
        # Get all unique month-year periods and sort them
        periods = sorted(self.monthly_data['month_year'].unique())
        
        # Get all unique SKUs
        all_skus = self.monthly_data['sku'].unique()
        
        features_list = []
        
        # Start from lookback_months to have enough historical data
        for i in range(self.lookback_months, len(periods)):
            current_period = periods[i]
            
            # Get current month data
            current_month_data = self.monthly_data[
                self.monthly_data['month_year'] == current_period
            ]
            
            # Identify top N SKUs for this month (labels)
            top_skus = current_month_data.nlargest(self.top_n, 'quantity')['sku'].tolist()
            self.historical_top_skus[str(current_period)] = top_skus
            
            # Create features for each SKU
            for sku in all_skus:
                feature_row = {'sku': sku, 'month_year': current_period}
                
                # Current month info
                sku_current = current_month_data[current_month_data['sku'] == sku]
                if not sku_current.empty:
                    feature_row.update({
                        'month': sku_current.iloc[0]['month'],
                        'quarter': sku_current.iloc[0]['quarter'],
                        'season': sku_current.iloc[0]['season'],
                        'category': sku_current.iloc[0]['category'],
                        'item': sku_current.iloc[0]['item']
                    })
                else:
                    # Fill with defaults if SKU not present in current month
                    period_date = pd.to_datetime(str(current_period))
                    feature_row.update({
                        'month': period_date.month,
                        'quarter': period_date.quarter,
                        'season': get_season(period_date.month),
                        'category': 'Unknown',
                        'item': 'Unknown'
                    })
                
                # Historical features (lookback months)
                historical_quantities = []
                historical_dispatches = []
                
                for j in range(1, self.lookback_months + 1):
                    hist_period = periods[i - j]
                    hist_data = self.monthly_data[
                        (self.monthly_data['month_year'] == hist_period) & 
                        (self.monthly_data['sku'] == sku)
                    ]
                    
                    if not hist_data.empty:
                        qty = hist_data.iloc[0]['quantity']
                        dispatches = hist_data.iloc[0]['dispatch_count']
                    else:
                        qty = 0
                        dispatches = 0
                    
                    feature_row[f'qty_lag_{j}'] = qty
                    feature_row[f'dispatch_lag_{j}'] = dispatches
                    historical_quantities.append(qty)
                    historical_dispatches.append(dispatches)
                
                # Rolling statistics
                feature_row['qty_mean'] = np.mean(historical_quantities)
                feature_row['qty_std'] = np.std(historical_quantities)
                feature_row['qty_max'] = np.max(historical_quantities)
                feature_row['qty_min'] = np.min(historical_quantities)
                feature_row['qty_trend'] = np.polyfit(range(len(historical_quantities)), historical_quantities, 1)[0]
                
                feature_row['dispatch_mean'] = np.mean(historical_dispatches)
                feature_row['dispatch_std'] = np.std(historical_dispatches)
                
                # Category-based features
                category_data = self.monthly_data[
                    (self.monthly_data['category'] == feature_row['category']) &
                    (self.monthly_data['month_year'].isin(periods[i-self.lookback_months:i]))
                ]
                feature_row['category_avg_qty'] = category_data['quantity'].mean() if not category_data.empty else 0
                
                # Seasonal features
                same_month_data = self.monthly_data[
                    (self.monthly_data['month'] == feature_row['month']) &
                    (self.monthly_data['sku'] == sku)
                ]
                feature_row['seasonal_avg'] = same_month_data['quantity'].mean() if not same_month_data.empty else 0
                
                # Label: 1 if SKU is in top N, 0 otherwise
                feature_row['is_top'] = 1 if sku in top_skus else 0
                
                features_list.append(feature_row)
        
        self.features_df = pd.DataFrame(features_list)
        return self.features_df
    
    def prepare_model_data(self):
        """Prepare data for machine learning model"""
        # Encode categorical variables
        categorical_columns = ['season', 'category', 'item']
        
        for col in categorical_columns:
            if col in self.features_df.columns:
                le = LabelEncoder()
                self.features_df[col] = le.fit_transform(self.features_df[col].astype(str))
                self.label_encoders[col] = le
        
        # Define feature columns (exclude target and identifiers)
        exclude_cols = ['sku', 'month_year', 'is_top']
        self.feature_columns = [col for col in self.features_df.columns if col not in exclude_cols]
        
        # Prepare X and y
        X = self.features_df[self.feature_columns]
        y = self.features_df['is_top']
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_model(self, model_type='rf', test_size=0.2, random_state=42):
        """Train the machine learning model"""
        # Prepare data
        X, y = self.prepare_model_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Handle class imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        # Train model
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=class_weight_dict,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == 'xgb':
            scale_pos_weight = class_weights[0] / class_weights[1]
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                eval_metric='logloss'
            )
        
        self.model.fit(X_train, y_train)
        
        # Store test data for analysis
        self.X_test = X_test
        self.y_test = y_test
        
        return {"message": "Model trained successfully"}
    
    def predict_next_month(self, target_month=None):
        """Predict top SKUs for next month"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        # Determine target month
        if target_month is None:
            latest_period = self.monthly_data['month_year'].max()
            next_period = latest_period + 1
        else:
            next_period = pd.Period(target_month, freq='M')
        
        # Get all SKUs
        all_skus = self.monthly_data['sku'].unique()
        
        # Create features for next month
        prediction_features = []
        
        for sku in all_skus:
            feature_row = {'sku': sku, 'month_year': next_period}
            
            # Basic time features
            next_date = pd.to_datetime(str(next_period))
            feature_row.update({
                'month': next_date.month,
                'quarter': next_date.quarter,
                'season': get_season(next_date.month)
            })
            
            # Get SKU info from historical data
            sku_info = self.monthly_data[self.monthly_data['sku'] == sku]
            if not sku_info.empty:
                feature_row['category'] = sku_info.iloc[0]['category']
                feature_row['item'] = sku_info.iloc[0]['item']
            else:
                feature_row['category'] = 'Unknown'
                feature_row['item'] = 'Unknown'
            
            # Historical features
            historical_quantities = []
            historical_dispatches = []
            
            # Get recent periods for features
            recent_periods = []
            current_period = next_period - 1
            for j in range(self.lookback_months):
                recent_periods.append(current_period - j)
            
            for j, hist_period in enumerate(recent_periods, 1):
                hist_data = self.monthly_data[
                    (self.monthly_data['month_year'] == hist_period) & 
                    (self.monthly_data['sku'] == sku)
                ]
                
                if not hist_data.empty:
                    qty = hist_data.iloc[0]['quantity']
                    dispatches = hist_data.iloc[0]['dispatch_count']
                else:
                    qty = 0
                    dispatches = 0
                
                feature_row[f'qty_lag_{j}'] = qty
                feature_row[f'dispatch_lag_{j}'] = dispatches
                historical_quantities.append(qty)
                historical_dispatches.append(dispatches)
            
            # Rolling statistics
            feature_row['qty_mean'] = np.mean(historical_quantities)
            feature_row['qty_std'] = np.std(historical_quantities)
            feature_row['qty_max'] = np.max(historical_quantities)
            feature_row['qty_min'] = np.min(historical_quantities)
            feature_row['qty_trend'] = np.polyfit(range(len(historical_quantities)), historical_quantities, 1)[0]
            
            feature_row['dispatch_mean'] = np.mean(historical_dispatches)
            feature_row['dispatch_std'] = np.std(historical_dispatches)
            
            # Category-based features
            category_data = self.monthly_data[
                (self.monthly_data['category'] == feature_row['category']) &
                (self.monthly_data['month_year'].isin(recent_periods))
            ]
            feature_row['category_avg_qty'] = category_data['quantity'].mean() if not category_data.empty else 0
            
            # Seasonal features
            same_month_data = self.monthly_data[
                (self.monthly_data['month'] == feature_row['month']) &
                (self.monthly_data['sku'] == sku)
            ]
            feature_row['seasonal_avg'] = same_month_data['quantity'].mean() if not same_month_data.empty else 0
            
            prediction_features.append(feature_row)
        
        # Create DataFrame and encode categorical variables
        pred_df = pd.DataFrame(prediction_features)
        
        for col in ['season', 'category', 'item']:
            if col in pred_df.columns and col in self.label_encoders:
                # Handle unseen categories
                le = self.label_encoders[col]
                pred_df[col] = pred_df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Prepare features for prediction
        X_pred = pred_df[self.feature_columns].fillna(0)
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Make predictions
        pred_proba = self.model.predict_proba(X_pred_scaled)[:, 1]
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'sku': pred_df['sku'],
            'prediction_score': pred_proba
        })
        
        # Get top N predictions
        top_predictions = results_df.nlargest(self.top_n, 'prediction_score')
        
        # Add item and category info
        sku_info = self.monthly_data[['sku', 'item', 'category']].drop_duplicates()
        top_predictions = top_predictions.merge(sku_info, on='sku', how='left')
        
        return top_predictions.to_dict('records')
    
    def generate_monthly_trends_plot(self):
        """Generate monthly trends plot and return as bytes"""
        # Get top SKUs overall
        top_skus_overall = self.monthly_data.groupby('sku')['quantity'].sum().nlargest(10).index
        
        # Filter data for top SKUs
        trend_data = self.monthly_data[self.monthly_data['sku'].isin(top_skus_overall)]
        
        # Create the plot
        plt.figure(figsize=(15, 10))
        
        for i, sku in enumerate(top_skus_overall):
            plt.subplot(2, 5, i + 1)
            sku_data = trend_data[trend_data['sku'] == sku]
            sku_data = sku_data.sort_values('month_year')
            
            plt.plot(sku_data['month_year'].dt.to_timestamp(), sku_data['quantity'], marker='o')
            plt.title(f'{sku}')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer.getvalue()
    
    def generate_seasonal_patterns_plot(self):
        """Generate seasonal patterns plot and return as bytes"""
        seasonal_data = self.monthly_data.groupby(['season', 'category'])['quantity'].sum().reset_index()
        
        plt.figure(figsize=(12, 8))
        
        # Seasonal patterns by category
        plt.subplot(2, 2, 1)
        seasonal_pivot = seasonal_data.pivot(index='season', columns='category', values='quantity')
        sns.heatmap(seasonal_pivot, annot=True, fmt='.0f', cmap='YlOrRd')
        plt.title('Seasonal Patterns by Category')
        
        # Monthly patterns
        plt.subplot(2, 2, 2)
        monthly_pattern = self.monthly_data.groupby('month')['quantity'].sum()
        monthly_pattern.plot(kind='bar')
        plt.title('Monthly Demand Patterns')
        plt.xlabel('Month')
        plt.ylabel('Total Quantity')
        
        # Category distribution
        plt.subplot(2, 2, 3)
        category_dist = self.monthly_data.groupby('category')['quantity'].sum().sort_values(ascending=False)
        category_dist.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Category Distribution')
        
        # Top SKUs by quantity
        plt.subplot(2, 2, 4)
        top_skus = self.monthly_data.groupby('sku')['quantity'].sum().nlargest(10)
        top_skus.plot(kind='barh')
        plt.title('Top 10 SKUs by Total Quantity')
        plt.xlabel('Total Quantity')
        
        plt.tight_layout()
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer.getvalue()

def upload_to_supabase(file_content: bytes, file_name: str) -> str:
    """Upload file to Supabase storage and return public URL."""
    try:
        # First, try to remove the file if it exists (for upsert behavior)
        try:
            supabase.storage.from_(BUCKET_NAME).remove([file_name])
        except:
            pass  # File might not exist, which is fine
        
        # Upload file to Supabase storage
        response = supabase.storage.from_(BUCKET_NAME).upload(
            file_name,
            file_content,
            file_options={
                "content-type": "image/png"
            }
        )

        # Check if upload was successful
        if hasattr(response, 'error') and response.error:
            raise Exception(f"Upload failed: {response.error}")

        # Get public URL
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_name)
        return public_url

    except Exception as e:
        logger.error(f"Error uploading to Supabase: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload to Supabase: {str(e)}")


async def process_ml_pipeline(df: pd.DataFrame) -> Dict[str, Any]:
    """Process the ML pipeline and return results"""
    try:
        # Initialize predictor
        predictor = SKUDemandPredictor(top_n=10, lookback_months=3)
        
        # Load and preprocess data
        monthly_data = predictor.load_and_preprocess_data(df)
        
        # Create features
        features = predictor.create_features()
        
        # Train model
        predictor.train_model(model_type='rf', test_size=0.2)
        
        # Generate predictions
        predictions = predictor.predict_next_month()
        
        # Generate visualizations
        monthly_trends_img = predictor.generate_monthly_trends_plot()
        seasonal_patterns_img = predictor.generate_seasonal_patterns_plot()
        
        # Upload images to Supabase
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trends_filename = f"monthly_trends_{timestamp}.png"
        seasonal_filename = f"seasonal_patterns_{timestamp}.png"
        
        trends_url = upload_to_supabase(monthly_trends_img, trends_filename)
        seasonal_url = upload_to_supabase(seasonal_patterns_img, seasonal_filename)
        
        # Fix datetime handling for data_summary
        # Ensure dates are datetime objects
        df['date'] = pd.to_datetime(df['date'])
        
        # Get min and max dates safely
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        # Format dates safely
        if pd.isna(min_date) or pd.isna(max_date):
            date_range = {"start": "N/A", "end": "N/A"}
        else:
            date_range = {
                "start": min_date.strftime('%Y-%m-%d'),
                "end": max_date.strftime('%Y-%m-%d')
            }
        
        return {
            "status": "success",
            "message": "ML pipeline completed successfully",
            "predictions": predictions,
            "visualizations": {
                "monthly_trends": trends_url,
                "seasonal_patterns": seasonal_url
            },
            "data_summary": {
                "total_records": len(df),
                "date_range": date_range,
                "unique_skus": df['sku'].nunique(),
                "total_quantity": int(df['quantity'].sum())
            }
        }
        
    except Exception as e:
        logger.error(f"Error in ML pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ML pipeline failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "SKU Demand Predictor API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict")
async def predict_demand(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload CSV file and run ML pipeline
    
    Expected CSV columns: date, sku, item, category, quantity, location
    """
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV file with better error handling
        contents = await file.read()
        
        # Try different encodings
        try:
            content_str = contents.decode('utf-8')
        except UnicodeDecodeError:
            try:
                content_str = contents.decode('utf-8-sig')
            except UnicodeDecodeError:
                content_str = contents.decode('latin-1')
        
        # Try to parse CSV with different parameters
        try:
            df = pd.read_csv(io.StringIO(content_str))
        except pd.errors.ParserError:
            try:
                df = pd.read_csv(io.StringIO(content_str), sep=';')
            except:
                df = pd.read_csv(io.StringIO(content_str), sep=',', quotechar='"', skipinitialspace=True)
        
        # Validate that we have data
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Validate required columns
        required_columns = ['date', 'sku', 'item', 'category', 'quantity', 'location']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Show available columns to help debug
            available_columns = list(df.columns)
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}. Available columns: {available_columns}"
            )
        
        # Process ML pipeline
        result = await process_ml_pipeline(df)
        
        return JSONResponse(content=result)
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict-from-url")
async def predict_from_url(
    background_tasks: BackgroundTasks,
    file_url: str
):
    """
    Download CSV from URL and run ML pipeline
    
    Args:
        file_url: URL to CSV file (supports Google Drive links)
    """
    try:
        # Convert Google Drive URL if necessary
        download_url = convert_google_drive_url(file_url)
        
        # Download file from URL with proper headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(download_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Check if we got HTML instead of CSV (happens with Google Drive sometimes)
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' in content_type:
            raise HTTPException(
                status_code=400, 
                detail="Received HTML instead of CSV. Please ensure the Google Drive file is publicly accessible and use a direct download link."
            )
        
        # Try to read CSV with different encodings
        try:
            df = pd.read_csv(io.StringIO(response.text))
        except UnicodeDecodeError:
            # Try with different encoding
            df = pd.read_csv(io.StringIO(response.content.decode('utf-8-sig')))
        except pd.errors.ParserError:
            # Try with different separator
            try:
                df = pd.read_csv(io.StringIO(response.text), sep=';')
            except:
                # Try with different parameters
                df = pd.read_csv(io.StringIO(response.text), sep=',', quotechar='"', skipinitialspace=True)
        
        # Validate that we have data
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Validate required columns
        required_columns = ['date', 'sku', 'item', 'category', 'quantity', 'location']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            # Show available columns to help debug
            available_columns = list(df.columns)
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}. Available columns: {available_columns}"
            )
        
        # Process ML pipeline
        result = await process_ml_pipeline(df)
        
        return JSONResponse(content=result)
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/storage-test")
async def test_storage():
    """Test Supabase storage connection"""
    try:
        # Create a simple test image
        plt.figure(figsize=(8, 6))
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
        plt.title("Test Plot")
        plt.xlabel("X")
        plt.ylabel("Y")
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150)
        img_buffer.seek(0)
        plt.close()
        
        # Upload to Supabase
        test_filename = f"test_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        url = upload_to_supabase(img_buffer.getvalue(), test_filename)
        
        return {"message": "Storage test successful", "url": url}
        
    except Exception as e:
        logger.error(f"Storage test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Storage test failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
