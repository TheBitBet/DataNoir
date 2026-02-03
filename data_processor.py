"""
Data Processor Module
Handles CSV loading, parsing, and technical indicator calculations
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict


class DataProcessor:
    """Processes financial CSV data and calculates technical indicators"""
    
    def __init__(self):
        self.df = None
        self.original_df = None
        self.column_mapping = None  # Stores the column mapping used
    
    @staticmethod
    def parse_volume(vol_str) -> float:
        """
        Parse volume string with K, M, B suffixes
        
        Args:
            vol_str: Volume string (e.g., "1.5M", "200K", "1.2B")
            
        Returns:
            float: Parsed volume value
        """
        try:
            vol_str = str(vol_str).strip()
            
            if vol_str == '-' or vol_str == '' or vol_str.lower() == 'nan':
                return 0.0
            
            vol_str = vol_str.replace(' ', '')
            multiplier = 1
            
            vol_str_upper = vol_str.upper()
            if 'K' in vol_str_upper:
                multiplier = 1_000
                vol_str = vol_str_upper.replace('K', '')
            elif 'M' in vol_str_upper:
                multiplier = 1_000_000
                vol_str = vol_str_upper.replace('M', '')
            elif 'B' in vol_str_upper:
                multiplier = 1_000_000_000
                vol_str = vol_str_upper.replace('B', '')
            
            vol_str = vol_str.replace(',', '').replace('%', '')
            return float(vol_str) * multiplier
            
        except (ValueError, AttributeError):
            return 0.0
    
    def load_csv(self, file_path: str, column_mapping: Optional[Dict[str, str]] = None) -> Tuple[bool, str, dict]:
        """
        Load and parse CSV file with flexible column mapping
        
        Args:
            file_path: Path to CSV file
            column_mapping: Optional dictionary mapping standard names to CSV columns
                          e.g., {'Date': 'Trading Date', 'Price': 'Close', 'Volume': 'Vol'}
            
        Returns:
            Tuple of (success: bool, message: str, stats: dict)
        """
        try:
            # Read CSV
            self.df = pd.read_csv(file_path)
            self.original_df = self.df.copy()
            
            # Clean column names
            self.df.columns = self.df.columns.str.replace('"', '').str.strip()
            
            # If no mapping provided, assume standard column names
            if column_mapping is None:
                column_mapping = {
                    'Date': 'Date',
                    'Price': 'Price', 
                    'Volume': 'Vol.'
                }
            
            # Store the mapping
            self.column_mapping = column_mapping
            
            # Check if mapped columns exist
            missing_cols = []
            for standard_name, csv_column in column_mapping.items():
                if csv_column not in self.df.columns:
                    missing_cols.append(f"{standard_name} (mapped to '{csv_column}')")
            
            if missing_cols:
                return False, f"Mapped columns not found in CSV: {', '.join(missing_cols)}", {}
            
            # Rename columns to standard names
            rename_dict = {csv_col: std_name for std_name, csv_col in column_mapping.items()}
            self.df = self.df.rename(columns=rename_dict)
            
            # Ensure we have the required columns after mapping
            required_cols = ['Date', 'Price', 'Volume']
            missing_required = [col for col in required_cols if col not in self.df.columns]
            if missing_required:
                return False, f"Required columns missing after mapping: {', '.join(missing_required)}", {}
            
            # Parse columns
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df['Price'] = self.df['Price'].astype(str).str.replace(',', '').str.replace('$', '').astype(float)
            self.df['Volume'] = self.df['Volume'].apply(self.parse_volume)
            
            # Sort by date descending (newest first)
            self.df = self.df.sort_values('Date', ascending=False)
            
            # Remove rows with NaN
            original_len = len(self.df)
            self.df = self.df.dropna(subset=['Date', 'Price', 'Volume'])
            removed_rows = original_len - len(self.df)
            
            if len(self.df) == 0:
                return False, "No valid data rows after parsing", {}
            
            # Calculate statistics
            stats = {
                'total_records': len(self.df),
                'removed_rows': removed_rows,
                'date_min': self.df['Date'].min(),
                'date_max': self.df['Date'].max(),
                'price_min': self.df['Price'].min(),
                'price_max': self.df['Price'].max(),
                'price_mean': self.df['Price'].mean(),
                'volume_min': self.df['Volume'].min(),
                'volume_max': self.df['Volume'].max(),
                'volume_mean': self.df['Volume'].mean(),
                'columns': list(self.df.columns),
                'original_columns': list(self.original_df.columns),
                'mapping_used': self.column_mapping
            }
            
            return True, "CSV loaded successfully", stats
            
        except Exception as e:
            return False, f"Error loading CSV: {str(e)}", {}
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """Get the processed dataframe"""
        return self.df
    
    def calculate_moving_average(self, window: int) -> pd.Series:
        """
        Calculate simple moving average
        
        Args:
            window: Window size for moving average
            
        Returns:
            Series with moving average values
        """
        if self.df is None:
            return pd.Series()
        return self.df['Price'].rolling(window=window, min_periods=1).mean()
    
    def calculate_bollinger_bands(self, window: int = 20, num_std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            window: Window size for moving average
            num_std: Number of standard deviations for bands
            
        Returns:
            Tuple of (sma, upper_band, lower_band)
        """
        if self.df is None:
            return pd.Series(), pd.Series(), pd.Series()
        
        sma = self.df['Price'].rolling(window=window, min_periods=1).mean()
        std = self.df['Price'].rolling(window=window, min_periods=1).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return sma, upper_band, lower_band
    
    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            period: Period for RSI calculation
            
        Returns:
            Series with RSI values
        """
        if self.df is None:
            return pd.Series()
        
        delta = self.df['Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)
            
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        if self.df is None:
            return pd.Series(), pd.Series(), pd.Series()
        
        # Calculate EMAs
        ema_fast = self.df['Price'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['Price'].ewm(span=slow, adjust=False).mean()
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD line)
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Histogram (difference between MACD and Signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_seasonality(self) -> Dict:
        """
        Calculate seasonality patterns (by month, day of week, day of month)
        
        Returns:
            Dictionary with seasonality statistics
        """
        if self.df is None:
            return {}
        
        df = self.df.copy()
        
        # Add time-based columns
        df['Month'] = df['Date'].dt.month
        df['MonthName'] = df['Date'].dt.strftime('%B')
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfWeekName'] = df['Date'].dt.strftime('%A')
        df['DayOfMonth'] = df['Date'].dt.day
        df['Year'] = df['Date'].dt.year
        
        # Calculate price changes
        df['PriceChange'] = df['Price'].pct_change() * 100
        
        # Monthly statistics
        monthly_stats = df.groupby('Month').agg({
            'PriceChange': ['mean', 'std', 'count'],
            'Volume': 'mean',
            'Price': ['mean', 'min', 'max']
        }).round(2)
        
        # Day of week statistics
        dow_stats = df.groupby('DayOfWeek').agg({
            'PriceChange': ['mean', 'std', 'count'],
            'Volume': 'mean',
            'Price': 'mean'
        }).round(2)
        
        # Day of month statistics (grouped into weeks)
        dom_stats = df.groupby('DayOfMonth').agg({
            'PriceChange': ['mean', 'count'],
            'Volume': 'mean'
        }).round(2)
        
        return {
            'monthly': monthly_stats,
            'day_of_week': dow_stats,
            'day_of_month': dom_stats,
            'raw_data': df,
            'month_names': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            'dow_names': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                         'Friday', 'Saturday', 'Sunday']
        }
    
    def calculate_correlations(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Calculate correlations between Price, Volume, and derived metrics
        
        Returns:
            Tuple of (correlation_matrix, correlation_data)
        """
        if self.df is None:
            return pd.DataFrame(), {}
        
        df = self.df.copy()
        
        # Calculate derived metrics
        df['PriceChange'] = df['Price'].pct_change() * 100
        df['VolumeChange'] = df['Volume'].pct_change() * 100
        df['MA7'] = df['Price'].rolling(window=7, min_periods=1).mean()
        df['MA30'] = df['Price'].rolling(window=30, min_periods=1).mean()
        df['Volatility'] = df['Price'].rolling(window=14, min_periods=1).std()
        
        # Select columns for correlation
        corr_columns = ['Price', 'Volume', 'PriceChange', 'VolumeChange', 
                       'MA7', 'MA30', 'Volatility']
        
        # Remove rows with NaN
        corr_df = df[corr_columns].dropna()
        
        # Calculate correlation matrix
        corr_matrix = corr_df.corr()
        
        # Additional statistics
        corr_data = {
            'matrix': corr_matrix,
            'columns': corr_columns,
            'strongest_positive': None,
            'strongest_negative': None,
            'data': corr_df
        }
        
        # Find strongest correlations (excluding diagonal)
        mask = np.ones_like(corr_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        
        # Strongest positive
        max_corr = corr_matrix.where(mask).max().max()
        max_idx = corr_matrix.where(mask).stack().idxmax()
        corr_data['strongest_positive'] = {
            'vars': max_idx,
            'value': max_corr
        }
        
        # Strongest negative
        min_corr = corr_matrix.where(mask).min().min()
        min_idx = corr_matrix.where(mask).stack().idxmin()
        corr_data['strongest_negative'] = {
            'vars': min_idx,
            'value': min_corr
        }
        
        return corr_matrix, corr_data
    
    def calculate_price_change(self) -> pd.Series:
        """
        Calculate daily price change percentage
        
        Returns:
            Series with price change percentages
        """
        if self.df is None:
            return pd.Series()
        
        return self.df['Price'].pct_change() * 100
    
    def calculate_cumulative_returns(self, price_change: pd.Series) -> pd.Series:
        """
        Calculate cumulative returns
        
        Args:
            price_change: Series with price change percentages
            
        Returns:
            Series with cumulative returns
        """
        return (1 + price_change / 100).cumprod() * 100 - 100
    
    def calculate_volume_profile(self, num_bins: int = 30) -> Tuple[list, list]:
        """
        Calculate volume profile (volume at each price level)
        
        Args:
            num_bins: Number of price bins
            
        Returns:
            Tuple of (bin_centers, volume_at_price)
        """
        if self.df is None:
            return [], []
        
        price_bins = np.linspace(self.df['Price'].min(), 
                                self.df['Price'].max(), 
                                num_bins)
        
        volume_at_price = []
        bin_centers = []
        
        for i in range(len(price_bins) - 1):
            mask = (self.df['Price'] >= price_bins[i]) & (self.df['Price'] < price_bins[i + 1])
            vol = self.df.loc[mask, 'Volume'].sum()
            volume_at_price.append(vol)
            bin_centers.append((price_bins[i] + price_bins[i + 1]) / 2)
        
        return bin_centers, volume_at_price
    
    def get_point_of_control(self, bin_centers: list, volume_at_price: list) -> float:
        """
        Get Point of Control (price with highest volume)
        
        Args:
            bin_centers: List of price bin centers
            volume_at_price: List of volumes at each price
            
        Returns:
            Price with highest volume (POC)
        """
        if not volume_at_price:
            return 0.0
        
        max_vol_idx = np.argmax(volume_at_price)
        return bin_centers[max_vol_idx]
    
    def get_statistics_summary(self) -> dict:
        """
        Get comprehensive statistics summary
        
        Returns:
            Dictionary with various statistics
        """
        if self.df is None:
            return {}
        
        price_change = self.calculate_price_change()
        
        return {
            'price': {
                'min': self.df['Price'].min(),
                'max': self.df['Price'].max(),
                'mean': self.df['Price'].mean(),
                'std': self.df['Price'].std()
            },
            'volume': {
                'min': self.df['Volume'].min(),
                'max': self.df['Volume'].max(),
                'mean': self.df['Volume'].mean(),
                'std': self.df['Volume'].std()
            },
            'change': {
                'avg_daily': price_change.mean(),
                'max_gain': price_change.max(),
                'max_loss': price_change.min(),
                'volatility': price_change.std()
            },
            'date_range': {
                'start': self.df['Date'].min(),
                'end': self.df['Date'].max(),
                'days': (self.df['Date'].max() - self.df['Date'].min()).days
            }
        }
