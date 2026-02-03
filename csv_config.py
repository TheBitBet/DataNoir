"""
CSV Configuration Module
Handles flexible CSV import with smart column detection and mapping
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd


class CSVConfig:
    """Manages CSV configurations and column mappings"""
    
    # Standard column names we need
    REQUIRED_COLUMNS = ['Date', 'Price', 'Volume']
    
    # Common column name variations for auto-detection
    COLUMN_PATTERNS = {
        'Date': [
            'date', 'time', 'datetime', 'timestamp', 'day', 
            'trading_date', 'trade_date', 'dt', 'data'
        ],
        'Price': [
            'price', 'close', 'closing', 'last', 'value',
            'closing_price', 'close_price', 'adj_close', 
            'adjusted_close', 'preco', 'ultimo'
        ],
        'Volume': [
            'volume', 'vol', 'vol.', 'quantity', 'qty',
            'trading_volume', 'trade_volume', 'quantidade',
            'vol', 'volume_total'
        ],
        # Optional columns
        'Open': [
            'open', 'opening', 'opening_price', 'open_price',
            'first', 'abertura'
        ],
        'High': [
            'high', 'max', 'maximum', 'top', 'highest',
            'high_price', 'maxima', 'maximo'
        ],
        'Low': [
            'low', 'min', 'minimum', 'bottom', 'lowest',
            'low_price', 'minima', 'minimo'
        ],
        'Change': [
            'change', 'change%', 'change_percent', 'pct_change',
            'variacao', 'var%', 'diff'
        ]
    }
    
    # Predefined configurations for common data sources
    PRESETS = {
        'Yahoo Finance': {
            'Date': 'Date',
            'Price': 'Close',
            'Volume': 'Volume',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low'
        },
        'Investing.com': {
            'Date': 'Date',
            'Price': 'Price',
            'Volume': 'Vol.',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Change': 'Change %'
        },
        'Google Finance': {
            'Date': 'Date',
            'Price': 'Close',
            'Volume': 'Volume',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low'
        },
        'Generic': {
            'Date': 'Date',
            'Price': 'Price',
            'Volume': 'Volume'
        }
    }
    
    def __init__(self, config_file: str = 'csv_configs.json'):
        """
        Initialize CSV Config manager
        
        Args:
            config_file: Path to JSON file storing custom configurations
        """
        self.config_file = config_file
        self.custom_configs = self._load_custom_configs()
    
    def _load_custom_configs(self) -> Dict:
        """Load custom configurations from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def save_custom_config(self, name: str, mapping: Dict[str, str]):
        """
        Save a custom configuration
        
        Args:
            name: Configuration name
            mapping: Column mapping dictionary
        """
        self.custom_configs[name] = mapping
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.custom_configs, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")
    
    def get_all_presets(self) -> Dict[str, Dict]:
        """Get all available presets (built-in + custom)"""
        all_presets = self.PRESETS.copy()
        all_presets.update(self.custom_configs)
        return all_presets
    
    def detect_columns(self, df: pd.DataFrame) -> Tuple[Dict[str, str], List[str]]:
        """
        Automatically detect column mappings
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Tuple of (detected_mapping, unmatched_required_columns)
        """
        detected = {}
        available_columns = [col.strip() for col in df.columns]
        
        # Try to match each required column
        for standard_name, patterns in self.COLUMN_PATTERNS.items():
            match = self._find_matching_column(available_columns, patterns)
            if match:
                detected[standard_name] = match
        
        # Check which required columns are missing
        unmatched = [col for col in self.REQUIRED_COLUMNS if col not in detected]
        
        return detected, unmatched
    
    def _find_matching_column(self, available_columns: List[str], 
                             patterns: List[str]) -> Optional[str]:
        """
        Find a matching column from available columns
        
        Args:
            available_columns: List of column names in the CSV
            patterns: List of patterns to match against
            
        Returns:
            Matching column name or None
        """
        # Try exact match first (case-insensitive)
        for col in available_columns:
            col_lower = col.lower().strip()
            for pattern in patterns:
                if col_lower == pattern.lower():
                    return col
        
        # Try partial match
        for col in available_columns:
            col_lower = col.lower().strip()
            for pattern in patterns:
                if pattern.lower() in col_lower or col_lower in pattern.lower():
                    return col
        
        return None
    
    def apply_preset(self, preset_name: str) -> Optional[Dict[str, str]]:
        """
        Get a preset configuration
        
        Args:
            preset_name: Name of the preset
            
        Returns:
            Column mapping dictionary or None
        """
        all_presets = self.get_all_presets()
        return all_presets.get(preset_name)
    
    def validate_mapping(self, mapping: Dict[str, str], 
                        df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that a mapping is valid for the dataframe
        
        Args:
            mapping: Column mapping to validate
            df: DataFrame to validate against
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required columns are mapped
        for required in self.REQUIRED_COLUMNS:
            if required not in mapping:
                errors.append(f"Missing required column: {required}")
            elif mapping[required] not in df.columns:
                errors.append(f"Mapped column '{mapping[required]}' not found in CSV")
        
        return len(errors) == 0, errors
    
    def create_mapping_from_columns(self, csv_columns: List[str],
                                   user_selections: Dict[str, str]) -> Dict[str, str]:
        """
        Create a mapping from user selections
        
        Args:
            csv_columns: Available columns in CSV
            user_selections: User's column selections
            
        Returns:
            Column mapping dictionary
        """
        mapping = {}
        for standard_name, selected_col in user_selections.items():
            if selected_col and selected_col != 'None':
                mapping[standard_name] = selected_col
        return mapping
    
    def get_column_info(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Get detailed information about each column
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with column info (type, sample values, etc.)
        """
        info = {}
        for col in df.columns:
            sample_values = df[col].head(5).tolist()
            info[col] = {
                'dtype': str(df[col].dtype),
                'non_null': df[col].notna().sum(),
                'total': len(df),
                'sample': sample_values,
                'has_dates': self._looks_like_date(df[col]),
                'has_numbers': self._looks_like_number(df[col])
            }
        return info
    
    def _looks_like_date(self, series: pd.Series) -> bool:
        """Check if a series looks like dates"""
        try:
            # Try to parse first few non-null values as dates
            sample = series.dropna().head(10)
            if len(sample) == 0:
                return False
            pd.to_datetime(sample, errors='coerce')
            return True
        except:
            return False
    
    def _looks_like_number(self, series: pd.Series) -> bool:
        """Check if a series looks like numbers"""
        try:
            sample = series.dropna().head(10)
            if len(sample) == 0:
                return False
            # Try to convert to numeric (handling strings with commas, symbols, etc.)
            sample_str = sample.astype(str).str.replace(',', '').str.replace('$', '')
            pd.to_numeric(sample_str, errors='coerce')
            return True
        except:
            return False
    
    def suggest_mapping(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Suggest possible columns for each required field
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping required columns to suggested candidates
        """
        suggestions = {}
        column_info = self.get_column_info(df)
        
        for standard_name in self.REQUIRED_COLUMNS:
            candidates = []
            
            # Get pattern matches
            patterns = self.COLUMN_PATTERNS.get(standard_name, [])
            for col in df.columns:
                col_lower = col.lower().strip()
                for pattern in patterns:
                    if pattern.lower() in col_lower or col_lower in pattern.lower():
                        if col not in candidates:
                            candidates.append(col)
                        break
            
            # Add columns based on data type
            if standard_name == 'Date':
                for col, info in column_info.items():
                    if info['has_dates'] and col not in candidates:
                        candidates.append(col)
            elif standard_name in ['Price', 'Volume']:
                for col, info in column_info.items():
                    if info['has_numbers'] and col not in candidates:
                        candidates.append(col)
            
            suggestions[standard_name] = candidates[:5]  # Top 5 suggestions
        
        return suggestions
