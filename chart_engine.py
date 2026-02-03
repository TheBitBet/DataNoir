"""
Chart Engine Module
Handles all visualization and chart rendering
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from typing import Optional, Callable


class ChartEngine:
    """Handles creation of various financial charts and visualizations"""
    
    def __init__(self, data_processor):
        """
        Initialize ChartEngine
        
        Args:
            data_processor: DataProcessor instance with loaded data
        """
        self.data_processor = data_processor
        self.logger = None
    
    def set_logger(self, logger_func: Callable):
        """
        Set logging function
        
        Args:
            logger_func: Function to call for logging messages
        """
        self.logger = logger_func
    
    def _log(self, message: str):
        """Internal logging helper"""
        if self.logger:
            self.logger(message)
    
    def create_3d_chart(self, scale_type: str = "linear", 
                       color_by: str = "price", 
                       point_size: int = 50):
        """
        Create 3D visualization of Date, Volume, and Price
        
        Args:
            scale_type: "linear", "log", or "normalized"
            color_by: "price", "volume", or "date"
            point_size: Size of scatter points
        """
        df = self.data_processor.get_data()
        if df is None or len(df) == 0:
            self._log("ERROR: No data available for 3D chart")
            return
        
        try:
            self._log(f"Creating 3D chart (scale={scale_type}, color={color_by}, size={point_size})...")
            
            # Prepare data
            date_numeric = np.arange(len(df))
            volumes = df['Volume'].values
            prices = df['Price'].values
            
            # Apply scaling to volume
            if scale_type == "log":
                volumes_plot = np.log10(volumes + 1)
                ylabel = "Volume (log scale)"
                self._log("Applied logarithmic scaling")
            elif scale_type == "normalized":
                vol_min = volumes.min()
                vol_max = volumes.max()
                if vol_max > vol_min:
                    volumes_plot = (volumes - vol_min) / (vol_max - vol_min) * 100
                else:
                    volumes_plot = volumes
                ylabel = "Volume (normalized 0-100)"
                self._log("Applied normalization")
            else:
                volumes_plot = volumes
                ylabel = "Volume"
            
            # Determine color values
            if color_by == "price":
                color_values = prices
                color_label = "Price"
            elif color_by == "volume":
                color_values = volumes
                color_label = "Volume"
            else:  # date
                color_values = date_numeric
                color_label = "Time"
            
            # Create the 3D plot
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot the data
            scatter = ax.scatter(date_numeric, volumes_plot, prices,
                                c=color_values, cmap='viridis', 
                                marker='o', s=point_size,
                                alpha=0.6, edgecolors='w', linewidth=0.5)
            
            # Add connecting line
            ax.plot(date_numeric, volumes_plot, prices, 
                    color='blue', alpha=0.2, linewidth=1)
            
            # Set labels
            ax.set_xlabel('Date Index', fontsize=12, labelpad=10)
            ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
            ax.set_zlabel('Price', fontsize=12, labelpad=10)
            
            title = f'3D Visualization: Date vs Volume vs Price\n(Scale: {scale_type}, Color: {color_by})'
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
            cbar.set_label(color_label, rotation=270, labelpad=15)
            
            # Customize date labels
            num_ticks = min(10, len(df))
            tick_indices = np.linspace(0, len(df)-1, num_ticks, dtype=int)
            ax.set_xticks(date_numeric[tick_indices])
            ax.set_xticklabels([df.iloc[i]['Date'].strftime('%Y-%m-%d') 
                                for i in tick_indices], 
                               rotation=45, ha='right', fontsize=8)
            
            # Set view angle
            ax.view_init(elev=20, azim=45)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            self._log("✓ 3D chart created successfully!")
            plt.show()
            
        except Exception as e:
            self._log(f"ERROR: Failed to create 3D chart - {str(e)}")
            raise
    
    def create_basic_2d_charts(self):
        """Create basic 2D analysis charts"""
        df = self.data_processor.get_data()
        if df is None or len(df) == 0:
            self._log("ERROR: No data available")
            return
        
        try:
            self._log("Creating basic 2D charts...")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Data Analysis - 2D Charts', fontsize=16, fontweight='bold')
            
            # Chart 1: Price over time
            ax1 = axes[0, 0]
            ax1.plot(df['Date'], df['Price'], color='blue', linewidth=2)
            ax1.set_xlabel('Date', fontsize=10)
            ax1.set_ylabel('Price', fontsize=10)
            ax1.set_title('Price Over Time', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Chart 2: Volume over time
            ax2 = axes[0, 1]
            ax2.plot(df['Date'], df['Volume'], color='green', linewidth=2)
            ax2.set_xlabel('Date', fontsize=10)
            ax2.set_ylabel('Volume', fontsize=10)
            ax2.set_title('Volume Over Time', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # Chart 3: Price vs Volume scatter
            ax3 = axes[1, 0]
            scatter = ax3.scatter(df['Volume'], df['Price'], 
                                 c=range(len(df)), cmap='viridis', alpha=0.6)
            ax3.set_xlabel('Volume', fontsize=10)
            ax3.set_ylabel('Price', fontsize=10)
            ax3.set_title('Price vs Volume (colored by time)', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax3, label='Time progression')
            
            # Chart 4: Distribution histograms
            ax4 = axes[1, 1]
            ax4_twin = ax4.twinx()
            ax4.hist(df['Price'], bins=30, alpha=0.7, color='blue', label='Price')
            ax4_twin.hist(df['Volume'], bins=30, alpha=0.7, color='green', label='Volume')
            ax4.set_xlabel('Value', fontsize=10)
            ax4.set_ylabel('Price Frequency', fontsize=10, color='blue')
            ax4_twin.set_ylabel('Volume Frequency', fontsize=10, color='green')
            ax4.set_title('Distribution of Values', fontweight='bold')
            ax4.legend(loc='upper left')
            ax4_twin.legend(loc='upper right')
            
            plt.tight_layout()
            self._log("✓ Basic 2D charts created!")
            plt.show()
            
        except Exception as e:
            self._log(f"ERROR: Failed to create 2D charts - {str(e)}")
            raise
    
    def create_moving_average_chart(self):
        """Create Moving Averages and Bollinger Bands chart"""
        df = self.data_processor.get_data()
        if df is None or len(df) == 0:
            self._log("ERROR: No data available")
            return
        
        try:
            self._log("Creating Moving Averages & Bollinger Bands chart...")
            
            # Calculate indicators
            ma7 = self.data_processor.calculate_moving_average(7)
            ma30 = self.data_processor.calculate_moving_average(30)
            ma90 = self.data_processor.calculate_moving_average(90)
            sma20, upper_bb, lower_bb = self.data_processor.calculate_bollinger_bands()
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot price and moving averages
            ax.plot(df['Date'], df['Price'], 
                   label='Price', color='black', linewidth=2, alpha=0.7)
            ax.plot(df['Date'], ma7, 
                   label='MA 7-day', color='blue', linewidth=1.5, alpha=0.7)
            ax.plot(df['Date'], ma30, 
                   label='MA 30-day', color='orange', linewidth=1.5, alpha=0.7)
            ax.plot(df['Date'], ma90, 
                   label='MA 90-day', color='red', linewidth=1.5, alpha=0.7)
            
            # Plot Bollinger Bands
            ax.plot(df['Date'], upper_bb, 
                   label='Upper BB', color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.plot(df['Date'], lower_bb, 
                   label='Lower BB', color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.fill_between(df['Date'], lower_bb, upper_bb, 
                           color='gray', alpha=0.1)
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            ax.set_title('Moving Averages & Bollinger Bands', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            self._log("✓ Moving Averages chart created!")
            plt.show()
            
        except Exception as e:
            self._log(f"ERROR: Failed to create Moving Averages chart - {str(e)}")
            raise
    
    def create_rsi_chart(self):
        """Create RSI (Relative Strength Index) chart"""
        df = self.data_processor.get_data()
        if df is None or len(df) == 0:
            self._log("ERROR: No data available")
            return
        
        try:
            self._log("Creating RSI chart...")
            
            rsi = self.data_processor.calculate_rsi()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                          gridspec_kw={'height_ratios': [2, 1]})
            
            # Top plot: Price
            ax1.plot(df['Date'], df['Price'], 
                    color='black', linewidth=2, label='Price')
            ax1.set_ylabel('Price', fontsize=12)
            ax1.set_title('Price and RSI (Relative Strength Index)', 
                         fontsize=14, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: RSI
            ax2.plot(df['Date'], rsi, 
                    color='purple', linewidth=2, label='RSI')
            ax2.axhline(y=70, color='r', linestyle='--', linewidth=1, 
                       label='Overbought (70)', alpha=0.7)
            ax2.axhline(y=30, color='g', linestyle='--', linewidth=1, 
                       label='Oversold (30)', alpha=0.7)
            ax2.fill_between(df['Date'], 30, 70, color='gray', alpha=0.1)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('RSI', fontsize=12)
            ax2.set_ylim(0, 100)
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            self._log("✓ RSI chart created!")
            self._log("  RSI > 70: Overbought (potential sell signal)")
            self._log("  RSI < 30: Oversold (potential buy signal)")
            plt.show()
            
        except Exception as e:
            self._log(f"ERROR: Failed to create RSI chart - {str(e)}")
            raise
    
    def create_price_change_chart(self):
        """Create Price Change % analysis chart"""
        df = self.data_processor.get_data()
        if df is None or len(df) == 0:
            self._log("ERROR: No data available")
            return
        
        try:
            self._log("Creating Price Change % chart...")
            
            price_change = self.data_processor.calculate_price_change()
            cumulative_returns = self.data_processor.calculate_cumulative_returns(price_change)
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
            
            # Chart 1: Price
            ax1.plot(df['Date'], df['Price'], color='black', linewidth=2)
            ax1.set_ylabel('Price', fontsize=12)
            ax1.set_title('Price Change % Analysis', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Chart 2: Daily % change
            colors = ['green' if x > 0 else 'red' for x in price_change]
            ax2.bar(df['Date'], price_change, color=colors, alpha=0.6, width=1)
            ax2.axhline(y=0, color='black', linewidth=1)
            ax2.set_ylabel('Daily Change %', fontsize=12)
            ax2.set_title('Daily Price Change %', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Chart 3: Cumulative returns
            ax3.plot(df['Date'], cumulative_returns, color='blue', linewidth=2)
            ax3.axhline(y=0, color='black', linewidth=1, linestyle='--')
            ax3.fill_between(df['Date'], 0, cumulative_returns, 
                            where=(cumulative_returns >= 0), 
                            color='green', alpha=0.3, label='Profit')
            ax3.fill_between(df['Date'], 0, cumulative_returns, 
                            where=(cumulative_returns < 0), 
                            color='red', alpha=0.3, label='Loss')
            ax3.set_xlabel('Date', fontsize=12)
            ax3.set_ylabel('Cumulative Return %', fontsize=12)
            ax3.set_title('Cumulative Returns', fontweight='bold')
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Log statistics
            self._log("✓ Price Change % chart created!")
            self._log(f"  Avg daily change: {price_change.mean():.2f}%")
            self._log(f"  Max gain: {price_change.max():.2f}%")
            self._log(f"  Max loss: {price_change.min():.2f}%")
            self._log(f"  Volatility (std): {price_change.std():.2f}%")
            
            plt.show()
            
        except Exception as e:
            self._log(f"ERROR: Failed to create Price Change chart - {str(e)}")
            raise
    
    def create_volume_profile_chart(self):
        """Create Volume Profile chart"""
        df = self.data_processor.get_data()
        if df is None or len(df) == 0:
            self._log("ERROR: No data available")
            return
        
        try:
            self._log("Creating Volume Profile chart...")
            
            bin_centers, volume_at_price = self.data_processor.calculate_volume_profile()
            poc_price = self.data_processor.get_point_of_control(bin_centers, volume_at_price)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), 
                                          gridspec_kw={'width_ratios': [3, 1]})
            
            # Left plot: Price over time
            ax1.plot(df['Date'], df['Price'], 
                    color='blue', linewidth=2, label='Price')
            ax1.axhline(y=poc_price, color='red', linestyle='--', 
                       linewidth=2, alpha=0.5, label=f'POC: ${poc_price:.2f}')
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Price', fontsize=12)
            ax1.set_title('Price with Volume Profile', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.tick_params(axis='x', rotation=45)
            ax1.legend(loc='best')
            
            # Right plot: Volume profile
            ax2.barh(bin_centers, volume_at_price, 
                    height=(bin_centers[1] - bin_centers[0]) * 0.9 if len(bin_centers) > 1 else 1,
                    color='green', alpha=0.6, edgecolor='darkgreen')
            ax2.axhline(y=poc_price, color='red', linestyle='--', 
                       linewidth=2, label=f'POC: ${poc_price:.2f}')
            ax2.set_xlabel('Total Volume', fontsize=12)
            ax2.set_ylabel('Price', fontsize=12)
            ax2.set_title('Volume Profile', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.legend(loc='best')
            
            plt.tight_layout()
            self._log("✓ Volume Profile chart created!")
            self._log(f"  Point of Control (POC): ${poc_price:.2f}")
            self._log("  POC = Price level with highest trading volume")
            plt.show()
            
        except Exception as e:
            self._log(f"ERROR: Failed to create Volume Profile chart - {str(e)}")
            raise
    
    def create_macd_chart(self):
        """Create MACD (Moving Average Convergence Divergence) chart"""
        df = self.data_processor.get_data()
        if df is None or len(df) == 0:
            self._log("ERROR: No data available")
            return
        
        try:
            self._log("Creating MACD chart...")
            
            macd_line, signal_line, histogram = self.data_processor.calculate_macd()
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                          gridspec_kw={'height_ratios': [2, 1]})
            
            # Top plot: Price
            ax1.plot(df['Date'], df['Price'], 
                    color='black', linewidth=2, label='Price')
            ax1.set_ylabel('Price', fontsize=12)
            ax1.set_title('Price and MACD (Moving Average Convergence Divergence)', 
                         fontsize=14, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: MACD
            # Plot histogram first (as bars)
            colors = ['green' if x >= 0 else 'red' for x in histogram]
            ax2.bar(df['Date'], histogram, color=colors, alpha=0.3, label='Histogram')
            
            # Plot MACD and Signal lines
            ax2.plot(df['Date'], macd_line, 
                    color='blue', linewidth=2, label='MACD Line')
            ax2.plot(df['Date'], signal_line, 
                    color='red', linewidth=2, label='Signal Line')
            
            # Zero line
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('MACD', fontsize=12)
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            self._log("✓ MACD chart created!")
            self._log("  MACD Line crosses above Signal = Bullish signal 📈")
            self._log("  MACD Line crosses below Signal = Bearish signal 📉")
            self._log("  Histogram > 0 = Bullish momentum")
            self._log("  Histogram < 0 = Bearish momentum")
            plt.show()
            
        except Exception as e:
            self._log(f"ERROR: Failed to create MACD chart - {str(e)}")
            raise
    
    def create_seasonality_chart(self):
        """Create Seasonality analysis chart"""
        df = self.data_processor.get_data()
        if df is None or len(df) == 0:
            self._log("ERROR: No data available")
            return
        
        try:
            self._log("Creating Seasonality chart...")
            
            seasonality = self.data_processor.calculate_seasonality()
            
            if not seasonality:
                self._log("ERROR: Could not calculate seasonality")
                return
            
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            # Chart 1: Average Returns by Month
            ax1 = fig.add_subplot(gs[0, :])
            monthly_returns = seasonality['monthly']['PriceChange']['mean']
            colors = ['green' if x > 0 else 'red' for x in monthly_returns.values]
            bars = ax1.bar(seasonality['month_names'], monthly_returns.values, 
                          color=colors, alpha=0.7, edgecolor='black')
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax1.set_ylabel('Average Return (%)', fontsize=11)
            ax1.set_title('Average Returns by Month', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%',
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=9)
            
            # Chart 2: Average Returns by Day of Week
            ax2 = fig.add_subplot(gs[1, 0])
            dow_returns = seasonality['day_of_week']['PriceChange']['mean']
            colors_dow = ['green' if x > 0 else 'red' for x in dow_returns.values]
            bars2 = ax2.bar(seasonality['dow_names'], dow_returns.values, 
                           color=colors_dow, alpha=0.7, edgecolor='black')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.set_ylabel('Average Return (%)', fontsize=11)
            ax2.set_title('Average Returns by Day of Week', fontsize=13, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}%',
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=8)
            
            # Chart 3: Average Volume by Month
            ax3 = fig.add_subplot(gs[1, 1])
            monthly_vol = seasonality['monthly']['Volume']['mean']
            ax3.bar(seasonality['month_names'], monthly_vol.values, 
                   color='steelblue', alpha=0.7, edgecolor='black')
            ax3.set_ylabel('Average Volume', fontsize=11)
            ax3.set_title('Average Volume by Month', fontsize=13, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.tick_params(axis='x', rotation=45)
            
            # Chart 4: Returns by Day of Month
            ax4 = fig.add_subplot(gs[2, 0])
            dom_returns = seasonality['day_of_month']['PriceChange']['mean']
            ax4.plot(dom_returns.index, dom_returns.values, 
                    marker='o', color='purple', linewidth=2, markersize=4)
            ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax4.set_xlabel('Day of Month', fontsize=11)
            ax4.set_ylabel('Average Return (%)', fontsize=11)
            ax4.set_title('Average Returns by Day of Month', fontsize=13, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(1, 31)
            
            # Chart 5: Heatmap of Returns by Year and Month
            ax5 = fig.add_subplot(gs[2, 1])
            raw_data = seasonality['raw_data']
            pivot_data = raw_data.pivot_table(
                values='PriceChange', 
                index='Year', 
                columns='Month', 
                aggfunc='mean'
            )
            
            if len(pivot_data) > 0:
                im = ax5.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', 
                               vmin=-5, vmax=5)
                ax5.set_yticks(range(len(pivot_data.index)))
                ax5.set_yticklabels(pivot_data.index)
                ax5.set_xticks(range(12))
                ax5.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 
                                    'J', 'A', 'S', 'O', 'N', 'D'])
                ax5.set_xlabel('Month', fontsize=11)
                ax5.set_ylabel('Year', fontsize=11)
                ax5.set_title('Return Heatmap (Year × Month)', fontsize=13, fontweight='bold')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax5)
                cbar.set_label('Return (%)', rotation=270, labelpad=15)
            else:
                ax5.text(0.5, 0.5, 'Insufficient data for heatmap', 
                        ha='center', va='center', transform=ax5.transAxes)
            
            fig.suptitle('Seasonality Analysis', fontsize=16, fontweight='bold', y=0.995)
            
            plt.tight_layout()
            
            # Log insights
            self._log("✓ Seasonality chart created!")
            best_month = monthly_returns.idxmax()
            worst_month = monthly_returns.idxmin()
            self._log(f"  Best month: {seasonality['month_names'][best_month-1]} ({monthly_returns[best_month]:.2f}%)")
            self._log(f"  Worst month: {seasonality['month_names'][worst_month-1]} ({monthly_returns[worst_month]:.2f}%)")
            
            best_dow = dow_returns.idxmax()
            worst_dow = dow_returns.idxmin()
            self._log(f"  Best day: {seasonality['dow_names'][best_dow]} ({dow_returns[best_dow]:.2f}%)")
            self._log(f"  Worst day: {seasonality['dow_names'][worst_dow]} ({dow_returns[worst_dow]:.2f}%)")
            
            plt.show()
            
        except Exception as e:
            self._log(f"ERROR: Failed to create Seasonality chart - {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            raise
    
    def create_correlation_heatmap(self):
        """Create Correlation Heatmap"""
        df = self.data_processor.get_data()
        if df is None or len(df) == 0:
            self._log("ERROR: No data available")
            return
        
        try:
            self._log("Creating Correlation Heatmap...")
            
            corr_matrix, corr_data = self.data_processor.calculate_correlations()
            
            if corr_matrix.empty:
                self._log("ERROR: Could not calculate correlations")
                return
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # Left plot: Heatmap
            im = ax1.imshow(corr_matrix.values, cmap='RdYlGn', aspect='auto', 
                           vmin=-1, vmax=1)
            
            # Set ticks and labels
            ax1.set_xticks(np.arange(len(corr_matrix.columns)))
            ax1.set_yticks(np.arange(len(corr_matrix.columns)))
            ax1.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax1.set_yticklabels(corr_matrix.columns)
            
            # Add correlation values as text
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    value = corr_matrix.iloc[i, j]
                    color = 'white' if abs(value) > 0.5 else 'black'
                    ax1.text(j, i, f'{value:.2f}',
                            ha="center", va="center", color=color, fontsize=9)
            
            ax1.set_title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax1)
            cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15)
            
            # Right plot: Scatter plots of strongest correlations
            if corr_data['strongest_positive'] and corr_data['strongest_negative']:
                # Top subplot: Strongest positive
                ax2_top = plt.subplot(2, 2, 2)
                pos_vars = corr_data['strongest_positive']['vars']
                pos_val = corr_data['strongest_positive']['value']
                
                data = corr_data['data']
                ax2_top.scatter(data[pos_vars[0]], data[pos_vars[1]], 
                               alpha=0.5, s=20, c='green')
                ax2_top.set_xlabel(pos_vars[0], fontsize=10)
                ax2_top.set_ylabel(pos_vars[1], fontsize=10)
                ax2_top.set_title(f'Strongest Positive: {pos_vars[0]} vs {pos_vars[1]}\n(r = {pos_val:.3f})', 
                                 fontsize=11, fontweight='bold')
                ax2_top.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(data[pos_vars[0]], data[pos_vars[1]], 1)
                p = np.poly1d(z)
                ax2_top.plot(data[pos_vars[0]], p(data[pos_vars[0]]), 
                            "r--", alpha=0.8, linewidth=2)
                
                # Bottom subplot: Strongest negative
                ax2_bottom = plt.subplot(2, 2, 4)
                neg_vars = corr_data['strongest_negative']['vars']
                neg_val = corr_data['strongest_negative']['value']
                
                ax2_bottom.scatter(data[neg_vars[0]], data[neg_vars[1]], 
                                  alpha=0.5, s=20, c='red')
                ax2_bottom.set_xlabel(neg_vars[0], fontsize=10)
                ax2_bottom.set_ylabel(neg_vars[1], fontsize=10)
                ax2_bottom.set_title(f'Strongest Negative: {neg_vars[0]} vs {neg_vars[1]}\n(r = {neg_val:.3f})', 
                                    fontsize=11, fontweight='bold')
                ax2_bottom.grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(data[neg_vars[0]], data[neg_vars[1]], 1)
                p = np.poly1d(z)
                ax2_bottom.plot(data[neg_vars[0]], p(data[neg_vars[0]]), 
                               "r--", alpha=0.8, linewidth=2)
            
            fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Log insights
            self._log("✓ Correlation Heatmap created!")
            if corr_data['strongest_positive']:
                pos = corr_data['strongest_positive']
                self._log(f"  Strongest positive: {pos['vars'][0]} ↔ {pos['vars'][1]} (r={pos['value']:.3f})")
            if corr_data['strongest_negative']:
                neg = corr_data['strongest_negative']
                self._log(f"  Strongest negative: {neg['vars'][0]} ↔ {neg['vars'][1]} (r={neg['value']:.3f})")
            
            plt.show()
            
        except Exception as e:
            self._log(f"ERROR: Failed to create Correlation Heatmap - {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            raise
