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
            ax1.set_xlabel('Date (newest → oldest)', fontsize=10)
            ax1.set_ylabel('Price', fontsize=10)
            ax1.set_title('Price Over Time', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Chart 2: Volume over time
            ax2 = axes[0, 1]
            ax2.plot(df['Date'], df['Volume'], color='green', linewidth=2)
            ax2.set_xlabel('Date (newest → oldest)', fontsize=10)
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
            
            ax.set_xlabel('Date (newest → oldest)', fontsize=12)
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
            ax2.set_xlabel('Date (newest → oldest)', fontsize=12)
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
            ax3.set_xlabel('Date (newest → oldest)', fontsize=12)
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
            ax1.set_xlabel('Date (newest → oldest)', fontsize=12)
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
            
            ax2.set_xlabel('Date (newest → oldest)', fontsize=12)
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
    
    def create_calendar_heatmap(self):
        """Create Calendar Heatmap showing daily returns"""
        df = self.data_processor.get_data()
        if df is None or len(df) == 0:
            self._log("ERROR: No data available")
            return
        
        try:
            self._log("Creating Calendar Heatmap...")
            
            calendar_data = self.data_processor.calculate_calendar_data()
            
            if not calendar_data:
                self._log("ERROR: Could not calculate calendar data")
                return
            
            data = calendar_data['data']
            years = calendar_data['years']
            
            # Create figure with subplots for each year
            num_years = len(years)
            fig_height = max(4, num_years * 3)
            fig = plt.figure(figsize=(16, fig_height))
            
            for idx, year in enumerate(years):
                year_data = data[data['Year'] == year].copy()
                
                if len(year_data) == 0:
                    continue
                
                # Create pivot table: weeks x days of week
                # Create a proper calendar structure
                year_data = year_data.sort_values('Date')
                
                # Create matrix for heatmap (53 weeks x 7 days)
                calendar_matrix = np.full((53, 7), np.nan)
                
                for _, row in year_data.iterrows():
                    week = int(row['WeekOfYear']) - 1  # 0-indexed
                    day = int(row['DayOfWeek'])
                    if 0 <= week < 53 and 0 <= day < 7:
                        calendar_matrix[week, day] = row['Return']
                
                # Create subplot
                ax = plt.subplot(num_years, 1, idx + 1)
                
                # Determine color scale
                vmax = max(abs(calendar_data['min_return']), abs(calendar_data['max_return']))
                vmax = min(vmax, 10)  # Cap at ±10% for better visualization
                
                # Create heatmap
                im = ax.imshow(calendar_matrix.T, cmap='RdYlGn', aspect='auto',
                              vmin=-vmax, vmax=vmax, interpolation='nearest')
                
                # Set labels
                ax.set_yticks(range(7))
                ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
                
                # Set x-axis (weeks/months)
                month_positions = []
                month_labels = []
                current_month = None
                
                for week_idx in range(53):
                    # Find first day of this week
                    week_data = year_data[year_data['WeekOfYear'] == week_idx + 1]
                    if len(week_data) > 0:
                        month = week_data.iloc[0]['Month']
                        if month != current_month:
                            month_positions.append(week_idx)
                            month_labels.append(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month - 1])
                            current_month = month
                
                ax.set_xticks(month_positions)
                ax.set_xticklabels(month_labels)
                ax.set_xlabel('Month', fontsize=10)
                ax.set_ylabel('Day of Week', fontsize=10)
                ax.set_title(f'{year} - Daily Returns Calendar', fontsize=12, fontweight='bold')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, aspect=30)
                cbar.set_label('Daily Return (%)', fontsize=9)
            
            fig.suptitle('Calendar Heatmap - Daily Returns Overview', 
                        fontsize=16, fontweight='bold', y=0.995)
            
            plt.tight_layout()
            
            # Log statistics
            self._log("✓ Calendar Heatmap created!")
            self._log(f"  Date range: {data['Date'].min().strftime('%Y-%m-%d')} to {data['Date'].max().strftime('%Y-%m-%d')}")
            self._log(f"  Positive days: {calendar_data['positive_days']} ({calendar_data['positive_days']/calendar_data['total_days']*100:.1f}%)")
            self._log(f"  Negative days: {calendar_data['negative_days']} ({calendar_data['negative_days']/calendar_data['total_days']*100:.1f}%)")
            self._log(f"  Avg daily return: {calendar_data['mean_return']:.2f}%")
            self._log(f"  Best day: {calendar_data['max_return']:.2f}%")
            self._log(f"  Worst day: {calendar_data['min_return']:.2f}%")
            
            plt.show()
            
        except Exception as e:
            self._log(f"ERROR: Failed to create Calendar Heatmap - {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            raise
    
    def create_price_forecast_chart(self, days_ahead: int = 30):
        """Create Price Forecast chart with multiple methods"""
        df = self.data_processor.get_data()
        if df is None or len(df) == 0:
            self._log("ERROR: No data available")
            return
        
        try:
            self._log(f"Creating Price Forecast ({days_ahead} days ahead)...")
            
            forecast_data = self.data_processor.calculate_price_forecast(days_ahead)
            
            if not forecast_data:
                self._log("ERROR: Could not calculate forecast")
                return
            
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 12))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            # Main chart: All forecasts together
            ax_main = fig.add_subplot(gs[0, :])
            
            hist_dates = forecast_data['historical_dates']
            hist_prices = forecast_data['historical_prices']
            future_dates = forecast_data['future_dates']
            
            # Plot historical prices
            ax_main.plot(hist_dates, hist_prices, 'k-', linewidth=2, label='Historical Price', alpha=0.7)
            
            # Colors for different methods
            colors = {'Linear Trend': '#2196F3', 'Exponential Smoothing': '#FF9800', 'Moving Average': '#4CAF50'}
            
            # Plot each forecast method
            for method_name, method_data in forecast_data['methods'].items():
                if 'forecast' in method_data:
                    color = colors.get(method_name, '#9C27B0')
                    
                    # Plot forecast line
                    ax_main.plot(future_dates, method_data['forecast'], 
                               linestyle='--', linewidth=2, color=color, 
                               label=f'{method_name} Forecast', alpha=0.8)
                    
                    # Plot confidence interval
                    ax_main.fill_between(future_dates, 
                                        method_data['confidence_lower'],
                                        method_data['confidence_upper'],
                                        color=color, alpha=0.15)
            
            # Vertical line at forecast start
            ax_main.axvline(x=hist_dates[-1], color='red', linestyle=':', alpha=0.5, linewidth=2)
            ax_main.text(hist_dates[-1], ax_main.get_ylim()[1], ' Forecast Start', 
                        rotation=0, va='top', ha='left', fontsize=9, color='red')
            
            ax_main.set_xlabel('Date', fontsize=12)
            ax_main.set_ylabel('Price ($)', fontsize=12)
            ax_main.set_title(f'Price Forecast - Multiple Methods ({days_ahead} Days)', 
                            fontsize=14, fontweight='bold')
            ax_main.legend(loc='best', fontsize=10)
            ax_main.grid(True, alpha=0.3)
            ax_main.tick_params(axis='x', rotation=45)
            
            # Individual method charts
            method_list = [m for m in forecast_data['methods'].items() if 'forecast' in m[1]]
            
            for idx, (method_name, method_data) in enumerate(method_list[:3]):
                row = (idx // 2) + 1
                col = idx % 2
                ax = fig.add_subplot(gs[row, col])
                
                # Plot historical
                ax.plot(hist_dates, hist_prices, 'gray', linewidth=1.5, 
                       label='Historical', alpha=0.5)
                
                # Plot method-specific trend line if available
                if 'trend_line' in method_data:
                    ax.plot(hist_dates, method_data['trend_line'], 
                           'b--', linewidth=1, alpha=0.5, label='Fitted Trend')
                elif 'smoothed_line' in method_data:
                    ax.plot(hist_dates, method_data['smoothed_line'], 
                           'orange', linewidth=1, alpha=0.5, label='Smoothed')
                elif 'ma_line' in method_data:
                    ax.plot(hist_dates, method_data['ma_line'], 
                           'green', linewidth=1, alpha=0.5, label='Moving Avg')
                
                # Plot forecast
                color = colors.get(method_name, '#9C27B0')
                ax.plot(future_dates, method_data['forecast'], 
                       linestyle='--', linewidth=2, color=color, label='Forecast')
                
                # Confidence interval
                ax.fill_between(future_dates, 
                               method_data['confidence_lower'],
                               method_data['confidence_upper'],
                               color=color, alpha=0.2, label='Confidence')
                
                # Vertical line
                ax.axvline(x=hist_dates[-1], color='red', linestyle=':', alpha=0.5)
                
                ax.set_xlabel('Date', fontsize=10)
                ax.set_ylabel('Price ($)', fontsize=10)
                ax.set_title(f'{method_name}\n{method_data.get("description", "")}', 
                           fontsize=11, fontweight='bold')
                ax.legend(loc='best', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            
            fig.suptitle('Price Forecasting Analysis', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Log predictions
            self._log("✓ Price Forecast created!")
            stats = forecast_data['statistics']
            self._log(f"  Current price: ${stats['current_price']:.2f}")
            self._log(f"  Forecast horizon: {days_ahead} days")
            
            # Convert numpy datetime to pandas timestamp
            final_date = pd.Timestamp(future_dates[-1])
            self._log(f"\n  Predictions for {final_date.strftime('%Y-%m-%d')} ({days_ahead} days):")
            
            for method_name, method_data in forecast_data['methods'].items():
                if 'forecast' in method_data:
                    final_price = method_data['forecast'][-1]
                    change = ((final_price - stats['current_price']) / stats['current_price']) * 100
                    conf_lower = method_data['confidence_lower'][-1]
                    conf_upper = method_data['confidence_upper'][-1]
                    
                    self._log(f"    {method_name}: ${final_price:.2f} ({change:+.2f}%)")
                    self._log(f"      Range: ${conf_lower:.2f} - ${conf_upper:.2f}")
            
            plt.show()
            
        except Exception as e:
            self._log(f"ERROR: Failed to create Price Forecast - {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            raise
    
    def create_ma_forecast_chart(self, days_ahead: int = 30):
        """Create Moving Average Forecast chart"""
        df = self.data_processor.get_data()
        if df is None or len(df) == 0:
            self._log("ERROR: No data available")
            return
        
        try:
            self._log(f"Creating MA Forecast ({days_ahead} days ahead)...")
            
            # Sort by date ascending
            df_sorted = df.sort_values('Date')
            dates = df_sorted['Date'].values
            prices = df_sorted['Price'].values
            n = len(prices)
            
            # Calculate multiple MAs
            ma_periods = [7, 14, 30, 50, 90]
            ma_data = {}
            
            for period in ma_periods:
                if n >= period:
                    ma = df_sorted['Price'].rolling(window=period).mean().values
                    
                    # Calculate trend from last N points
                    lookback = min(period, 30)
                    recent_ma = ma[-lookback:]
                    recent_ma_clean = recent_ma[~np.isnan(recent_ma)]
                    
                    if len(recent_ma_clean) > 1:
                        # Linear fit to recent MA
                        x = np.arange(len(recent_ma_clean))
                        coeffs = np.polyfit(x, recent_ma_clean, 1)
                        trend = coeffs[0]
                        
                        # Forecast
                        last_ma = recent_ma_clean[-1]
                        forecast = [last_ma + trend * (i + 1) for i in range(days_ahead)]
                        
                        ma_data[period] = {
                            'ma': ma,
                            'forecast': np.array(forecast),
                            'trend': trend,
                            'last_value': last_ma
                        }
            
            # Generate future dates
            last_date = pd.Timestamp(dates[-1])
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                         periods=days_ahead, freq='D')
            
            # Create figure
            fig, axes = plt.subplots(2, 1, figsize=(16, 12), 
                                    gridspec_kw={'height_ratios': [2, 1]})
            
            # Top chart: Price + MAs + Forecasts
            ax1 = axes[0]
            
            # Plot price
            ax1.plot(dates, prices, 'k-', linewidth=2, label='Price', alpha=0.7)
            
            # Colors for MAs
            colors = {7: '#2196F3', 14: '#4CAF50', 30: '#FF9800', 50: '#9C27B0', 90: '#F44336'}
            
            # Plot MAs and their forecasts
            for period, data in ma_data.items():
                color = colors.get(period, '#666')
                
                # Historical MA
                ax1.plot(dates, data['ma'], '-', linewidth=1.5, 
                        color=color, label=f'MA{period}', alpha=0.7)
                
                # Forecast
                ax1.plot(future_dates, data['forecast'], '--', linewidth=2,
                        color=color, label=f'MA{period} Forecast', alpha=0.8)
            
            # Vertical line at forecast start
            ax1.axvline(x=dates[-1], color='red', linestyle=':', alpha=0.5, linewidth=2)
            ax1.text(dates[-1], ax1.get_ylim()[1], ' Forecast Start', 
                    rotation=0, va='top', ha='left', fontsize=9, color='red')
            
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.set_title(f'Moving Average Forecast ({days_ahead} Days)', 
                         fontsize=14, fontweight='bold')
            ax1.legend(loc='best', fontsize=9, ncol=2)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Bottom chart: MA Trends
            ax2 = axes[1]
            
            trend_periods = list(ma_data.keys())
            trend_values = [ma_data[p]['trend'] for p in trend_periods]
            colors_list = [colors.get(p, '#666') for p in trend_periods]
            
            bars = ax2.bar([f'MA{p}' for p in trend_periods], trend_values, 
                          color=colors_list, alpha=0.7, edgecolor='black')
            
            # Add value labels
            for bar, val in zip(bars, trend_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'${val:.2f}/day',
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=9, fontweight='bold')
            
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax2.set_ylabel('Trend ($/day)', fontsize=12)
            ax2.set_title('MA Trend Strength', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            # Log predictions
            self._log("✓ MA Forecast created!")
            self._log(f"  Forecast horizon: {days_ahead} days")
            
            # Convert numpy datetime to pandas timestamp
            final_date = pd.Timestamp(future_dates[-1])
            self._log(f"\n  MA Predictions for {final_date.strftime('%Y-%m-%d')}:")
            
            for period, data in sorted(ma_data.items()):
                final_value = data['forecast'][-1]
                current_price = prices[-1]
                change = ((final_value - current_price) / current_price) * 100
                
                self._log(f"    MA{period}: ${final_value:.2f} ({change:+.2f}%) - Trend: ${data['trend']:.2f}/day")
            
            # Trading signals
            self._log("\n  🔔 Potential Trading Signals:")
            
            # Check for MA crossovers in forecast
            if 7 in ma_data and 30 in ma_data:
                ma7_end = ma_data[7]['forecast'][-1]
                ma30_end = ma_data[30]['forecast'][-1]
                
                if ma7_end > ma30_end:
                    self._log("    📈 BULLISH: MA7 above MA30 (Golden Cross territory)")
                else:
                    self._log("    📉 BEARISH: MA7 below MA30 (Death Cross territory)")
            
            plt.show()
            
        except Exception as e:
            self._log(f"ERROR: Failed to create MA Forecast - {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            raise
    
    def create_arima_forecast_chart(self, days_ahead: int = 30):
        """Create ARIMA Forecast chart"""
        df = self.data_processor.get_data()
        if df is None or len(df) == 0:
            self._log("ERROR: No data available")
            return
        
        try:
            self._log(f"Creating ARIMA Forecast ({days_ahead} days ahead)...")
            
            arima_data = self.data_processor.calculate_arima_forecast(days_ahead)
            
            if 'error' in arima_data:
                self._log(f"ERROR: {arima_data['error']}")
                import tkinter.messagebox as messagebox
                messagebox.showerror("ARIMA Error", arima_data['error'])
                return
            
            # Create figure with constrained_layout instead of tight_layout
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                          gridspec_kw={'height_ratios': [2, 1]},
                                          constrained_layout=True)
            
            hist_dates = arima_data['historical_dates']
            hist_prices = arima_data['historical_prices']
            future_dates = arima_data['future_dates']
            forecast = arima_data['forecast']
            
            # Top chart: Historical + Forecast
            ax1.plot(hist_dates, hist_prices, 'k-', linewidth=2, 
                    label='Historical Price', alpha=0.7)
            
            # Fitted values
            if 'fitted_values' in arima_data:
                ax1.plot(hist_dates, arima_data['fitted_values'], 
                        'b-', linewidth=1, alpha=0.5, label='ARIMA Fitted')
            
            # Forecast
            ax1.plot(future_dates, forecast, 'r--', linewidth=2.5, 
                    label='ARIMA Forecast', alpha=0.9)
            
            # Confidence interval
            ax1.fill_between(future_dates, 
                            arima_data['confidence_lower'],
                            arima_data['confidence_upper'],
                            color='red', alpha=0.2, label='95% Confidence')
            
            # Vertical line
            ax1.axvline(x=hist_dates[-1], color='orange', linestyle=':', 
                       alpha=0.6, linewidth=2)
            ax1.text(hist_dates[-1], ax1.get_ylim()[1], ' Forecast Start', 
                    rotation=0, va='top', ha='left', fontsize=9, color='orange')
            
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.set_title(f'ARIMA Forecast - Order{arima_data["model_order"]} ({days_ahead} Days)', 
                         fontsize=14, fontweight='bold')
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Bottom chart: Residuals
            if 'fitted_values' in arima_data:
                residuals = hist_prices - arima_data['fitted_values']
                
                ax2.plot(hist_dates, residuals, 'gray', linewidth=1, alpha=0.6)
                ax2.axhline(y=0, color='red', linestyle='-', linewidth=1)
                ax2.fill_between(hist_dates, 0, residuals, 
                                where=(residuals >= 0), color='green', alpha=0.3)
                ax2.fill_between(hist_dates, 0, residuals, 
                                where=(residuals < 0), color='red', alpha=0.3)
                
                ax2.set_xlabel('Date', fontsize=11)
                ax2.set_ylabel('Residuals ($)', fontsize=11)
                ax2.set_title('Model Fit Quality (Residuals)', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='x', rotation=45)
            
            # Don't use plt.tight_layout() since we're using constrained_layout=True
            
            # Log results
            self._log("✓ ARIMA Forecast created!")
            self._log(f"  Model: ARIMA{arima_data['model_order']}")
            self._log(f"  AIC Score: {arima_data['aic']:.2f} (lower is better)")
            self._log(f"  Data stationarity: {'Yes' if arima_data['is_stationary'] else 'No (differenced)'}")
            
            current_price = hist_prices[-1]
            final_forecast = forecast[-1]
            change = ((final_forecast - current_price) / current_price) * 100
            
            # Convert numpy datetime to pandas timestamp for strftime
            final_date = pd.Timestamp(future_dates[-1])
            
            self._log(f"\n  Prediction for {final_date.strftime('%Y-%m-%d')}:")
            self._log(f"    Price: ${final_forecast:.2f} ({change:+.2f}%)")
            self._log(f"    95% Range: ${arima_data['confidence_lower'][-1]:.2f} - ${arima_data['confidence_upper'][-1]:.2f}")
            
            plt.show()
            
        except Exception as e:
            self._log(f"ERROR: Failed to create ARIMA Forecast - {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            raise
    
    def create_prophet_forecast_chart(self, days_ahead: int = 30):
        """Create Prophet Forecast chart"""
        df = self.data_processor.get_data()
        if df is None or len(df) == 0:
            self._log("ERROR: No data available")
            return
        
        try:
            self._log(f"Creating Prophet Forecast ({days_ahead} days ahead)...")
            
            prophet_data = self.data_processor.calculate_prophet_forecast(days_ahead)
            
            if 'error' in prophet_data:
                self._log(f"ERROR: {prophet_data['error']}")
                import tkinter.messagebox as messagebox
                messagebox.showerror("Prophet Error", prophet_data['error'])
                return
            
            # Create figure with constrained_layout
            fig = plt.figure(figsize=(16, 12), constrained_layout=True)
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
            
            hist_dates = prophet_data['historical_dates']
            hist_prices = prophet_data['historical_prices']
            future_dates = prophet_data['future_dates']
            forecast = prophet_data['forecast']
            
            # Main chart: Historical + Forecast
            ax_main = fig.add_subplot(gs[0, :])
            
            ax_main.plot(hist_dates, hist_prices, 'ko-', linewidth=2, 
                        markersize=2, label='Historical Price', alpha=0.6)
            
            # Fitted values
            if 'fitted_values' in prophet_data:
                ax_main.plot(hist_dates, prophet_data['fitted_values'], 
                            'b-', linewidth=1.5, alpha=0.6, label='Prophet Fitted')
            
            # Forecast
            ax_main.plot(future_dates, forecast, 'g--', linewidth=2.5, 
                        label='Prophet Forecast', alpha=0.9)
            
            # Confidence interval
            ax_main.fill_between(future_dates, 
                                prophet_data['confidence_lower'],
                                prophet_data['confidence_upper'],
                                color='green', alpha=0.2, label='Uncertainty')
            
            # Vertical line
            ax_main.axvline(x=hist_dates[-1], color='purple', linestyle=':', 
                           alpha=0.6, linewidth=2)
            ax_main.text(hist_dates[-1], ax_main.get_ylim()[1], ' Forecast Start', 
                        rotation=0, va='top', ha='left', fontsize=9, color='purple')
            
            ax_main.set_xlabel('Date', fontsize=12)
            ax_main.set_ylabel('Price ($)', fontsize=12)
            ax_main.set_title(f'Prophet Forecast - ML-based with Seasonality ({days_ahead} Days)', 
                             fontsize=14, fontweight='bold')
            ax_main.legend(loc='best', fontsize=10)
            ax_main.grid(True, alpha=0.3)
            ax_main.tick_params(axis='x', rotation=45)
            
            # Trend component
            if 'trend' in prophet_data and prophet_data['trend'] is not None:
                ax1 = fig.add_subplot(gs[1, 0])
                ax1.plot(future_dates, prophet_data['trend'], 'r-', linewidth=2)
                ax1.set_title('Trend Component', fontsize=11, fontweight='bold')
                ax1.set_xlabel('Date', fontsize=10)
                ax1.set_ylabel('Trend', fontsize=10)
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='x', rotation=45)
            
            # Weekly seasonality
            if 'weekly' in prophet_data and prophet_data['weekly'] is not None:
                ax2 = fig.add_subplot(gs[1, 1])
                ax2.plot(future_dates, prophet_data['weekly'], 'b-', linewidth=2)
                ax2.set_title('Weekly Seasonality', fontsize=11, fontweight='bold')
                ax2.set_xlabel('Date', fontsize=10)
                ax2.set_ylabel('Weekly Effect', fontsize=10)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='x', rotation=45)
            
            # Forecast summary
            ax3 = fig.add_subplot(gs[2, :])
            
            # Create bins for forecast
            n_bins = min(10, days_ahead)
            bin_size = days_ahead // n_bins
            
            bin_labels = []
            bin_values = []
            
            for i in range(n_bins):
                start_idx = i * bin_size
                end_idx = min((i + 1) * bin_size, days_ahead)
                
                if start_idx < len(forecast):
                    bin_mean = np.mean(forecast[start_idx:end_idx])
                    bin_values.append(bin_mean)
                    
                    days_out = start_idx + bin_size // 2
                    bin_labels.append(f'Day {days_out}')
            
            current_price = hist_prices[-1]
            colors = ['green' if v > current_price else 'red' for v in bin_values]
            
            bars = ax3.bar(bin_labels, bin_values, color=colors, alpha=0.7, edgecolor='black')
            ax3.axhline(y=current_price, color='blue', linestyle='--', 
                       linewidth=2, label=f'Current: ${current_price:.2f}')
            
            # Add value labels
            for bar, val in zip(bars, bin_values):
                height = bar.get_height()
                change = ((val - current_price) / current_price) * 100
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'${val:.0f}\n({change:+.1f}%)',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax3.set_ylabel('Predicted Price ($)', fontsize=11)
            ax3.set_title('Forecast Progression', fontsize=12, fontweight='bold')
            ax3.legend(loc='best')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Don't use plt.tight_layout() since we're using constrained_layout=True
            
            # Log results
            self._log("✓ Prophet Forecast created!")
            self._log(f"  Algorithm: Facebook Prophet (ML-based)")
            self._log(f"  Seasonality: Daily + Weekly" + (" + Yearly" if 'yearly' in prophet_data and prophet_data['yearly'] is not None else ""))
            
            final_forecast = forecast[-1]
            change = ((final_forecast - current_price) / current_price) * 100
            
            # Convert numpy datetime to pandas timestamp
            final_date = pd.Timestamp(future_dates[-1])
            
            self._log(f"\n  Prediction for {final_date.strftime('%Y-%m-%d')}:")
            self._log(f"    Price: ${final_forecast:.2f} ({change:+.2f}%)")
            self._log(f"    Range: ${prophet_data['confidence_lower'][-1]:.2f} - ${prophet_data['confidence_upper'][-1]:.2f}")
            
            plt.show()
            
        except Exception as e:
            self._log(f"ERROR: Failed to create Prophet Forecast - {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            raise
    
    def create_forecast_comparison_chart(self, days_ahead: int = 30):
        """Create comparison chart of all forecast methods"""
        df = self.data_processor.get_data()
        if df is None or len(df) == 0:
            self._log("ERROR: No data available")
            return
        
        try:
            self._log(f"Creating Forecast Comparison ({days_ahead} days ahead)...")
            
            # Get all forecasts
            basic_forecast = self.data_processor.calculate_price_forecast(days_ahead)
            arima_forecast = self.data_processor.calculate_arima_forecast(days_ahead)
            prophet_forecast = self.data_processor.calculate_prophet_forecast(days_ahead)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Historical data
            hist_dates = basic_forecast['historical_dates']
            hist_prices = basic_forecast['historical_prices']
            future_dates = basic_forecast['future_dates']
            
            # Plot historical
            ax.plot(hist_dates, hist_prices, 'k-', linewidth=3, 
                   label='Historical Price', alpha=0.8, zorder=10)
            
            # Plot each forecast method
            methods_to_plot = []
            
            # Basic methods
            if 'Linear Trend' in basic_forecast['methods']:
                data = basic_forecast['methods']['Linear Trend']
                ax.plot(future_dates, data['forecast'], '--', linewidth=2, 
                       color='#2196F3', label='Linear Trend', alpha=0.7)
                methods_to_plot.append(('Linear', data['forecast'][-1]))
            
            if 'Moving Average' in basic_forecast['methods']:
                data = basic_forecast['methods']['Moving Average']
                ax.plot(future_dates, data['forecast'], '--', linewidth=2,
                       color='#4CAF50', label='Moving Average', alpha=0.7)
                methods_to_plot.append(('MA', data['forecast'][-1]))
            
            # ARIMA
            if 'error' not in arima_forecast:
                ax.plot(future_dates, arima_forecast['forecast'], '--', linewidth=2.5,
                       color='#FF9800', label='ARIMA', alpha=0.8)
                ax.fill_between(future_dates, 
                               arima_forecast['confidence_lower'],
                               arima_forecast['confidence_upper'],
                               color='#FF9800', alpha=0.1)
                methods_to_plot.append(('ARIMA', arima_forecast['forecast'][-1]))
            
            # Prophet
            if 'error' not in prophet_forecast:
                ax.plot(future_dates, prophet_forecast['forecast'], '--', linewidth=2.5,
                       color='#9C27B0', label='Prophet', alpha=0.8)
                ax.fill_between(future_dates,
                               prophet_forecast['confidence_lower'],
                               prophet_forecast['confidence_upper'],
                               color='#9C27B0', alpha=0.1)
                methods_to_plot.append(('Prophet', prophet_forecast['forecast'][-1]))
            
            # Vertical line
            ax.axvline(x=hist_dates[-1], color='red', linestyle=':', 
                      alpha=0.6, linewidth=2)
            ax.text(hist_dates[-1], ax.get_ylim()[1], ' Forecast Start', 
                   rotation=0, va='top', ha='left', fontsize=10, color='red')
            
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.set_title(f'Forecast Method Comparison - {days_ahead} Days Ahead', 
                        fontsize=16, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Log comparison
            self._log("✓ Forecast Comparison created!")
            
            # Convert numpy datetime to pandas timestamp
            final_date = pd.Timestamp(future_dates[-1])
            self._log(f"\n  Predictions for {final_date.strftime('%Y-%m-%d')} ({days_ahead} days):")
            
            current_price = hist_prices[-1]
            for method_name, final_price in sorted(methods_to_plot, key=lambda x: x[1], reverse=True):
                change = ((final_price - current_price) / current_price) * 100
                self._log(f"    {method_name:12s}: ${final_price:>10,.2f} ({change:+6.2f}%)")
            
            # Calculate consensus
            prices = [p for _, p in methods_to_plot]
            avg_price = np.mean(prices)
            std_price = np.std(prices)
            avg_change = ((avg_price - current_price) / current_price) * 100
            
            self._log(f"\n  📊 Consensus:")
            self._log(f"    Average: ${avg_price:.2f} ({avg_change:+.2f}%)")
            self._log(f"    Spread: ±${std_price:.2f}")
            self._log(f"    Agreement: {'High' if std_price < avg_price * 0.05 else 'Moderate' if std_price < avg_price * 0.10 else 'Low'}")
            
            plt.show()
            
        except Exception as e:
            self._log(f"ERROR: Failed to create Forecast Comparison - {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            raise

