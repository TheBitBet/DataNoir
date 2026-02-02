import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from datetime import datetime
import sys

class CSVVisualizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CSV 3D Visualizer - Advanced")
        self.root.geometry("750x650")
        self.root.resizable(True, True)
        
        self.csv_path = None
        self.df = None
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="CSV Advanced Visualizer", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # File selection frame
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=10, padx=20, fill=tk.X)
        
        self.file_label = tk.Label(file_frame, text="No file selected", 
                                   fg="gray", wraplength=400)
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        upload_btn = tk.Button(file_frame, text="Upload CSV", 
                              command=self.upload_file,
                              bg="#4CAF50", fg="white", 
                              font=("Arial", 10, "bold"),
                              padx=10, pady=5)
        upload_btn.pack(side=tk.RIGHT, padx=5)
        
        # Visualization options frame
        options_frame = tk.LabelFrame(self.root, text="3D Visualization Options", 
                                     font=("Arial", 10, "bold"), padx=10, pady=10)
        options_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Scale option
        scale_frame = tk.Frame(options_frame)
        scale_frame.pack(fill=tk.X, pady=5)
        tk.Label(scale_frame, text="Volume Scale:", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        
        self.scale_var = tk.StringVar(value="linear")
        scale_options = [("Linear", "linear"), ("Logarithmic", "log"), ("Normalized", "normalized")]
        for text, value in scale_options:
            tk.Radiobutton(scale_frame, text=text, variable=self.scale_var, 
                          value=value, font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        
        # Color scheme option
        color_frame = tk.Frame(options_frame)
        color_frame.pack(fill=tk.X, pady=5)
        tk.Label(color_frame, text="Color by:", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        
        self.color_var = tk.StringVar(value="price")
        color_options = [("Price", "price"), ("Volume", "volume"), ("Date", "date")]
        for text, value in color_options:
            tk.Radiobutton(color_frame, text=text, variable=self.color_var, 
                          value=value, font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        
        # Point size
        size_frame = tk.Frame(options_frame)
        size_frame.pack(fill=tk.X, pady=5)
        tk.Label(size_frame, text="Point Size:", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        self.size_var = tk.IntVar(value=50)
        tk.Scale(size_frame, from_=20, to=150, orient=tk.HORIZONTAL, 
                variable=self.size_var, length=200).pack(side=tk.LEFT, padx=5)
        
        # Buttons frame - Row 1
        button_frame1 = tk.Frame(self.root)
        button_frame1.pack(pady=5)
        
        visualize_btn = tk.Button(button_frame1, text="📊 3D Chart", 
                                 command=self.visualize_data,
                                 bg="#2196F3", fg="white",
                                 font=("Arial", 9, "bold"),
                                 padx=10, pady=5,
                                 state=tk.DISABLED)
        visualize_btn.pack(side=tk.LEFT, padx=3)
        self.visualize_btn = visualize_btn
        
        chart_btn = tk.Button(button_frame1, text="📈 Basic 2D", 
                             command=self.show_2d_charts,
                             bg="#673AB7", fg="white",
                             font=("Arial", 9, "bold"),
                             padx=10, pady=5,
                             state=tk.DISABLED)
        chart_btn.pack(side=tk.LEFT, padx=3)
        self.chart_btn = chart_btn
        
        ma_btn = tk.Button(button_frame1, text="📉 Moving Avg", 
                          command=self.show_moving_averages,
                          bg="#009688", fg="white",
                          font=("Arial", 9, "bold"),
                          padx=10, pady=5,
                          state=tk.DISABLED)
        ma_btn.pack(side=tk.LEFT, padx=3)
        self.ma_btn = ma_btn
        
        rsi_btn = tk.Button(button_frame1, text="📊 RSI", 
                           command=self.show_rsi,
                           bg="#FF5722", fg="white",
                           font=("Arial", 9, "bold"),
                           padx=10, pady=5,
                           state=tk.DISABLED)
        rsi_btn.pack(side=tk.LEFT, padx=3)
        self.rsi_btn = rsi_btn
        
        # Buttons frame - Row 2
        button_frame2 = tk.Frame(self.root)
        button_frame2.pack(pady=5)
        
        change_btn = tk.Button(button_frame2, text="💹 Price Change %", 
                              command=self.show_price_change,
                              bg="#E91E63", fg="white",
                              font=("Arial", 9, "bold"),
                              padx=10, pady=5,
                              state=tk.DISABLED)
        change_btn.pack(side=tk.LEFT, padx=3)
        self.change_btn = change_btn
        
        vol_profile_btn = tk.Button(button_frame2, text="📊 Volume Profile", 
                                    command=self.show_volume_profile,
                                    bg="#795548", fg="white",
                                    font=("Arial", 9, "bold"),
                                    padx=10, pady=5,
                                    state=tk.DISABLED)
        vol_profile_btn.pack(side=tk.LEFT, padx=3)
        self.vol_profile_btn = vol_profile_btn
        
        clear_btn = tk.Button(button_frame2, text="🗑️ Clear Log", 
                            command=self.clear_log,
                            bg="#FF9800", fg="white",
                            font=("Arial", 9, "bold"),
                            padx=10, pady=5)
        clear_btn.pack(side=tk.LEFT, padx=3)
        
        export_btn = tk.Button(button_frame2, text="💾 Export Log", 
                              command=self.export_log,
                              bg="#9C27B0", fg="white",
                              font=("Arial", 9, "bold"),
                              padx=10, pady=5)
        export_btn.pack(side=tk.LEFT, padx=3)
        
        # Log frame
        log_label = tk.Label(self.root, text="Log:", font=("Arial", 11, "bold"))
        log_label.pack(pady=(10, 0), padx=20, anchor=tk.W)
        
        # Scrolled text for logging
        self.log_text = scrolledtext.ScrolledText(self.root, 
                                                  width=80, 
                                                  height=10,
                                                  font=("Courier", 9),
                                                  bg="#f5f5f5")
        self.log_text.pack(pady=5, padx=20, fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_label = tk.Label(self.root, text="Ready", 
                                    bd=1, relief=tk.SUNKEN, 
                                    anchor=tk.W, bg="#e0e0e0")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initial log message
        self.log("Application started. Please upload a CSV file.")
        self.log("📊 Available visualizations: 3D, Basic 2D, Moving Averages, RSI, Price Change %, Volume Profile")
        
    def log(self, message):
        """Add a message to the log with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def update_status(self, message):
        """Update the status bar"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
        
    def clear_log(self):
        """Clear the log window"""
        self.log_text.delete(1.0, tk.END)
        self.log("Log cleared.")
        
    def export_log(self):
        """Export log to a text file"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Save Log File"
            )
            
            if file_path:
                log_content = self.log_text.get(1.0, tk.END)
                with open(file_path, 'w') as f:
                    f.write(log_content)
                self.log(f"Log exported to: {file_path}")
                messagebox.showinfo("Success", "Log exported successfully!")
        except Exception as e:
            self.log(f"ERROR: Failed to export log - {str(e)}")
            messagebox.showerror("Error", f"Failed to export log:\n{str(e)}")
    
    def upload_file(self):
        """Open file dialog to select CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.csv_path = file_path
            filename = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
            self.file_label.config(text=f"File: {filename}", fg="black")
            self.log(f"File selected: {file_path}")
            self.update_status(f"File loaded: {filename}")
            
            # Load and validate the file
            if self.load_csv():
                self.visualize_btn.config(state=tk.NORMAL)
                self.chart_btn.config(state=tk.NORMAL)
                self.ma_btn.config(state=tk.NORMAL)
                self.rsi_btn.config(state=tk.NORMAL)
                self.change_btn.config(state=tk.NORMAL)
                self.vol_profile_btn.config(state=tk.NORMAL)
            else:
                self.visualize_btn.config(state=tk.DISABLED)
                self.chart_btn.config(state=tk.DISABLED)
                self.ma_btn.config(state=tk.DISABLED)
                self.rsi_btn.config(state=tk.DISABLED)
                self.change_btn.config(state=tk.DISABLED)
                self.vol_profile_btn.config(state=tk.DISABLED)
    
    def parse_volume(self, vol_str):
        """Parse volume string with K, M, B suffixes"""
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
            
        except (ValueError, AttributeError) as e:
            return 0.0
    
    def load_csv(self):
        """Load and parse CSV file"""
        try:
            self.log("Loading CSV file...")
            
            self.df = pd.read_csv(self.csv_path)
            self.df.columns = self.df.columns.str.replace('"', '').str.strip()
            
            self.log(f"Columns found: {', '.join(self.df.columns)}")
            self.log(f"Total rows: {len(self.df)}")
            
            required_cols = ['Date', 'Price', 'Vol.']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            
            if missing_cols:
                self.log(f"ERROR: Missing required columns: {', '.join(missing_cols)}")
                messagebox.showerror("Error", 
                    f"CSV must contain columns: Date, Price, Vol.\nMissing: {', '.join(missing_cols)}")
                return False
            
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df['Price'] = self.df['Price'].astype(str).str.replace(',', '').str.replace('$', '').astype(float)
            self.df['Vol.'] = self.df['Vol.'].apply(self.parse_volume)
            
            # Sort by date descending (newest first in dataframe) for proper left-to-right display
            self.df = self.df.sort_values('Date', ascending=False)
            
            original_len = len(self.df)
            self.df = self.df.dropna(subset=['Date', 'Price', 'Vol.'])
            if len(self.df) < original_len:
                self.log(f"Removed {original_len - len(self.df)} rows with missing data.")
            
            self.log(f"\n--- Data Summary ---")
            self.log(f"Total valid records: {len(self.df)}")
            self.log(f"Date range: {self.df['Date'].min().strftime('%Y-%m-%d')} to {self.df['Date'].max().strftime('%Y-%m-%d')}")
            
            if len(self.df) > 0:
                self.log(f"\nPrice: Min=${self.df['Price'].min():.2f}, Max=${self.df['Price'].max():.2f}, Mean=${self.df['Price'].mean():.2f}")
                self.log(f"Volume: Min={self.df['Vol.'].min():,.0f}, Max={self.df['Vol.'].max():,.0f}, Mean={self.df['Vol.'].mean():,.0f}")
            
            self.log("--- End Summary ---\n")
            
            if len(self.df) == 0:
                self.log("ERROR: No valid data rows after parsing!")
                messagebox.showerror("Error", "No valid data found in CSV file!")
                return False
            
            self.log("✓ CSV loaded and parsed successfully!")
            self.update_status("CSV loaded successfully")
            return True
            
        except Exception as e:
            self.log(f"ERROR: Failed to load CSV - {str(e)}")
            messagebox.showerror("Error", f"Failed to load CSV:\n{str(e)}")
            self.update_status("Error loading CSV")
            return False
    
    def calculate_moving_average(self, data, window):
        """Calculate simple moving average"""
        return data.rolling(window=window, min_periods=1).mean()
    
    def calculate_bollinger_bands(self, data, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = data.rolling(window=window, min_periods=1).mean()
        std = data.rolling(window=window, min_periods=1).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return sma, upper_band, lower_band
    
    def calculate_rsi(self, data, period=14):
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def show_moving_averages(self):
        """Show Moving Averages and Bollinger Bands"""
        if self.df is None or len(self.df) == 0:
            messagebox.showwarning("Warning", "Please load a valid CSV file first!")
            return
        
        try:
            self.log("Creating Moving Averages & Bollinger Bands chart...")
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Calculate moving averages
            ma7 = self.calculate_moving_average(self.df['Price'], 7)
            ma30 = self.calculate_moving_average(self.df['Price'], 30)
            ma90 = self.calculate_moving_average(self.df['Price'], 90)
            
            # Calculate Bollinger Bands
            sma20, upper_bb, lower_bb = self.calculate_bollinger_bands(self.df['Price'])
            
            # Plot price and moving averages (dataframe is newest first)
            ax.plot(self.df['Date'], self.df['Price'], 
                   label='Price', color='black', linewidth=2, alpha=0.7)
            ax.plot(self.df['Date'], ma7, 
                   label='MA 7-day', color='blue', linewidth=1.5, alpha=0.7)
            ax.plot(self.df['Date'], ma30, 
                   label='MA 30-day', color='orange', linewidth=1.5, alpha=0.7)
            ax.plot(self.df['Date'], ma90, 
                   label='MA 90-day', color='red', linewidth=1.5, alpha=0.7)
            
            # Plot Bollinger Bands
            ax.plot(self.df['Date'], upper_bb, 
                   label='Upper BB', color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.plot(self.df['Date'], lower_bb, 
                   label='Lower BB', color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.fill_between(self.df['Date'], lower_bb, upper_bb, 
                           color='gray', alpha=0.1)
            
            ax.set_xlabel('Date (newest → oldest)', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            ax.set_title('Moving Averages & Bollinger Bands', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            self.log("✓ Moving Averages chart created!")
            plt.show()
            
        except Exception as e:
            self.log(f"ERROR: Failed to create Moving Averages chart - {str(e)}")
            messagebox.showerror("Error", f"Failed to create chart:\n{str(e)}")
    
    def show_rsi(self):
        """Show RSI (Relative Strength Index)"""
        if self.df is None or len(self.df) == 0:
            messagebox.showwarning("Warning", "Please load a valid CSV file first!")
            return
        
        try:
            self.log("Creating RSI chart...")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                          gridspec_kw={'height_ratios': [2, 1]})
            
            # Calculate RSI
            rsi = self.calculate_rsi(self.df['Price'])
            
            # Top plot: Price
            ax1.plot(self.df['Date'], self.df['Price'], 
                    color='black', linewidth=2, label='Price')
            ax1.set_ylabel('Price', fontsize=12)
            ax1.set_title('Price and RSI (Relative Strength Index)', 
                         fontsize=14, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: RSI
            ax2.plot(self.df['Date'], rsi, 
                    color='purple', linewidth=2, label='RSI')
            ax2.axhline(y=70, color='r', linestyle='--', linewidth=1, 
                       label='Overbought (70)', alpha=0.7)
            ax2.axhline(y=30, color='g', linestyle='--', linewidth=1, 
                       label='Oversold (30)', alpha=0.7)
            ax2.fill_between(self.df['Date'], 30, 70, color='gray', alpha=0.1)
            ax2.set_xlabel('Date (newest → oldest)', fontsize=12)
            ax2.set_ylabel('RSI', fontsize=12)
            ax2.set_ylim(0, 100)
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            self.log("✓ RSI chart created!")
            self.log("  RSI > 70: Overbought (potential sell signal)")
            self.log("  RSI < 30: Oversold (potential buy signal)")
            plt.show()
            
        except Exception as e:
            self.log(f"ERROR: Failed to create RSI chart - {str(e)}")
            messagebox.showerror("Error", f"Failed to create chart:\n{str(e)}")
    
    def show_price_change(self):
        """Show Price Change % analysis"""
        if self.df is None or len(self.df) == 0:
            messagebox.showwarning("Warning", "Please load a valid CSV file first!")
            return
        
        try:
            self.log("Creating Price Change % chart...")
            
            # Calculate price changes
            price_change = self.df['Price'].pct_change() * 100
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
            
            # Chart 1: Price with change colored
            ax1.plot(self.df['Date'], self.df['Price'], 
                    color='black', linewidth=2)
            ax1.set_ylabel('Price', fontsize=12)
            ax1.set_title('Price Change % Analysis', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Chart 2: Daily % change
            colors = ['green' if x > 0 else 'red' for x in price_change]
            ax2.bar(self.df['Date'], price_change, 
                   color=colors, alpha=0.6, width=1)
            ax2.axhline(y=0, color='black', linewidth=1)
            ax2.set_ylabel('Daily Change %', fontsize=12)
            ax2.set_title('Daily Price Change %', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Chart 3: Cumulative returns
            cumulative_returns = (1 + price_change / 100).cumprod() * 100 - 100
            ax3.plot(self.df['Date'], cumulative_returns, 
                    color='blue', linewidth=2)
            ax3.axhline(y=0, color='black', linewidth=1, linestyle='--')
            ax3.fill_between(self.df['Date'], 0, cumulative_returns, 
                            where=(cumulative_returns >= 0), 
                            color='green', alpha=0.3, label='Profit')
            ax3.fill_between(self.df['Date'], 0, cumulative_returns, 
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
            self.log("✓ Price Change % chart created!")
            self.log(f"  Avg daily change: {price_change.mean():.2f}%")
            self.log(f"  Max gain: {price_change.max():.2f}%")
            self.log(f"  Max loss: {price_change.min():.2f}%")
            self.log(f"  Volatility (std): {price_change.std():.2f}%")
            
            plt.show()
            
        except Exception as e:
            self.log(f"ERROR: Failed to create Price Change % chart - {str(e)}")
            messagebox.showerror("Error", f"Failed to create chart:\n{str(e)}")
    
    def show_volume_profile(self):
        """Show Volume Profile - horizontal volume histogram"""
        if self.df is None or len(self.df) == 0:
            messagebox.showwarning("Warning", "Please load a valid CSV file first!")
            return
        
        try:
            self.log("Creating Volume Profile chart...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), 
                                          gridspec_kw={'width_ratios': [3, 1]})
            
            # Left plot: Price over time
            ax1.plot(self.df['Date'], self.df['Price'], 
                    color='blue', linewidth=2, label='Price')
            ax1.set_xlabel('Date (newest → oldest)', fontsize=12)
            ax1.set_ylabel('Price', fontsize=12)
            ax1.set_title('Price with Volume Profile', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.tick_params(axis='x', rotation=45)
            
            # Right plot: Volume profile (horizontal histogram)
            # Create price bins
            num_bins = 30
            price_bins = np.linspace(self.df['Price'].min(), 
                                    self.df['Price'].max(), num_bins)
            
            # Aggregate volume for each price level
            volume_at_price = []
            bin_centers = []
            
            for i in range(len(price_bins) - 1):
                mask = (self.df['Price'] >= price_bins[i]) & (self.df['Price'] < price_bins[i + 1])
                vol = self.df.loc[mask, 'Vol.'].sum()
                volume_at_price.append(vol)
                bin_centers.append((price_bins[i] + price_bins[i + 1]) / 2)
            
            # Plot horizontal bars
            ax2.barh(bin_centers, volume_at_price, 
                    height=(price_bins[1] - price_bins[0]) * 0.9,
                    color='green', alpha=0.6, edgecolor='darkgreen')
            ax2.set_xlabel('Total Volume', fontsize=12)
            ax2.set_ylabel('Price', fontsize=12)
            ax2.set_title('Volume Profile', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Find and mark the price with highest volume (Point of Control)
            max_vol_idx = np.argmax(volume_at_price)
            poc_price = bin_centers[max_vol_idx]
            ax2.axhline(y=poc_price, color='red', linestyle='--', 
                       linewidth=2, label=f'POC: ${poc_price:.2f}')
            ax1.axhline(y=poc_price, color='red', linestyle='--', 
                       linewidth=2, alpha=0.5, label=f'POC: ${poc_price:.2f}')
            
            ax1.legend(loc='best')
            ax2.legend(loc='best')
            
            plt.tight_layout()
            self.log("✓ Volume Profile chart created!")
            self.log(f"  Point of Control (POC): ${poc_price:.2f}")
            self.log("  POC = Price level with highest trading volume")
            plt.show()
            
        except Exception as e:
            self.log(f"ERROR: Failed to create Volume Profile chart - {str(e)}")
            messagebox.showerror("Error", f"Failed to create chart:\n{str(e)}")
    
    def show_2d_charts(self):
        """Show basic 2D charts"""
        if self.df is None or len(self.df) == 0:
            messagebox.showwarning("Warning", "Please load a valid CSV file first!")
            return
        
        try:
            self.log("Creating 2D charts...")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Data Analysis - 2D Charts', fontsize=16, fontweight='bold')
            
            # Chart 1: Price over time
            ax1 = axes[0, 0]
            ax1.plot(self.df['Date'], self.df['Price'], color='blue', linewidth=2)
            ax1.set_xlabel('Date (newest → oldest)', fontsize=10)
            ax1.set_ylabel('Price', fontsize=10)
            ax1.set_title('Price Over Time', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Chart 2: Volume over time
            ax2 = axes[0, 1]
            ax2.plot(self.df['Date'], self.df['Vol.'], color='green', linewidth=2)
            ax2.set_xlabel('Date (newest → oldest)', fontsize=10)
            ax2.set_ylabel('Volume', fontsize=10)
            ax2.set_title('Volume Over Time', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
            
            # Chart 3: Price vs Volume scatter
            ax3 = axes[1, 0]
            scatter = ax3.scatter(self.df['Vol.'], self.df['Price'], 
                                 c=range(len(self.df)), cmap='viridis', alpha=0.6)
            ax3.set_xlabel('Volume', fontsize=10)
            ax3.set_ylabel('Price', fontsize=10)
            ax3.set_title('Price vs Volume (colored by time)', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax3, label='Time progression')
            
            # Chart 4: Distribution histograms
            ax4 = axes[1, 1]
            ax4_twin = ax4.twinx()
            ax4.hist(self.df['Price'], bins=30, alpha=0.7, color='blue', label='Price')
            ax4_twin.hist(self.df['Vol.'], bins=30, alpha=0.7, color='green', label='Volume')
            ax4.set_xlabel('Value', fontsize=10)
            ax4.set_ylabel('Price Frequency', fontsize=10, color='blue')
            ax4_twin.set_ylabel('Volume Frequency', fontsize=10, color='green')
            ax4.set_title('Distribution of Values', fontweight='bold')
            ax4.legend(loc='upper left')
            ax4_twin.legend(loc='upper right')
            
            plt.tight_layout()
            self.log("✓ 2D charts created successfully!")
            plt.show()
            
        except Exception as e:
            self.log(f"ERROR: Failed to create 2D charts - {str(e)}")
            messagebox.showerror("Error", f"Failed to create charts:\n{str(e)}")
    
    def visualize_data(self):
        """Create 3D visualization with enhanced options"""
        if self.df is None or len(self.df) == 0:
            messagebox.showwarning("Warning", "Please load a valid CSV file first!")
            return
        
        try:
            self.log("Creating 3D visualization...")
            self.update_status("Generating visualization...")
            
            # Get visualization options
            scale_type = self.scale_var.get()
            color_by = self.color_var.get()
            point_size = self.size_var.get()
            
            self.log(f"Options: Scale={scale_type}, Color={color_by}, Size={point_size}")
            
            # Prepare data
            # Dataframe is sorted newest first, so regular index will show newest on left
            date_numeric = np.arange(len(self.df))
            volumes = self.df['Vol.'].values
            prices = self.df['Price'].values
            
            # Apply scaling to volume
            if scale_type == "log":
                volumes_plot = np.log10(volumes + 1)
                ylabel = "Volume (log scale)"
                self.log("Applied logarithmic scaling to volume")
            elif scale_type == "normalized":
                vol_min = volumes.min()
                vol_max = volumes.max()
                if vol_max > vol_min:
                    volumes_plot = (volumes - vol_min) / (vol_max - vol_min) * 100
                else:
                    volumes_plot = volumes
                ylabel = "Volume (normalized 0-100)"
                self.log("Applied normalization to volume")
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
            scatter = ax.scatter(date_numeric, 
                                volumes_plot, 
                                prices,
                                c=color_values, 
                                cmap='viridis', 
                                marker='o',
                                s=point_size,
                                alpha=0.6,
                                edgecolors='w',
                                linewidth=0.5)
            
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
            
            # Customize date labels (dataframe is newest first)
            num_ticks = min(10, len(self.df))
            tick_indices = np.linspace(0, len(self.df)-1, num_ticks, dtype=int)
            ax.set_xticks(date_numeric[tick_indices])
            ax.set_xticklabels([self.df.iloc[i]['Date'].strftime('%Y-%m-%d') 
                                for i in tick_indices], 
                               rotation=45, ha='right', fontsize=8)
            
            # Set view angle
            ax.view_init(elev=20, azim=45)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            self.log(" 3D visualization created successfully!")
            self.log(" TIP: Use your mouse to rotate the plot")
            self.update_status("Visualization displayed")
            
            plt.show()
            
        except Exception as e:
            self.log(f"ERROR: Failed to create visualization - {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to create visualization:\n{str(e)}")
            self.update_status("Error creating visualization")


def main():
    root = tk.Tk()
    app = CSVVisualizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
