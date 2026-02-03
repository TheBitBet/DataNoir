"""
Main GUI Application
Handles user interface, file uploads, and coordinates between data processing and visualization
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from datetime import datetime
import sys
import pandas as pd

# Import our custom modules
from data_processor import DataProcessor
from chart_engine import ChartEngine
from csv_config import CSVConfig
from column_mapper_dialog import ColumnMapperDialog


class CSVVisualizerGUI:
    """Main GUI application for CSV visualization"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("DataNoir")
        self.root.geometry("900x750")  # Increased from 800x700
        self.root.resizable(True, True)
        
        self.csv_path = None
        self.data_processor = DataProcessor()
        self.chart_engine = ChartEngine(self.data_processor)
        self.csv_config = CSVConfig()  # Initialize CSV configuration manager
        
        # Set up logging for chart engine
        self.chart_engine.set_logger(self.log)
        
        # Store button references for enabling/disabling
        self.viz_buttons = []
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets with tabbed interface"""
        # Title
        title_frame = tk.Frame(self.root, bg="#1976D2", height=70)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="DataNoir", 
                font=("Arial", 18, "bold"), bg="#1976D2", fg="white").pack(pady=20)
        
        # File selection frame
        file_frame = tk.Frame(self.root, bg="#f5f5f5", height=60)
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        file_frame.pack_propagate(False)
        
        tk.Label(file_frame, text="📁 File:", font=("Arial", 10, "bold"), 
                bg="#f5f5f5").pack(side=tk.LEFT, padx=(10, 5), pady=15)
        
        self.file_label = tk.Label(file_frame, text="No file selected", 
                                   fg="gray", bg="#f5f5f5", wraplength=400)
        self.file_label.pack(side=tk.LEFT, padx=5, pady=15)
        
        upload_btn = tk.Button(file_frame, text="📤 Upload CSV", 
                              command=self.upload_file,
                              bg="#4CAF50", fg="white", 
                              font=("Arial", 10, "bold"),
                              padx=15, pady=8, cursor="hand2")
        upload_btn.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Create notebook (tabbed interface) with fixed height
        notebook_frame = tk.Frame(self.root, height=350)  # Fixed height container
        notebook_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        notebook_frame.pack_propagate(False)  # Don't shrink
        
        self.notebook = ttk.Notebook(notebook_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self._create_basic_charts_tab()
        self._create_technical_indicators_tab()
        self._create_advanced_analysis_tab()
        self._create_settings_tab()
        
        # Log frame
        log_frame = tk.LabelFrame(self.root, text="📋 Activity Log", 
                                 font=("Arial", 10, "bold"))
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Scrolled text for logging
        self.log_text = scrolledtext.ScrolledText(log_frame, 
                                                  width=80, 
                                                  height=6,  # Reduced from 8 to 6
                                                  font=("Courier", 9),
                                                  bg="#f5f5f5")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bottom button frame
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        tk.Button(bottom_frame, text="🗑️ Clear Log", 
                 command=self.clear_log,
                 bg="#FF9800", fg="white",
                 font=("Arial", 9, "bold"),
                 padx=12, pady=5).pack(side=tk.LEFT, padx=3)
        
        tk.Button(bottom_frame, text="💾 Export Log", 
                 command=self.export_log,
                 bg="#9C27B0", fg="white",
                 font=("Arial", 9, "bold"),
                 padx=12, pady=5).pack(side=tk.LEFT, padx=3)
        
        # Status bar
        self.status_label = tk.Label(self.root, text="Ready", 
                                    bd=1, relief=tk.SUNKEN, 
                                    anchor=tk.W, bg="#e0e0e0")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initial log message
        self.log("🚀 Application started. Please upload a CSV file.")
        self.log("📊 8 visualizations available across 3 categories!")
    
    def _create_basic_charts_tab(self):
        """Create Basic Charts tab"""
        tab = tk.Frame(self.notebook, bg="white")
        self.notebook.add(tab, text="  📊 Basic Charts  ")
        
        # Description
        desc_frame = tk.Frame(tab, bg="#E3F2FD", relief=tk.RIDGE, bd=2)
        desc_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(desc_frame, 
                text="Basic visualizations for price, volume, and distribution analysis",
                font=("Arial", 9, "italic"), bg="#E3F2FD", fg="#1565C0").pack(padx=10, pady=5)
        
        # Buttons container - using pack with expand
        button_container = tk.Frame(tab, bg="white")
        button_container.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        # Row 1
        row1 = tk.Frame(button_container, bg="white")
        row1.pack(anchor=tk.CENTER, pady=5)
        
        btn = self._create_viz_button(row1, "📊 3D Visualization", 
                                      "Interactive 3D plot of Date × Volume × Price",
                                      self.show_3d_chart, "#2196F3")
        self.viz_buttons.append(btn)
        
        btn = self._create_viz_button(row1, "📈 Basic 2D Charts", 
                                      "Price/Volume over time + distributions",
                                      self.show_2d_charts, "#673AB7")
        self.viz_buttons.append(btn)
        
        # Row 2
        row2 = tk.Frame(button_container, bg="white")
        row2.pack(anchor=tk.CENTER, pady=5)
        
        btn = self._create_viz_button(row2, "💹 Price Change %", 
                                      "Daily changes + cumulative returns",
                                      self.show_price_change, "#E91E63")
        self.viz_buttons.append(btn)
        
        btn = self._create_viz_button(row2, "📊 Volume Profile", 
                                      "Volume distribution by price level",
                                      self.show_volume_profile, "#795548")
        self.viz_buttons.append(btn)
    
    def _create_technical_indicators_tab(self):
        """Create Technical Indicators tab"""
        tab = tk.Frame(self.notebook, bg="white")
        self.notebook.add(tab, text="  📈 Technical Indicators  ")
        
        # Description
        desc_frame = tk.Frame(tab, bg="#E8F5E9", relief=tk.RIDGE, bd=2)
        desc_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(desc_frame, 
                text="Professional trading indicators for trend analysis and momentum",
                font=("Arial", 9, "italic"), bg="#E8F5E9", fg="#2E7D32").pack(padx=10, pady=5)
        
        # Buttons container
        button_container = tk.Frame(tab, bg="white")
        button_container.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        # Row 1
        row1 = tk.Frame(button_container, bg="white")
        row1.pack(anchor=tk.CENTER, pady=5)
        
        btn = self._create_viz_button(row1, "📉 Moving Averages", 
                                      "7/30/90-day MA + Bollinger Bands",
                                      self.show_moving_averages, "#009688")
        btn.pack(side=tk.LEFT, padx=5)
        self.viz_buttons.append(btn)
        
        btn = self._create_viz_button(row1, "📊 RSI Indicator", 
                                      "Relative Strength Index (overbought/oversold)",
                                      self.show_rsi, "#FF5722")
        btn.pack(side=tk.LEFT, padx=5)
        self.viz_buttons.append(btn)
        
        # Row 2
        row2 = tk.Frame(button_container, bg="white")
        row2.pack(anchor=tk.CENTER, pady=5)
        
        btn = self._create_viz_button(row2, "📉 MACD", 
                                      "Moving Average Convergence Divergence",
                                      self.show_macd, "#3F51B5")
        btn.pack(side=tk.LEFT, padx=5)
        self.viz_buttons.append(btn)
    
    def _create_advanced_analysis_tab(self):
        """Create Advanced Analysis tab"""
        tab = tk.Frame(self.notebook, bg="white")
        self.notebook.add(tab, text="  🔬 Advanced Analysis  ")
        
        # Description
        desc_frame = tk.Frame(tab, bg="#FFF3E0", relief=tk.RIDGE, bd=2)
        desc_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(desc_frame, 
                text="Advanced statistical analysis and pattern recognition",
                font=("Arial", 9, "italic"), bg="#FFF3E0", fg="#E65100").pack(padx=10, pady=5)
        
        # Buttons container
        button_container = tk.Frame(tab, bg="white")
        button_container.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        # Row 1
        row1 = tk.Frame(button_container, bg="white")
        row1.pack(anchor=tk.CENTER, pady=5)
        
        btn = self._create_viz_button(row1, "📅 Seasonality", 
                                      "Monthly/weekly patterns & trends",
                                      self.show_seasonality, "#00BCD4")
        btn.pack(side=tk.LEFT, padx=5)
        self.viz_buttons.append(btn)
        
        btn = self._create_viz_button(row1, "🔥 Correlation Heatmap", 
                                      "Price/Volume/Change correlations",
                                      self.show_heatmap, "#F44336")
        btn.pack(side=tk.LEFT, padx=5)
        self.viz_buttons.append(btn)
    
    def _create_settings_tab(self):
        """Create Settings/Options tab"""
        tab = tk.Frame(self.notebook, bg="white")
        self.notebook.add(tab, text="  ⚙️ 3D Settings  ")
        
        # Description
        desc_frame = tk.Frame(tab, bg="#F3E5F5", relief=tk.RIDGE, bd=2)
        desc_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(desc_frame, 
                text="Customize 3D visualization options",
                font=("Arial", 9, "italic"), bg="#F3E5F5", fg="#6A1B9A").pack(padx=10, pady=8)
        
        # Settings frame
        settings_frame = tk.Frame(tab, bg="white")
        settings_frame.pack(expand=True, padx=20, pady=20)
        
        # Volume Scale
        scale_frame = tk.LabelFrame(settings_frame, text="Volume Scale", 
                                   font=("Arial", 10, "bold"), bg="white", padx=15, pady=10)
        scale_frame.pack(fill=tk.X, pady=10)
        
        self.scale_var = tk.StringVar(value="linear")
        scale_options = [
            ("Linear", "linear", "Standard linear scale"),
            ("Logarithmic", "log", "Better for large ranges"),
            ("Normalized", "normalized", "Scale 0-100 for comparison")
        ]
        
        for text, value, desc in scale_options:
            frame = tk.Frame(scale_frame, bg="white")
            frame.pack(anchor=tk.W, pady=3)
            tk.Radiobutton(frame, text=text, variable=self.scale_var, 
                          value=value, font=("Arial", 9), bg="white").pack(side=tk.LEFT)
            tk.Label(frame, text=f"  ({desc})", font=("Arial", 8), 
                    fg="gray", bg="white").pack(side=tk.LEFT)
        
        # Color Scheme
        color_frame = tk.LabelFrame(settings_frame, text="Color By", 
                                   font=("Arial", 10, "bold"), bg="white", padx=15, pady=10)
        color_frame.pack(fill=tk.X, pady=10)
        
        self.color_var = tk.StringVar(value="price")
        color_options = [
            ("Price", "price", "Color gradient by price"),
            ("Volume", "volume", "Color gradient by volume"),
            ("Date", "date", "Color gradient by time")
        ]
        
        for text, value, desc in color_options:
            frame = tk.Frame(color_frame, bg="white")
            frame.pack(anchor=tk.W, pady=3)
            tk.Radiobutton(frame, text=text, variable=self.color_var, 
                          value=value, font=("Arial", 9), bg="white").pack(side=tk.LEFT)
            tk.Label(frame, text=f"  ({desc})", font=("Arial", 8), 
                    fg="gray", bg="white").pack(side=tk.LEFT)
        
        # Point Size
        size_frame = tk.LabelFrame(settings_frame, text="Point Size", 
                                  font=("Arial", 10, "bold"), bg="white", padx=15, pady=10)
        size_frame.pack(fill=tk.X, pady=10)
        
        self.size_var = tk.IntVar(value=50)
        
        size_control_frame = tk.Frame(size_frame, bg="white")
        size_control_frame.pack(fill=tk.X)
        
        tk.Label(size_control_frame, text="Small", font=("Arial", 8), 
                bg="white").pack(side=tk.LEFT, padx=5)
        
        tk.Scale(size_control_frame, from_=20, to=150, orient=tk.HORIZONTAL, 
                variable=self.size_var, length=300, bg="white", 
                showvalue=True).pack(side=tk.LEFT, padx=10)
        
        tk.Label(size_control_frame, text="Large", font=("Arial", 8), 
                bg="white").pack(side=tk.LEFT, padx=5)
    
    def _create_viz_button(self, parent, text, description, command, color):
        """Create a standardized visualization button with description"""
        frame = tk.Frame(parent, bg="white", relief=tk.RAISED, bd=1)
        frame.pack(side=tk.LEFT, padx=5, pady=5)  # <-- PACK THE FRAME HERE

        btn = tk.Button(
            frame, text=text, command=command,
            bg=color,
            fg="gray20",                # text color when enabled (initially we'll override later)
            disabledforeground="gray60",# text when disabled
            activebackground=color,
            activeforeground="white",   # hover/click, after enabling
            font=("Arial", 10, "bold"),
            padx=20, pady=15, width=22,
            cursor="hand2",
            state=tk.DISABLED
        )
        btn.pack()

        tk.Label(
            frame, text=description, font=("Arial", 8),
            fg="gray", bg="white", wraplength=200
        ).pack(pady=(5, 10))

        return btn    
    def log(self, message: str):
        """Add a message to the log with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def update_status(self, message: str):
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
                self.log(f"💾 Log exported to: {file_path}")
                messagebox.showinfo("Success", "Log exported successfully!")
        except Exception as e:
            self.log(f"❌ ERROR: Failed to export log - {str(e)}")
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
            self.log(f"📁 File selected: {file_path}")
            self.update_status(f"File loaded: {filename}")
            
            # Load and validate the file
            self.load_csv()
    
    def load_csv(self):
        """Load and parse CSV file using DataProcessor with flexible column mapping"""
        try:
            self.log("⏳ Loading CSV file...")
            self.log(f"DEBUG: CSV path = {self.csv_path}")
            
            # Read CSV to analyze columns
            temp_df = pd.read_csv(self.csv_path)
            temp_df.columns = temp_df.columns.str.replace('"', '').str.strip()
            
            self.log(f"DEBUG: Found {len(temp_df)} rows, {len(temp_df.columns)} columns")
            self.log(f"DEBUG: Columns = {list(temp_df.columns)}")
            
            # Try auto-detection
            detected_mapping, unmatched = self.csv_config.detect_columns(temp_df)
            
            self.log(f"DEBUG: Detected mapping = {detected_mapping}")
            self.log(f"DEBUG: Unmatched = {unmatched}")
            
            if unmatched:
                self.log(f"⚠️  Could not auto-detect columns: {', '.join(unmatched)}")
                self.log("📋 Opening column mapper dialog...")
                
                # Show column mapper dialog
                dialog = ColumnMapperDialog(self.root, temp_df, self.csv_config, detected_mapping)
                column_mapping = dialog.show()
                
                if column_mapping is None:
                    self.log("❌ Import cancelled by user")
                    self.update_status("Import cancelled")
                    self._enable_viz_buttons(False)
                    return
                
                if dialog.save_as_preset:
                    self.log(f"💾 Saved mapping as preset: '{dialog.preset_name}'")
            else:
                # Auto-detection successful
                column_mapping = detected_mapping
                self.log(f"✓ Auto-detected column mapping:")
                for std_name, csv_col in column_mapping.items():
                    self.log(f"  {std_name} → {csv_col}")
            
            self.log(f"DEBUG: Final mapping = {column_mapping}")
            
            # Load CSV with the mapping
            success, message, stats = self.data_processor.load_csv(self.csv_path, column_mapping)
            
            self.log(f"DEBUG: Load result - success={success}, message={message}")
            
            if not success:
                self.log(f"❌ ERROR: {message}")
                messagebox.showerror("Error", message)
                self.update_status("Error loading CSV")
                self._enable_viz_buttons(False)
                return
            
            # Log statistics
            self.log(f"✓ {message}")
            self.log(f"\n--- Data Summary ---")
            self.log(f"Total records: {stats['total_records']}")
            if stats['removed_rows'] > 0:
                self.log(f"Removed rows: {stats['removed_rows']}")
            self.log(f"Date range: {stats['date_min'].strftime('%Y-%m-%d')} to {stats['date_max'].strftime('%Y-%m-%d')}")
            self.log(f"\nPrice: Min=${stats['price_min']:.2f}, Max=${stats['price_max']:.2f}, Mean=${stats['price_mean']:.2f}")
            self.log(f"Volume: Min={stats['volume_min']:,.0f}, Max={stats['volume_max']:,.0f}, Mean={stats['volume_mean']:,.0f}")
            
            # Log the mapping used
            self.log(f"\n--- Column Mapping Used ---")
            for std_name, csv_col in stats['mapping_used'].items():
                self.log(f"  {std_name} ← {csv_col}")
            self.log("--- End Summary ---\n")
            
            self.log(f"DEBUG: About to enable buttons. Button count = {len(self.viz_buttons)}")
            self.update_status("✓ CSV loaded successfully")
            self._enable_viz_buttons(True)
            self.log(f"DEBUG: Buttons enabled!")
            
        except Exception as e:
            self.log(f"❌ ERROR: Unexpected error - {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to load CSV:\n{str(e)}")
            self.update_status("Error loading CSV")
            self._enable_viz_buttons(False)
    
    def _enable_viz_buttons(self, enable):
        if enable:
            for btn in self.viz_buttons:
                btn.config(
                    state=tk.NORMAL,
                    fg="white",            # enabled text becomes white
                    activeforeground="white"
                )
        else:
            for btn in self.viz_buttons:
                btn.config(
                    state=tk.DISABLED,
                    fg="gray20",           # reset if you care
                    disabledforeground="gray60"
                )
    
    # Visualization methods - delegate to ChartEngine
    
    def show_3d_chart(self):
        """Show 3D visualization"""
        try:
            self.chart_engine.create_3d_chart(
                scale_type=self.scale_var.get(),
                color_by=self.color_var.get(),
                point_size=self.size_var.get()
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create 3D chart:\n{str(e)}")
    
    def show_2d_charts(self):
        """Show basic 2D charts"""
        try:
            self.chart_engine.create_basic_2d_charts()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create 2D charts:\n{str(e)}")
    
    def show_moving_averages(self):
        """Show Moving Averages and Bollinger Bands"""
        try:
            self.chart_engine.create_moving_average_chart()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create Moving Averages chart:\n{str(e)}")
    
    def show_rsi(self):
        """Show RSI chart"""
        try:
            self.chart_engine.create_rsi_chart()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create RSI chart:\n{str(e)}")
    
    def show_price_change(self):
        """Show Price Change % chart"""
        try:
            self.chart_engine.create_price_change_chart()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create Price Change chart:\n{str(e)}")
    
    def show_volume_profile(self):
        """Show Volume Profile chart"""
        try:
            self.chart_engine.create_volume_profile_chart()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create Volume Profile chart:\n{str(e)}")
    
    def show_macd(self):
        """Show MACD chart"""
        try:
            self.chart_engine.create_macd_chart()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create MACD chart:\n{str(e)}")
    
    def show_seasonality(self):
        """Show Seasonality chart"""
        try:
            self.chart_engine.create_seasonality_chart()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create Seasonality chart:\n{str(e)}")
    
    def show_heatmap(self):
        """Show Correlation Heatmap"""
        try:
            self.chart_engine.create_correlation_heatmap()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create Heatmap:\n{str(e)}")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = CSVVisualizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
