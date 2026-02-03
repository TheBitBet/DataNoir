# CSV Advanced Visualizer - Modular Architecture

Fight against enshittifcation. DataNoir is a modular Python application for financial data visualization with **smart CSV import** that works with any column format. 


## Project Structure

```
csv_visualizer/
├── main.py                    # GUI application (entry point)
├── data_processor.py          # Data loading & calculations
├── chart_engine.py            # Visualization & chart rendering
├── csv_config.py              # Flexible CSV configuration system
├── column_mapper_dialog.py    # Interactive column mapping GUI
├── csv_configs.json           # User's saved presets (auto-created)
└── README.md                  # This file
```

## Architecture

### **main.py** - GUI Layer
- **Responsibility**: User interface, file dialogs, logging, event handling
- **Key Classes**: `CSVVisualizerGUI`
- **Dependencies**: `tkinter`, `data_processor`, `chart_engine`
- **What it does**:
  - Creates and manages the GUI
  - Handles file upload events
  - Coordinates between data processor and chart engine
  - Manages application state and logging

### **data_processor.py** - Data Layer
- **Responsibility**: Data loading, parsing, transformations, calculations
- **Key Classes**: `DataProcessor`
- **Dependencies**: `pandas`, `numpy`
- **What it does**:
  - Loads and parses CSV files
  - Validates data structure
  - Calculates technical indicators (MA, RSI, Bollinger Bands, etc.)
  - Provides clean data access methods
  - Returns statistics and summaries

### **chart_engine.py** - Visualization Layer
- **Responsibility**: Chart creation and rendering
- **Key Classes**: `ChartEngine`
- **Dependencies**: `matplotlib`, `mpl_toolkits`, `numpy`
- **What it does**:
  - Creates 3D visualizations
  - Generates 2D charts
  - Renders technical analysis charts
  - Handles all matplotlib plotting logic

 ## Flexible CSV Import**
### **1. Auto-Detection (Smart Column Matching)**
When you upload a CSV, the system automatically tries to detect columns:

```python
# Recognizes variations like:
Date: 'date', 'time', 'datetime', 'trading_date', 'dt', 'data'
Price: 'price', 'close', 'closing', 'last', 'value', 'adj_close'
Volume: 'volume', 'vol', 'vol.', 'quantity', 'trading_volume'
```

 If all required columns are detected → **Loads automatically!**  
 If some columns can't be detected → **Opens mapping dialog**

### **2. Interactive Column Mapper**
GUI lets you:
- See all CSV columns with data previews
- Map required fields (Date, Price, Volume)
- Map optional fields (Open, High, Low, Change)
- Preview your mapped data before confirming
- Save your mapping as a preset for future use

### **Saved Configurations**
Custom presets are saved in `csv_configs.json`:
```json
{
  "My Broker": {
    "Date": "TradeDate",
    "Price": "LastPrice",
    "Volume": "TotalVol"
  }
}
```


## Data Flow

```
User Action (Upload CSV)
    ↓
main.py (GUI) 
    ↓
data_processor.py (Parse & Validate)
    ↓
main.py (Update UI & Enable Buttons)
    ↓
User Action (Click Visualization Button)
    ↓
main.py (Get User Options)
    ↓
chart_engine.py (Create Chart using data_processor)
    ↓
Display Chart to User
```

##  Usage

### Running the Application
```python
python main.py
```

## 📊 Available Visualizations

1. **3D Chart** - Date × Volume × Price (with scaling options)
2. **Basic 2D Charts** - Price/Volume over time, distributions
3. **Moving Averages** - 7/30/90-day MA + Bollinger Bands
4. **RSI** - Relative Strength Index with overbought/oversold zones
5. **Price Change %** - Daily changes and cumulative returns
6. **Volume Profile** - Volume distribution across price levels
7. **MACD** - 
8. **Seasonality** -

##  Technical Indicators

### Implemented:
- Simple Moving Average (SMA)
- Bollinger Bands
- Relative Strength Index (RSI)
- Price Change %
- Cumulative Returns
- Volume Profile & Point of Control (POC)

### Easy to Add:
Thanks to the modular design, new indicators can be added to `data_processor.py` as methods, then visualized in `chart_engine.py`.

##  Dependencies

```
pandas>=1.3.0
matplotlib>=3.4.0
numpy>=1.21.0
```

Install with:
```bash
pip install pandas matplotlib numpy
```

##  Extending the Application

### Adding a New Indicator:

1. **Add calculation to `data_processor.py`:**
```python
def calculate_macd(self, fast=12, slow=26, signal=9):
    # Your calculation logic
    return macd_line, signal_line, histogram
```

2. **Add visualization to `chart_engine.py`:**
```python
def create_macd_chart(self):
    macd, signal, hist = self.data_processor.calculate_macd()
    # Your plotting logic
```

3. **Add button to `main.py`:**
```python
btn = tk.Button(text="MACD", command=self.show_macd)
```

### Adding a New Chart Type:

Simply add a new method to `ChartEngine` and wire it to a button in the GUI!


##  Notes
- Charts support various color schemes based on price/volume/date
- All technical indicators use standard financial formulas
- 3d section need further refinement

##  Contributing

When adding new features:
1. Put calculations in `data_processor.py`
2. Put visualizations in `chart_engine.py`
3. Put UI elements in `main.py`
4. Keep the separation clean!

---

**Python**: 3.10+  
**License**: MIT
