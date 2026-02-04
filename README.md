# DataNoir -  Data Visualizer & Forecasting 

**Version:** 3.0 - Comprehensive Forecasting Edition  
**Platform:** Python 3.10+ with Tkinter  
**Status:** Need tester

---

## Overview

**DataNoir** is a financial data analysis tool featuring 16 visualizations and 5 forecasting methods. 
---

##  Key Features

### ** 16 Visualization Types**

**Basic Charts (4):**
- 3D Visualization (Date × Volume × Price)
- Basic 2D Charts (4-panel analysis)
- Price Change % (Daily + Cumulative)
- Volume Profile (Distribution by price level)

**Technical Indicators (3):**
- Moving Averages + Bollinger Bands
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)

**Advanced Analysis (3):**
- Seasonality Analysis (5 sub-charts)
- Correlation Heatmap (7-variable matrix)
- Calendar Heatmap (Daily returns calendar)

**Forecasting Suite (5):**
-  **Price Forecast** - 3 methods comparison
-  **MA Forecast** - Moving average projections
-  **ARIMA** - Statistical time series model
-  **Prophet** - Facebook's ML forecasting
-  **Compare All** - Side-by-side consensus

**Settings (1):**
- 3D Visualization customization

### Flexible CSV Import**
- **1. Auto-Detection (Smart Column Matching)**
When you upload a CSV, the system automatically tries to detect columns:

```python
# Recognizes variations like:
Date: 'date', 'time', 'datetime', 'trading_date', 'dt', 'data'
Price: 'price', 'close', 'closing', 'last', 'value', 'adj_close'
Volume: 'volume', 'vol', 'vol.', 'quantity', 'trading_volume'
```

 If all required columns are detected → **Loads automatically!**  
 If some columns can't be detected → **Opens mapping dialog**

- **2. Interactive Column Mapper**
GUI lets you:
- See all CSV columns with data previews
- Map required fields (Date, Price, Volume)
- Map optional fields (Open, High, Low, Change)
- Get smart suggestions for each field
- Preview your mapped data before confirming
- Save your mapping as a preset for future use

- **3. Preset Configurations**
Built-in presets for common sources:
- Yahoo Finance
- Investing.com
- Google Finance
- Generic


- **Saved Configurations**
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
---

##  Quick Start

### **Installation**

```bash
# Core dependencies
pip install pandas matplotlib numpy --break-system-packages

# Forecasting (optional but recommended)
pip install statsmodels prophet --break-system-packages
```

### **Launch**

```bash
python main.py
```

### **Usage**
1. Click **"Upload CSV"**
2. Select your financial data file
3. Choose a visualization category
4. Click any chart button

---

##  CSV Format

### **Auto-Detection**
DataNoir automatically detects column formats from:
- Yahoo Finance
- Investing.com
- Google Finance
- Custom formats (with mapper dialog)

### **Required Columns**
- **Date** (any date format)
- **Price/Close** (numeric)
- **Volume** (supports K/M/B suffixes)

### **Optional Columns**
- Open, High, Low, Change %

### **Example CSV**
```csv
Date,Price,Volume
2024-01-01,45000.50,2.5M
2024-01-02,45200.00,3.1M
```
## 🔄 Data Flow

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

##  Forecasting Guide

### **Adjustable Horizon**
- **Presets:** 1 Week, 2 Weeks, 1 Month, 2 Months, 3 Months
- **Custom:** 1-180 days (enter manually)

### **Method Comparison**

| Method | Best For | Speed  | Confidence Intervals |
|--------|----------|-------|----------|---------------------|
| **Linear Trend** | Short-term, clear trends | ⚡⚡⚡  | ✅ |
| **Moving Average** | Trading signals | ⚡⚡⚡  | ❌ |
| **ARIMA** | Statistical analysis | ⚡⚡ | ✅ |
| **Prophet** | Long-term, seasonality | ⚡  | ✅ |

---

## Chart Details

### **3D Visualization**
- Configurable scale (Linear/Log/Normalized)
- Color by Price/Volume/Date
- Adjustable point size
- Interactive rotation

### **ARIMA Forecast**
- Auto-selects optimal parameters
- Shows model fit quality
- 95% confidence intervals
- AIC score reporting

### **Prophet Forecast**
- ML-powered predictions
- Automatic seasonality detection
- Trend decomposition
- Component visualization

### **Correlation Heatmap**
- 7 variables analyzed
- Color-coded matrix (-1 to +1)
- Strongest correlations highlighted
- Scatter plots with trend lines

### **Calendar Heatmap**
- Visual daily returns
- Multi-year support
- Green = gains, Red = losses
- Intensity shows magnitude

---

## 🔧 Technical Details

### **Architecture**
- **main.py** - GUI & user interface
- **data_processor.py** - Data loading & calculations
- **chart_engine.py** - Visualization engine
- **csv_config.py** - Column detection & mapping
- **column_mapper_dialog.py** - Interactive mapper

---

## Troubleshooting

### **"statsmodels/Prophet not installed"**
```bash
pip install statsmodels --break-system-packages
pip install prophet --break-system-packages
```

### **Buttons not visible**
- Drag the divider up to show more buttons
- Resize window larger
- Check if CSV loaded successfully

### **Charts not opening**
- Check Activity Log for errors
- Ensure data has enough records (30+ for Prophet, 60+ for ARIMA)
- Verify date column is properly formatted

### **CSV won't load**
- Use the Column Mapper dialog
- Check for proper date format
- Remove header rows if any

---

## 📈 Performance Tips

### **Large Datasets (10,000+ rows)**
- Use Basic 2D Charts for quick overview
- ARIMA may take  up to 30 seconds
- Prophet may take up to 60 seconds
- Consider shorter forecast horizons

### **Small Datasets (<100 rows)**
- Prophet may not work well
- Use Linear Trend or MA Forecast
- ARIMA needs minimum 60 points

---


## Privacy & Data

- **100% Local** - No data sent to servers
- **No API calls** - Works offline
- **No tracking** - Your data stays private
- **No account** - No login required

---

### **Issues**
- Check Activity Log for detailed errors
- Verify CSV format
- Ensure libraries are installed
- Try smaller forecast horizons



## 🎉 Credits

**DataNoir** - Professional Financial Analysis Suite  
Built with Python, Tkinter, Matplotlib, Pandas, Statsmodels, Prophet

**Key Technologies:**
- Python 3.10+
- Matplotlib (visualization)
- Pandas (data processing)
- Statsmodels (ARIMA)
- Prophet (ML forecasting)
- Tkinter (GUI)

---

## 📄 License

This is an analysis tool. Use at your own risk. Not financial advice. 

The MIT License (MIT)
Copyright © 2026

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


---

