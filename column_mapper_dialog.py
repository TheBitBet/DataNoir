"""
Column Mapper Dialog
GUI for manual column mapping when auto-detection fails or user wants to customize
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Optional, Callable
import pandas as pd


class ColumnMapperDialog:
    """Dialog for mapping CSV columns to required fields"""
    
    def __init__(self, parent, df: pd.DataFrame, csv_config, 
                 auto_detected: Dict[str, str] = None):
        """
        Initialize column mapper dialog
        
        Args:
            parent: Parent window
            df: DataFrame to map
            csv_config: CSVConfig instance
            auto_detected: Auto-detected column mappings
        """
        self.parent = parent
        self.df = df
        self.csv_config = csv_config
        self.auto_detected = auto_detected or {}
        self.result = None
        self.save_as_preset = False
        self.preset_name = None
        
        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Column Mapping")
        self.dialog.geometry("700x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self._create_widgets()
        self._populate_selections()
        
        # Center dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (self.dialog.winfo_width() // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f'+{x}+{y}')
    
    def _create_widgets(self):
        """Create all widgets"""
        # Title
        title_frame = tk.Frame(self.dialog, bg="#2196F3", height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="📋 Map CSV Columns", 
                font=("Arial", 16, "bold"), bg="#2196F3", fg="white").pack(pady=15)
        
        # Instructions
        instr_frame = tk.Frame(self.dialog, bg="#f0f0f0")
        instr_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(instr_frame, 
                text="Map the columns from your CSV to the required fields.\n"
                     "Auto-detected mappings are pre-selected.",
                font=("Arial", 9), bg="#f0f0f0", justify=tk.LEFT).pack(padx=10, pady=5)
        
        # Preset selection
        preset_frame = tk.LabelFrame(self.dialog, text="Use Preset Configuration", 
                                    font=("Arial", 10, "bold"), padx=10, pady=10)
        preset_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(preset_frame, text="Preset:", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        
        self.preset_var = tk.StringVar(value="Custom")
        presets = ["Custom"] + list(self.csv_config.get_all_presets().keys())
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                   values=presets, state='readonly', width=20)
        preset_combo.pack(side=tk.LEFT, padx=5)
        preset_combo.bind('<<ComboboxSelected>>', self._on_preset_selected)
        
        tk.Button(preset_frame, text="Apply Preset", 
                 command=self._apply_selected_preset,
                 bg="#4CAF50", fg="white", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        
        # Main mapping frame with scrollbar
        main_frame = tk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Column mappings
        self.mapping_widgets = {}
        self._create_mapping_widgets()
        
        # Data preview
        preview_frame = tk.LabelFrame(self.dialog, text="Data Preview", 
                                     font=("Arial", 10, "bold"))
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.preview_text = tk.Text(preview_frame, height=6, font=("Courier", 8))
        self.preview_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self._update_preview()
        
        # Save as preset option
        save_frame = tk.Frame(self.dialog)
        save_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.save_preset_var = tk.BooleanVar(value=False)
        tk.Checkbutton(save_frame, text="Save this mapping as a preset", 
                      variable=self.save_preset_var,
                      command=self._toggle_preset_name).pack(side=tk.LEFT)
        
        tk.Label(save_frame, text="Name:", font=("Arial", 9)).pack(side=tk.LEFT, padx=(20, 5))
        self.preset_name_entry = tk.Entry(save_frame, width=20, state=tk.DISABLED)
        self.preset_name_entry.pack(side=tk.LEFT)
        
        # Buttons
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(button_frame, text="✓ OK", command=self._on_ok,
                 bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                 padx=20, pady=5).pack(side=tk.RIGHT, padx=5)
        
        tk.Button(button_frame, text="✗ Cancel", command=self._on_cancel,
                 bg="#f44336", fg="white", font=("Arial", 10, "bold"),
                 padx=20, pady=5).pack(side=tk.RIGHT, padx=5)
        
        tk.Button(button_frame, text="🔄 Auto-Detect", command=self._auto_detect,
                 bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                 padx=20, pady=5).pack(side=tk.LEFT, padx=5)
    
    def _create_mapping_widgets(self):
        """Create dropdown widgets for each required field"""
        csv_columns = ['None'] + list(self.df.columns)
        
        # Required columns
        tk.Label(self.scrollable_frame, text="Required Fields:", 
                font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=3, 
                                                 sticky=tk.W, pady=(5, 10))
        
        row = 1
        for standard_name in self.csv_config.REQUIRED_COLUMNS:
            self._create_mapping_row(row, standard_name, csv_columns, required=True)
            row += 1
        
        # Optional columns
        tk.Label(self.scrollable_frame, text="Optional Fields:", 
                font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=3, 
                                                 sticky=tk.W, pady=(15, 10))
        row += 1
        
        optional_fields = ['Open', 'High', 'Low', 'Change']
        for standard_name in optional_fields:
            self._create_mapping_row(row, standard_name, csv_columns, required=False)
            row += 1
    
    def _create_mapping_row(self, row: int, standard_name: str, 
                           csv_columns: List[str], required: bool):
        """Create a single mapping row"""
        # Label
        label_text = f"{standard_name}:" + (" *" if required else "")
        label = tk.Label(self.scrollable_frame, text=label_text, 
                        font=("Arial", 9, "bold" if required else "normal"),
                        fg="red" if required else "black")
        label.grid(row=row, column=0, sticky=tk.W, padx=(10, 5), pady=5)
        
        # Dropdown
        var = tk.StringVar(value='None')
        combo = ttk.Combobox(self.scrollable_frame, textvariable=var,
                            values=csv_columns, state='readonly', width=25)
        combo.grid(row=row, column=1, padx=5, pady=5)
        combo.bind('<<ComboboxSelected>>', lambda e: self._update_preview())
        
        self.mapping_widgets[standard_name] = var
        
        # Info button
        info_btn = tk.Button(self.scrollable_frame, text="ℹ️", 
                           command=lambda: self._show_column_info(standard_name),
                           font=("Arial", 8), width=3)
        info_btn.grid(row=row, column=2, padx=5)
    
    def _populate_selections(self):
        """Populate dropdowns with auto-detected or preset values"""
        for standard_name, csv_column in self.auto_detected.items():
            if standard_name in self.mapping_widgets:
                self.mapping_widgets[standard_name].set(csv_column)
    
    def _on_preset_selected(self, event):
        """Handle preset selection"""
        # Just update the combo box, actual application happens on button click
        pass
    
    def _apply_selected_preset(self):
        """Apply the selected preset"""
        preset_name = self.preset_var.get()
        if preset_name == "Custom":
            return
        
        preset_mapping = self.csv_config.apply_preset(preset_name)
        if not preset_mapping:
            messagebox.showwarning("Warning", f"Preset '{preset_name}' not found")
            return
        
        # Apply the preset mapping
        for standard_name, csv_column in preset_mapping.items():
            if standard_name in self.mapping_widgets:
                if csv_column in self.df.columns:
                    self.mapping_widgets[standard_name].set(csv_column)
                else:
                    self.mapping_widgets[standard_name].set('None')
        
        self._update_preview()
        messagebox.showinfo("Success", f"Applied preset: {preset_name}")
    
    def _auto_detect(self):
        """Re-run auto-detection"""
        detected, _ = self.csv_config.detect_columns(self.df)
        for standard_name, csv_column in detected.items():
            if standard_name in self.mapping_widgets:
                self.mapping_widgets[standard_name].set(csv_column)
        self._update_preview()
        messagebox.showinfo("Auto-Detect", f"Detected {len(detected)} column mappings")
    
    def _show_column_info(self, standard_name: str):
        """Show information about suggested columns"""
        suggestions = self.csv_config.suggest_mapping(self.df)
        suggested = suggestions.get(standard_name, [])
        
        if suggested:
            msg = f"Suggested columns for {standard_name}:\n\n"
            msg += "\n".join(f"  • {col}" for col in suggested)
        else:
            msg = f"No suggestions found for {standard_name}"
        
        messagebox.showinfo(f"Suggestions for {standard_name}", msg)
    
    def _update_preview(self):
        """Update the data preview"""
        self.preview_text.delete(1.0, tk.END)
        
        try:
            # Get current mappings
            current_mapping = {}
            for standard_name, var in self.mapping_widgets.items():
                selected = var.get()
                if selected != 'None':
                    current_mapping[standard_name] = selected
            
            if not current_mapping:
                self.preview_text.insert(tk.END, "No columns mapped yet...")
                return
            
            # Show preview of mapped columns
            preview_df = self.df[list(current_mapping.values())].head(5)
            preview_df.columns = [f"{std} ({csv})" 
                                 for std, csv in current_mapping.items()]
            
            self.preview_text.insert(tk.END, preview_df.to_string())
        except Exception as e:
            self.preview_text.insert(tk.END, f"Preview error: {str(e)}")
    
    def _toggle_preset_name(self):
        """Toggle preset name entry"""
        if self.save_preset_var.get():
            self.preset_name_entry.config(state=tk.NORMAL)
        else:
            self.preset_name_entry.config(state=tk.DISABLED)
    
    def _on_ok(self):
        """Handle OK button"""
        # Get mapping
        mapping = {}
        for standard_name, var in self.mapping_widgets.items():
            selected = var.get()
            if selected != 'None':
                mapping[standard_name] = selected
        
        # Validate
        is_valid, errors = self.csv_config.validate_mapping(mapping, self.df)
        if not is_valid:
            messagebox.showerror("Invalid Mapping", 
                               "Errors:\n" + "\n".join(errors))
            return
        
        # Save as preset if requested
        if self.save_preset_var.get():
            preset_name = self.preset_name_entry.get().strip()
            if not preset_name:
                messagebox.showwarning("Warning", "Please enter a preset name")
                return
            self.csv_config.save_custom_config(preset_name, mapping)
            self.save_as_preset = True
            self.preset_name = preset_name
        
        self.result = mapping
        self.dialog.destroy()
    
    def _on_cancel(self):
        """Handle Cancel button"""
        self.result = None
        self.dialog.destroy()
    
    def show(self) -> Optional[Dict[str, str]]:
        """
        Show dialog and wait for result
        
        Returns:
            Column mapping or None if cancelled
        """
        self.dialog.wait_window()
        return self.result
