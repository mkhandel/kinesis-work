import os
import glob
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns
import io
from fpdf import FPDF
from PIL import Image

from capture_report import EXACT_RULES

CATEGORY_MAP = {
    "Is_Waste":       "Waste",
    "Is_Roads":       "Roads",
    "Is_Water_Sewer": "Water/Sewer",
    "Is_Property":    "Property",
    "Is_Environment": "Trees",
    "Is_Animal":      "Animal",
    "Is_Noise":       "Noise",
}

DATE_COL = "Creation Date"
WARD_COL = "Ward"
TYPE_COL = "Service Request Type"
STATUS_COL = "Status"
POSTAL_COL = "First 3 Chars of Postal Code"
INT1_COL = "Intersection Street 1"
INT2_COL = "Intersection Street 2"
DIVISION_COL = "Division"
SECTION_COL = "Section"

def load_data(selected_dir_path_str): 
    ''' 
    This function turns raw 311 csv files into a single dataframe by:
    1. Going to the selected directory
    2. Ensuring said directory actually conatins csv files 
    3. Reading in the columns from each csv 
    4. Converting creation date into datetime 
    '''
    try: 
        dir_path = Path(selected_dir_path_str)

        # Check if directory exists
        if not dir_path.exists():
            print(f"Error: Directory {dir_path} does not exist")
            return None, "Directory not found"
        
        csv_files = list(dir_path.glob("*.csv"))

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV files found in ./data folder")
        
        all_dfs = []
        for file in csv_files:
            print(f"Loading {file.name}...")
            temp_df = pd.read_csv(file, encoding="latin1", engine="python", usecols=range(9))
            # DEBUG: 
            print(f"  -> Shape: {temp_df.shape}")
            print(f"  -> Columns: {list(temp_df.columns)}")
            if not temp_df.empty:
                print(f"  -> First few TYPE_COL values: {temp_df.iloc[:, 7].head().tolist()}")
            #DEBUG END
            all_dfs.append(temp_df.iloc[:, :9])

        df = pd.concat(all_dfs, ignore_index=True)
        # DEBUG:
        print(f"Combined dataframe shape: {df.shape}")

        # Check for empty dataframe
        if df.empty:
            print("Warning: Combined dataframe is empty")
            return df, None

        df = df.iloc[:, :9]
        df.columns = [
            DATE_COL, STATUS_COL, POSTAL_COL,
            INT1_COL, INT2_COL, WARD_COL,
            TYPE_COL, DIVISION_COL, SECTION_COL
        ]

        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=[DATE_COL, TYPE_COL]).copy()
        # Extract time period (e.g., week or month)
        df['time_period'] = df[DATE_COL].dt.to_period('W')  # Weekly
        df["year_month"] = df[DATE_COL].dt.to_period("M").astype(str)

        # Check for nulls in critical columns
        print(f"Nulls in DATE_COL: {df.iloc[:, 0].isnull().sum()}")
        print(f"Nulls in TYPE_COL: {df.iloc[:, 7].isnull().sum()}")

        return df, None
    
    except Exception as e: 
        print(f"Error loading data: {str(e)}")
        return None, str(e)
    
def clean_data(df):
    '''
    This function takes the created dataframe and applies the EXTRACT_RULES to said data in order to clean the categories for easier data comparsions
    It also splits dates into month and year in order to easily compare years 
    '''
    apply_categories(df)
    split_dates(df)
    
def apply_categories(df): 
    '''
    This function maps each TYPE_COL to a category from extract rules
    '''
    # Build a mapping from each string value to its category
    reverse_mapping = {}
    for category, keywords in EXACT_RULES.items():
        for keyword in keywords:
            reverse_mapping[keyword] = category

    # Apply the mapping to TYPE_COL
    df['category'] = df[TYPE_COL].map(reverse_mapping)

    # Apply the CATEGORY_MAP to remove "Is_" prefix and normalize categories
    df['category'] = df['category'].map(CATEGORY_MAP)

def split_dates(df): 
    ''' 
    this function takes the dates from the dataframe and splits them into month and year 
    '''
    # Extract year and month from DATE_COL
    df['year'] = df[DATE_COL].dt.year
    df['month'] = df[DATE_COL].dt.month  # as integer (1–12)

def filter_ward(df, ward_num): 
    '''
    This function creates a secondary df containing only the selected wards events
    since wards contain name and num I am filtering based on the number selected by user 
    '''
    ward_df = df[df[WARD_COL].str.contains(str(ward_num), regex=False, case=False, na=False)]
    return ward_df

def calc_annual_portions(ward_df):
    '''
    This function 
    1. calls filter_ward in order to retrieve a ward specific df 
    2. counts the amount of 
    '''
    # Count occurrences by category and year
    counts = ward_df.groupby(['year', 'category']).size().reset_index(name='count')

    # Calculate total counts per year
    year_totals = counts.groupby('year')['count'].transform('sum')

    # Calculate proportion
    counts['proportion'] = counts['count'] / year_totals

    # Result
    print(counts[['year', 'category', 'count', 'proportion']])
    return counts

def count_total_call_requests_for_ward(ward_df):
    '''
    This function counts the total number of each service category from the selected ward over the given years of data 
    '''
    # Count occurrences of each service type
    service_counts = ward_df['category'].value_counts()
    print(service_counts)
    return service_counts

def chart_request_counts_for_ward(ward_df, ward_num, pdf):
    '''
    This function creates a pie chart of the total percentages of each service type over the given time period in the selected ward 
    aim to visualize the total amount of each service category present in the ward over the total time period 
    '''
    service_counts = count_total_call_requests_for_ward(ward_df)
    service_labels = service_counts.index 
    counts = service_counts.values        

    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=service_labels, autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title('Percentage of Service Calls by Type for Ward')
    plt.axis('equal')

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)  # Always reset before returning
    plt.close()  # Prevent display in non-interactive environments

    img_buffer.seek(0)
    pdf.image(img_buffer, w=pdf.epw * 0.9) 

def calc_rising_trends_porportion(counts, pdf): 
    '''
    This function calculates the portion of each service category per each year in the given ward
    '''
    trends = {}
    for category in counts['category'].unique():
        cat_data = counts[counts['category'] == category]
        if len(cat_data) > 1:  # Need at least 2 years
            slope = linregress(cat_data['year'], cat_data['proportion']).slope
            trends[category] = slope

    # Get top 5 rising categories
    rising_categories = pd.Series(trends).sort_values(ascending=False).head(5)
        
    # Print top 5 rising categories 
    print("Top 5 Rising Categories Based on Proportions of requests:")
    for category in rising_categories.index: 
        cat_data = counts[counts['category'] == category]
        first_year = cat_data['year'].min()
        last_year = cat_data['year'].max()
        first_prop = cat_data[cat_data['year'] == first_year]['proportion'].iloc[0]
        last_prop = cat_data[cat_data['year'] == last_year]['proportion'].iloc[0]

        # Calculate percentage change
        if first_prop > 0:
            pct_change = ((last_prop - first_prop) / first_prop) * 100
        else:
            pct_change = float('inf') if last_prop > 0 else 0

        print(f"{category}: {pct_change:+.1f}% ({first_prop:.3f} → {last_prop:.3f})")

    # add top 5 info to pdf 
    # Add heading to PDF
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Top 5 Rising Categories Based on Proportions of Requests:', ln=True)

    # Add results to PDF
    pdf.set_font('Arial', '', 10)
    for category in rising_categories.index: 
        cat_data = counts[counts['category'] == category]
        first_year = cat_data['year'].min()
        last_year = cat_data['year'].max()
        first_prop = cat_data[cat_data['year'] == first_year]['proportion'].iloc[0]
        last_prop = cat_data[cat_data['year'] == last_year]['proportion'].iloc[0]

        # Calculate percentage change
        if first_prop > 0:
            pct_change = ((last_prop - first_prop) / first_prop) * 100
        else:
            pct_change = float('inf') if last_prop > 0 else 0

        # Add formatted row to PDF
        pdf.cell(0, 8, f"{category}: {pct_change:+.1f}% ({first_prop:.3f} -> {last_prop:.3f})", ln=True)

    # Add spacing after the list
    pdf.ln(5)
        
    display_linear_regression_plot(counts, pdf)

def calc_rising_trends_raw_counts(counts, pdf): 
    '''
    This function calculates the total count of each service category per each year in the given ward
    '''
    trends = {}
    for category in counts['category'].unique():
        cat_data = counts[counts['category'] == category]
        if len(cat_data) > 1:  # Need at least 2 years
            slope = linregress(cat_data['year'], cat_data['count']).slope
            trends[category] = slope
    
    # Get top 5 rising categories
    rising_categories = pd.Series(trends).sort_values(ascending=False).head(5)

    # Print top 5 rising categories 
    print("Top 5 Rising Categories Based on raw count of requests:")
    for category in rising_categories.index: 
        cat_data = counts[counts['category'] == category]
        first_year = cat_data['year'].min()
        last_year = cat_data['year'].max()
        first_count = cat_data[cat_data['year'] == first_year]['count'].iloc[0]
        last_count = cat_data[cat_data['year'] == last_year]['count'].iloc[0]

        # Calculate percentage change
        if first_count > 0:
            pct_change = ((last_count - first_count) / first_count) * 100
        else:
            pct_change = float('inf') if last_count > 0 else 0
        
        print(f"{category}: {pct_change:+.1f}% ({first_count:.3f} → {last_count:.3f})")

    # Add results to PDF
    # Add heading to PDF
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Top 5 Rising Categories Based on Raw Counts of Requests:', ln=True)

    pdf.set_font('Arial', '', 10)
    for category in rising_categories.index: 
        cat_data = counts[counts['category'] == category]
        first_year = cat_data['year'].min()
        last_year = cat_data['year'].max()
        first_count = cat_data[cat_data['year'] == first_year]['count'].iloc[0]
        last_count = cat_data[cat_data['year'] == last_year]['count'].iloc[0]

        # Calculate percentage change
        if first_count > 0:
            pct_change = ((last_count - first_count) / first_count) * 100
        else:
            pct_change = float('inf') if last_count > 0 else 0

        # Add formatted row to PDF
        pdf.cell(0, 8, f"{category}: {pct_change:+.1f}% ({first_count:.3f} -> {last_count:.3f})", ln=True)

    # Add spacing after the list
    pdf.ln(5)

    display_linear_regression_count_plot(counts, pdf)
        

def display_linear_regression_plot(df, pdf): 
    '''
    This function uses Seaborn lmplot to preform linear regression of the given data frame and output a graph visualizing this regression
    '''
    # Create the plot
    plot = sns.lmplot(data=df, x='year', y='proportion', hue='category', height=6, aspect=1.2)
    plt.title('Linear Trends by Category Based on Proportion in This Ward per Each Year')

    # Save to BytesIO buffer
    img_buffer = io.BytesIO()
    plot.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close()

    # Convert to PIL Image
    img_buffer.seek(0)
    pil_image = Image.open(img_buffer)

    # Add to PDF
    pdf.image(pil_image, w=pdf.epw * 0.9)

    # Cleanup
    pil_image.close()
    img_buffer.close()

def display_linear_regression_count_plot(df, pdf): 
    '''
    This function uses Seaborn lmplot to preform linear regression of the given data frame and output a graph visualizing this regression
    '''
    # Create the plot
    plot = sns.lmplot(data=df, x='year', y='count', hue='category', height=6, aspect=1.2)
    plt.title('Linear Trends by Category Based on Raw Count in This Ward per Each Year')

    # Save to BytesIO buffer
    img_buffer = io.BytesIO()
    plot.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close()

    # Convert to PIL Image
    img_buffer.seek(0)
    pil_image = Image.open(img_buffer)

    # Add to PDF
    pdf.image(pil_image, w=pdf.epw * 0.9)

    # Cleanup
    pil_image.close()
    img_buffer.close()
    

def calc_drifting_locations(ward_df): 
    '''
    TODO
    '''
    df_growth = temporal_aggregation(ward_df)

    df_growth['growth_rate'] = df_growth.groupby('POSTAL_COL')['count'].pct_change() * 100

    # Get the most recent period's data for visualization
    most_recent = df_growth[df_growth[DATE_COL] == df_growth[DATE_COL].max()]

    # Sort by growth rate and take top/bottom 10 for clear visualization
    most_recent = most_recent.sort_values('growth_rate', ascending=False)

    top_drifting = most_recent.head(10)
    bottom_drifting = most_recent.tail(10)

    # Create the bar chart
    plt.figure(figsize=(12, 6))

    # Use different colors for positive and negative growth
    colors = ['green' if x > 0 else 'red' for x in top_drifting['growth_rate']]

    bars = plt.bar(top_drifting['POSTAL_CODE'], top_drifting['growth_rate'], color=colors, alpha=0.8)

    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (max(top_drifting['growth_rate'])*0.01),
                f'{top_drifting.iloc[i].growth_rate:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.title('Top 10 FSAs by Month-over-Month Growth Rate', fontsize=14, fontweight='bold')
    plt.xlabel('FSA (Postal Code)')
    plt.ylabel('Growth Rate (%)')
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    plt.tight_layout()
    plt.show()
    return
    
def temporal_aggregation(ward_df): 
    # Group by FSA and time period
    fsa_groups = ward_df.groupby([ward_df[DATE_COL], ward_df[POSTAL_COL].str[:3]]).size().reset_index(name='count');
    
    fsa_time_groups = fsa_groups.groupby([pd.Grouper(key=DATE_COL, freq='ME'),ward_df[POSTAL_COL].str[:3]]).size().reset_index(name='count')

    return fsa_time_groups; 




def ward_to_city_comparison(df, ward_df, ward_num): 
    '''
    This function implements the is_rising_category and plot_rising_category to visualize a ward to city comparison
    '''
    rising_categories = is_rising_category(df, ward_df)
    plot_rising_categories(df, rising_categories, ward_num)

def is_rising_category(df, ward_df):
    '''
    This function 
    '''
    # Group by time period and category for ward and total
    ward_counts = ward_df.groupby(['time_period', 'category']).size().reset_index(name='count')
    total_counts = df.groupby(['time_period', 'category']).size().reset_index(name='count')

    # Convert period to numeric for regression
    ward_counts['time_num'] = ward_counts['time_period'].apply(lambda x: x.ordinal)
    total_counts['time_num'] = total_counts['time_period'].apply(lambda x: x.ordinal)

    rising_categories = []

    for category in df['category'].unique():
        ward_data = ward_counts[ward_counts['category'] == category]
        total_data = total_counts[total_counts['category'] == category]

        if len(ward_data) < 2 or len(total_data) < 2:
            continue  # Need at least 2 points for trend

        # Fit linear trend: slope indicates growth rate
        ward_slope = linregress(ward_data['time_num'], ward_data['count']).slope
        total_slope = linregress(total_data['time_num'], total_data['count']).slope

        # Check if ward growth is faster
        if ward_slope > total_slope:
            rising_categories.append({
                'category': category,
                'ward_growth_rate': ward_slope,
                'total_growth_rate': total_slope
            })
    return pd.DataFrame(rising_categories)

def plot_rising_categories(df, rising_df, ward_num):
    if rising_df.empty:
        print("No categories are rising faster in the ward.")
        return

    # Filter data for rising categories
    top_categories = rising_df['category'].tolist()
    plot_data = df[df['category'].isin(top_categories)].copy()
    plot_data['time_period_str'] = plot_data['time_period'].astype(str)

    # Group by time and category for ward and total
    ward_data = plot_data[plot_data[WARD_COL].str.contains(str(ward_num), regex=False, case=False, na=False)]
    ward_counts = ward_data.groupby(['time_period_str', 'category']).size().reset_index(name='count')
    ward_counts['label'] = ward_num

    total_counts = plot_data.groupby(['time_period_str', 'category']).size().reset_index(name='count')
    total_counts['label'] = 'All Wards'

    # Combine
    viz_data = pd.concat([
        ward_counts.assign(source=f'Ward {ward_num}'),
        total_counts.assign(source='Citywide')
    ])

    # Plot
    g = sns.FacetGrid(viz_data, col='category', col_wrap=2, sharey=False, hue='source')
    g.map(plt.plot, 'time_period_str', 'count', marker='o')
    g.add_legend(title="Data Source")
    g.set_axis_labels("Time Period", "Call Count")
    g.set_titles("{col_name}")
    g.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

# App class used to create a GUI to select Folder with CSV file 
class wowGeneratorApp(tk.Tk):
    '''
        This class creates an interface hat allows the user to enter in ward number, select input directory, and output directory 
    '''
    def __init__(self):
        super().__init__()
        self.title("Wow Generator")
        self.resizable(False, False)

        # ── Variables ──
        self.selected_dir_path = tk.StringVar()
        self.output_dir = tk.StringVar(value=os.path.join(os.getcwd(), "Wow Reports"))
        self.options = ["01", "02", "03", "04", "05","06", "07", "08", "09", "10", "11", "12", 
                         "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25"]
        self.ward_options_var = tk.StringVar(value=self.options)
        
        self.ward_number = tk.IntVar()

        self._build_ui()

    # helper function to build the GUI 
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # Row 0 – Ward Number
        tk.Label(self, text="Ward Number:", anchor="w").grid(
            row=0, column=0, sticky="nw", **pad
        )
        scrollbar = tk.Scrollbar(self)
        scrollbar.grid(row=0, column=2, sticky="ns", pady=6)

        self.lb = tk.Listbox(self, listvariable=self.ward_options_var, 
                   width=50, height=10, yscrollcommand=scrollbar.set
        )
        self.lb.grid(row=0, column=1, sticky="ew", **pad)

        scrollbar.config(command=self.lb.yview)

        # Bind selection event to capture the selected item
        self.lb.bind('<<ListboxSelect>>', self.on_select)

        # Button to demonstrate using the selected value
        tk.Button(self, text="Get Selected Ward", command=self.get_selected_ward).grid(
            row=1, column=0, columnspan=3, pady=10
        )

        # Row 1 – CSV Directory 
        tk.Label(self, text="CSV Directory:", anchor="w").grid(
            row=1, column=0, sticky="w", **pad
            )
        dir_frame = tk.Frame(self)
        dir_frame.grid(row=1, column=1, sticky="ew", **pad)
        tk.Entry(dir_frame, textvariable=self.selected_dir_path, width=40).pack(
            side="left", fill="x", expand=True
        )
        tk.Button(dir_frame, text="Browse…", command=self._browse_dir).pack(
            side="left", padx=(5, 0)
        )

        # Row 2 – Output directory
        tk.Label(self, text="Output Folder:", anchor="w").grid(
            row=2, column=0, sticky="w", **pad
        )
        out_frame = tk.Frame(self)
        out_frame.grid(row=2, column=1, sticky="ew", **pad)
        tk.Entry(out_frame, textvariable=self.output_dir, width=40).pack(
            side="left", fill="x", expand=True
        )
        tk.Button(out_frame, text="Browse…", command=self._browse_output).pack(
            side="left", padx=(5, 0)
        )

        # Row 3 – Run button
        tk.Button(
            self, text="Generate WOW", command=self._run,
            bg="#4CAF50", fg="white", font=("Arial", 11, "bold"),
            padx=20, pady=6
        ).grid(row=3, column=0, columnspan=2, pady=14)

    # ── Callbacks ──
    def _browse_dir(self):
        path = filedialog.askdirectory(
            title="Select Directory"
        )
        if path:
            self.selected_dir_path.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir.set(path)

    def on_select(self, event): 
        """Handle the selection event."""
        selection = self.lb.curselection()
        if selection:
            index = selection[0]
            self.ward_number.set(int(self.options[index]))  # Store as integer
            print(f"Selected Ward Number: {self.ward_number.get()}")

    def get_selected_ward(self):
        """Retrieve and use the selected ward number."""
        try:
            selected_ward = self.ward_number.get()
            print(f"Using Ward Number in program: {selected_ward}")
            # You can now use `selected_ward` elsewhere in your program
            return selected_ward
        except tk._tkinter.TclError:
            print("No ward number selected.")
            return None
        
    def create_pdf(self): 
        pdf = FPDF()
        pdf.add_page()

        return pdf
     
    def _run(self):
        # Validate inputs
        selected_dir = self.selected_dir_path.get().strip()
        out_dir = self.output_dir.get().strip()
        ward_number = self.ward_number.get()

        if not ward_number: 
            messagebox.showwarning("Missing Info", "Please input a ward number.")
            return
        if (ward_number < 1 | ward_number > 25): 
            messagebox.showwarning("Missing Info", "Please input a valid ward number.")
            return    
        if not selected_dir or not os.path.isdir(selected_dir):
            messagebox.showwarning("Missing Info", "Please select a valid Directory.")
            return
        if not out_dir:
            messagebox.showwarning("Missing Info", "Please select an output folder.")
            return
  
        # Run Application
        try:
            pdf = self.create_pdf()
            df, error = load_data(selected_dir)
            clean_data(df)
            # filter the df to find only selected ward info
            ward_df = filter_ward(df, ward_number)
            print(ward_df)

            # Add a heading
            pdf.set_font('Arial', 'B', 14)  # Bold, size 14
            pdf.cell(0, 10, 'Ward ' + str(ward_number) + " WOW Anaylsis", ln=True, align='C')  # Centered


            # Add Chart Title
            pdf.set_font('Arial', '', 12)  # Regular weight, size 12
            pdf.cell(0, 10, 'Request Type Pie Chart:', ln=True)

            # chart the total number of each request type in ward during time period 
            chart_request_counts_for_ward(ward_df, ward_number, pdf)

            # gather the portion of each service request type per each year
            counts = calc_annual_portions(ward_df)
    
            # use linear regression to plot request type for selected ward over the years
            calc_rising_trends_porportion(counts, pdf)

            # use raw counts to plot request type for selected ward ver the years 
            # calc_rising_trends_raw_counts(counts, pdf)

            ward_to_city_comparison(df, ward_df, ward_number)

            calc_drifting_locations(ward_df)

            pdf.output("report.pdf")

        except Exception as e:
            messagebox.showerror("Data Loading Error", str(e))
            return

        if error:
            messagebox.showerror("DIR Error", error)
        else:
            messagebox.showinfo(
                "Success",
                f"Done! Loaded data from {len(list(Path(selected_dir).glob('*.csv')))} CSV files"
            )
            self.destroy() 


if __name__ == "__main__":
    app = wowGeneratorApp()
    app.mainloop()