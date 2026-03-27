# generate_ward7_wow.py
# Ward 7 WOW Factor Analysis — Combined Version
#
# outputs:
# 1.  year-over-year trend chart per category (2018-2025) = yoy_trends.png [Instruction #1]
# 2.  monthly category trend chart = ward7_category_trends_monthly.png [Instruction #1]
# 3.  top 5 rising categories (annual) = top_5_rising_annual.csv + top_5_rising_annual.png [Instruction #1]
# 4.  top 5 rising categories (monthly) = top_5_rising_monthly.csv + top_5_rising_monthly.png [Instruction #1]
# 5.  top 3 drifting locations = top_3_drifting_locations.csv + top_3_drifting_locations_trend.png [Instruction #1]
# 6.  categories rising faster than city = categories_faster_than_city.csv + categories_faster_than_city.png [Instruction #1/#3]
# 7.  repeated complaint locations = repeated_complaint_locations.csv + repeated_complaint_locations.png [Instruction #1]
# 8.  repeated complaints stacked = repeated_complaints_stacked.png [Instruction #1]
# 9.  ward 7 vs citywide (year-by-year proportional) = ward7_vs_city_trend.png [Instruction #3]
# 10. ward 7 vs city (ratio snapshot) = ward7_vs_city_ratio.png + ward7_vs_city_baseline.csv [Instruction #3]
# 11. ward 7 vs its own history = ward7_vs_its_own_history.csv + ward7_vs_its_own_history.png [Instruction #2]
# 12. consecutive month risers = third_consecutive_month_risers.csv [Instruction #2]
# 13. category heatmap = ward7_category_heatmap.png [Instruction #1]
# 14. top 5 unified hotspots = top_5_hotspots.csv + top_5_hotspots.png [Instruction #4]
# 15. 2026 early warning = early_warning_2026.csv [Instruction #1]
# 16. signal summary = ward7_story.txt [Instruction #5/#2]
#
# needs capture_report.py in the same folder and SR csvs at DATA_DIR

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats

# pulling rules from capture_report.py so everything stays in sync if rules change
from capture_report import EXACT_RULES

# maps internal rule keys to readable names for charts and output
# leaving out Is_Parking and Is_Admin - not civic stress signals worth tracking here
CATEGORY_MAP = {
    "Is_Waste":       "Waste",
    "Is_Roads":       "Roads",
    "Is_Water_Sewer": "Water/Sewer",
    "Is_Property":    "Property",
    "Is_Environment": "Trees",
    "Is_Animal":      "Animal",
    "Is_Noise":       "Noise",
}

# all signal columns including parking and admin needed for monthly aggregation functions
SIGNAL_COLS = list(EXACT_RULES.keys()) + ["Is_Other"]

# column name constants
DATE_COL   = "Creation Date"
WARD_COL   = "Ward"
TYPE_COL   = "Service Request Type"
POSTAL_COL = "First 3 Chars of Postal Code"
INT1_COL   = "Intersection Street 1"
INT2_COL   = "Intersection Street 2"

# ward filter - regex pattern matches "(07)" in the ward column
# using pattern matching instead of exact labels so this can eventually be
# parameterized for any ward number when the app version gets built
TARGET_WARD_NUM = "07"
WARD_PATTERN    = rf"\({TARGET_WARD_NUM}\)"

# update DATA_DIR to point at wherever your SR csvs live
DATA_DIR   = Path(r"C:\Users\nikki\projects\capstone-pca-and-clustering\data")
OUTPUT_DIR = Path("ward7_wow_outputs")
PLOTS_DIR  = OUTPUT_DIR / "plots"
DATA_DIR_OUT = OUTPUT_DIR / "data"

# time windows for annual analysis
# baseline is 2018-2024 (7 full years of "normal" for ward 7)
# recent is just 2025, one clean full year to compare against that baseline
# 2026 is kept separate since it's a partial year, only used in early warning
FULL_YEARS     = list(range(2018, 2026))
BASELINE_YEARS = list(range(2018, 2025))
RECENT_YEARS   = [2025]
PARTIAL_YEAR   = 2026

# time windows for monthly analysis
# recent = last 3 months, baseline = prior 12 months
RECENT_MONTHS   = 3
BASELINE_MONTHS = 12
DRIFT_MONTHS    = 6

# thresholds
MIN_REPEAT_THRESHOLD = 3
MIN_CITY_BASELINE    = 5
TOP_N_RISING         = 5
TOP_N_DRIFTING       = 3
TOP_N_HOTSPOTS       = 5


# ETL layer: EVENT
# each row in the raw csvs is a 311 service request = one event (timestamped, located, typed)
def load_data() -> pd.DataFrame:
    all_dfs = []

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}/")

    for f in csv_files:
        print(f"  Loading {f.name}...")
        try:
            df = pd.read_csv(
                f,
                encoding="latin1",   # some street names have special characters
                engine="python",
                usecols=range(9),
                on_bad_lines="skip",
            )
            df = df.iloc[:, :9]

            # column names vary slightly across years so rename them all consistently
            df.columns = [
                DATE_COL,
                "Status",
                POSTAL_COL,
                INT1_COL,
                INT2_COL,
                WARD_COL,
                TYPE_COL,
                "Division",
                "Section",
            ]
            all_dfs.append(df)
        except Exception as e:
            print(f"  WARNING: Could not load {f.name}: {e}")

    df = pd.concat(all_dfs, ignore_index=True)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TYPE_COL]).copy()

    # filter to 2018+ - some files contain pre-restructure rows
    df = df[df[DATE_COL].dt.year >= 2018].copy()

    # pull these out once so every function downstream can reference them directly
    df["Year"]       = df[DATE_COL].dt.year
    df["Month"]      = df[DATE_COL].dt.month
    df["year_month"] = df[DATE_COL].dt.to_period("M").astype(str)

    print(f"\n  Total rows loaded (2018+): {len(df):,}")
    return df


# ETL layer: MEASURE
# turns raw event text (service request type) into structured category counts
def apply_categories(df: pd.DataFrame) -> pd.DataFrame:
    # build a reverse lookup dict: service request type -> category key
    # faster than looping through all of EXACT_RULES for every single row
    # if a type shows up in multiple categories, first match wins (follows CATEGORY_MAP order)
    type_to_cat = {}
    for cat_key in CATEGORY_MAP:
        for srt in EXACT_RULES.get(cat_key, set()):
            if srt not in type_to_cat:
                type_to_cat[srt] = cat_key

    def assign(srt):
        return type_to_cat.get(srt, "other")

    df["category_key"]  = df[TYPE_COL].map(assign)
    df["category_name"] = df["category_key"].map(lambda k: CATEGORY_MAP.get(k, "Other"))

    # also apply boolean columns needed by the monthly aggregation functions
    # both sets of columns get created in one pass so nothing has to run twice
    for col, names in EXACT_RULES.items():
        df[col] = df[TYPE_COL].isin(names)
    df["Is_Other"] = ~df[list(EXACT_RULES.keys())].any(axis=1)

    # if this number is low the rules aren't catching enough service request types
    coverage = (df["category_key"] != "other").mean() * 100
    print(f"  Category coverage: {coverage:.1f}%")

    return df


def apply_micro_location(df: pd.DataFrame) -> pd.DataFrame:
    def build_micro_location(row):
        """
        This will build a usable micro-location field from the location columns.

        What it does:
        - Will use intersection streets if they are both available. 
        - If intersection is missing, it'll fall back to postal prefix.
        - Will fall back to one street name if it is needed.
        - If nothing usable is found, we'll return UNKNOWN_LOCATION 

        Why it matters: 
        - The sponsor specifically wants drifting locations, repeated complaint locations, and hotspot style microareas. 

        Ward 7 WOW Factor connection:
        - This will support top 3 drifting locations 
        - Will also support repeated complaint locations 
        - Will also support top 5 hotspots / microareas 
        """
        i1     = str(row.get(INT1_COL, "")).strip()
        i2     = str(row.get(INT2_COL, "")).strip()
        postal = str(row.get(POSTAL_COL, "")).strip()

        if i1 and i1.lower() != "nan" and i2 and i2.lower() != "nan":
            return f"{i1} & {i2}"
        if postal and postal.lower() != "nan":
            return postal
        if i1 and i1.lower() != "nan":
            return i1
        if i2 and i2.lower() != "nan":
            return i2
        return "UNKNOWN_LOCATION"

    df["micro_location"] = df.apply(build_micro_location, axis=1)
    return df


def split_ward7(df: pd.DataFrame):
    # returns both so city is available for the ward 7 vs citywide comparison later
    # pattern matching on "(07)" so this is easy to parameterize for other wards later
    ward7 = df[df[WARD_COL].astype(str).str.contains(WARD_PATTERN, na=False)].copy()
    print(f"  Ward 7 rows: {len(ward7):,}  |  City rows: {len(df):,}")
    return ward7, df


# helper functions
def _pct_change(current, baseline):
    if pd.isna(baseline) or baseline == 0:
        return np.nan
    return (current - baseline) / baseline * 100.0


def _safe_ratio(a, b):
    if pd.isna(b) or b == 0:
        return np.nan
    return a / b


def monthly_signal_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-bucket categories are aggregated by month 

    What it does:
    - We will sum each category bucket by year_month
    - Reshaping results into long format with columns:
        year_month, signal, count 

    Why it matters:
    - Category flags are converted to time-series measures 
    - This is what makes it possible to detect rising signals and historical deviations 

    Ward 7 WOW Factor connection:
    - Will support top 5 rising categories 
    - Will support Ward 7 vs history 
    - Will support Ward 7 vs city baseline 
    - Will support this consecutive month drift detection 
    """
    monthly = (
        df.groupby("year_month")[SIGNAL_COLS]
        .sum()
        .reset_index()
        .melt(id_vars="year_month", var_name="signal", value_name="count")
    )
    return monthly


def get_recent_and_baseline(months, recent_n=3, baseline_n=12):
    months = sorted(pd.Period(m, freq="M") for m in months)
    recent = months[-recent_n:]
    baseline = months[-(recent_n + baseline_n):-recent_n]
    return [str(x) for x in recent], [str(x) for x in baseline]


def save_plot(fig, filename: str):
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ETL layer: SIGNAL
# takes the category measures and computes change detection metrics

def compute_yoy_trends(ward7: pd.DataFrame) -> pd.DataFrame:
    # group by year + category, count rows, pivot categories into columns
    # gives a clean year x category table that everything downstream builds on
    # 2026 excluded here - partial year would make the trend line look like it drops off
    full = ward7[ward7["Year"].isin(FULL_YEARS)]
    pivot = (
        full.groupby(["Year", "category_name"])
        .size()
        .unstack(fill_value=0)
    )
    # if a category had zero complaints across all years it won't appear as a column
    # add it back as zeros so nothing crashes downstream
    for name in CATEGORY_MAP.values():
        if name not in pivot.columns:
            pivot[name] = 0
    return pivot[list(CATEGORY_MAP.values())]


def plot_yoy_trends(pivot: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10.colors

    for i, cat in enumerate(pivot.columns):
        ax.plot(pivot.index, pivot[cat], marker="o", label=cat,
                color=colors[i % len(colors)], linewidth=2)

    ax.set_title("Ward 7 — Service Requests by Category (2018–2025)", fontsize=14)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Requests")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xticks(FULL_YEARS)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    save_plot(fig, "yoy_trends.png")
    print("  Saved: yoy_trends.png")


def plot_monthly_trends(ward7: pd.DataFrame):
    # monthly trend chart - more granular than annual, shows seasonal patterns
    monthly = monthly_signal_counts(ward7)
    if monthly.empty:
        return

    # only plot the top 5 categories by total volume to keep it readable
    top_signals = (
        monthly.groupby("signal")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index
        .tolist()
    )

    # filter to only categories in CATEGORY_MAP
    top_signals = [s for s in top_signals if s in CATEGORY_MAP]

    plot_df = monthly[monthly["signal"].isin(top_signals)].copy()
    plot_df["period"] = pd.PeriodIndex(plot_df["year_month"], freq="M").to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 7))
    for signal in top_signals:
        signal_df = plot_df[plot_df["signal"] == signal].sort_values("period")
        label = CATEGORY_MAP.get(signal, signal)
        ax.plot(signal_df["period"], signal_df["count"], marker="o", label=label)

    ax.set_title("Ward 7 Category Trends by Month (2018–2025)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.legend()
    # show every month as Jan-18 style, rotated 90 degrees so they don't overlap
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    plt.xticks(rotation=90)
    plt.tight_layout()
    save_plot(fig, "ward7_category_trends_monthly.png")
    print("  Saved: ward7_category_trends_monthly.png")


def compute_signals_annual(pivot: pd.DataFrame) -> dict:
    # for each category a few different metrics are computed:
    #
    # slope - linear regression across all 8 years, gives us the long term trend direction
    # using regression instead of just first vs last year because it uses all the data
    # and is way more reliable than just comparing two endpoints
    #
    # pct_change - how much 2025 differs from the 2018-2024 average
    #
    # z_score - how many standard deviations 2025 is from the baseline distribution
    # this is important because some categories swing a lot naturally (roads in winter)
    # a big pct_change for those isn't actually unusual, z_score catches that difference
    #
    # r_squared and p_value - stored for future analysis, not used in classify_signal yet
    # r_squared shows how well the trend line fits, p_value shows if the trend is real

    signals = {}
    baseline_idx = [y for y in BASELINE_YEARS if y in pivot.index]
    recent_idx   = [y for y in RECENT_YEARS   if y in pivot.index]

    for cat in pivot.columns:
        series = pivot[cat]

        x = np.array(series.index, dtype=float)
        y = np.array(series.values, dtype=float)
        slope, _, r, p, _ = stats.linregress(x, y)

        baseline_vals = series[series.index.isin(baseline_idx)]
        recent_vals   = series[series.index.isin(recent_idx)]

        baseline_mean = baseline_vals.mean() if len(baseline_vals) else np.nan
        recent_mean   = recent_vals.mean()   if len(recent_vals)   else np.nan

        pct_change = (
            (recent_mean - baseline_mean) / baseline_mean * 100
            if baseline_mean and baseline_mean > 0 else np.nan
        )

        baseline_std = baseline_vals.std() if len(baseline_vals) > 1 else np.nan
        z = (
            (recent_mean - baseline_mean) / baseline_std
            if baseline_std and baseline_std > 0 else np.nan
        )

        signals[cat] = {
            "slope":         slope,
            "pct_change":    pct_change,
            "z_score":       z,
            "baseline_mean": baseline_mean,
            "recent_mean":   recent_mean,
            "r_squared":     r**2,
            "p_value":       p,
        }

    return signals


def classify_signal(sig: dict) -> str:
    # RISING and FALLING both require z_score AND pct_change to cross their thresholds
    # requiring both prevents false positives from categories that just naturally swing a lot
    # DRIFTING is a softer signal - uses slope to catch gradual movement that isn't alarming yet
    z   = sig["z_score"]
    pct = sig["pct_change"]
    s   = sig["slope"]

    if pd.isna(z) or pd.isna(pct):
        return "STABLE"
    if z > 1.0 and pct > 10:
        return "RISING"
    if z < -1.0 and pct < -10:
        return "FALLING"
    if s > 0 and pct > 5:
        return "DRIFTING UP"
    if s < 0 and pct < -5:
        return "DRIFTING DOWN"
    return "STABLE"


def compute_top_rising_monthly(ward_df: pd.DataFrame) -> pd.DataFrame:
    """
    This is how we find the top rising rule-bucket category in Ward 7 

    What it does:
    - Recent monthly averages for each category is computed 
    - They are compared to historical baseline period 
    - Absolute and percent changes are calculated 
    - Returns top 5 strongest risers 

    Why it matters: 
    - This is when we move the analysis from static counts to change detection 

    Ward 7 WOW Factor connection:
    - This is what directly answers the question: 'Top 5 rising categories in Ward 7'
    - Also answers the question: "What is rising"
    """
    monthly = monthly_signal_counts(ward_df)
    months  = monthly["year_month"].unique().tolist()
    recent, baseline = get_recent_and_baseline(months, RECENT_MONTHS, BASELINE_MONTHS)

    recent_avg = (
        monthly[monthly["year_month"].isin(recent)]
        .groupby("signal")["count"].mean()
        .rename("recent_avg")
    )
    baseline_avg = (
        monthly[monthly["year_month"].isin(baseline)]
        .groupby("signal")["count"].mean()
        .rename("baseline_avg")
    )

    out = pd.concat([recent_avg, baseline_avg], axis=1).fillna(0)
    out["abs_change"] = out["recent_avg"] - out["baseline_avg"]
    out["pct_change"] = out.apply(lambda r: _pct_change(r["recent_avg"], r["baseline_avg"]), axis=1)

    # filter to only the categories we care about before sorting
    # this prevents Is_Parking, Is_Admin, Is_Other from showing up in the chart
    out = out[out.index.isin(CATEGORY_MAP.keys())].copy()
    out = out.sort_values(["abs_change", "recent_avg"], ascending=False).reset_index()

    # map signal keys to display names
    out["signal"] = out["signal"].map(lambda s: CATEGORY_MAP.get(s, s))
    return out.head(TOP_N_RISING)


def plot_top_rising(rising: pd.DataFrame, filename: str, title: str):
    # reusable for both annual and monthly rising charts - just pass different filename and title
    if rising.empty:
        return

    plot_df = rising.copy().sort_values("abs_change", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["signal"], plot_df["abs_change"], color="#2563EB")
    ax.set_title(title)
    ax.set_xlabel("Recent Avg - Baseline Avg")
    ax.set_ylabel("Category")

    for i, (_, row) in enumerate(plot_df.iterrows()):
        label = f"{row['pct_change']:.1f}%" if pd.notna(row["pct_change"]) else "NA"
        ax.text(row["abs_change"], i, f"  {label}", va="center")

    save_plot(fig, filename)
    print(f"  Saved: {filename}")


def compute_top_drifting_locations(ward_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifying which locations in Ward 7 whose complaints are drifting upward over time 

    What it does:
    - Micro-location counts are built by month 
    - A slope trend is computed over the recent drift window 
    - The top 3 locations with the strongest upward drift is returned 

    Why it matters: 
    - Rather than just category-level growth, we also detect location-level emergence 

    Ward 7 WOW Factor connection:
    - This helps answer: 'Top 3 drifting locations'
    - This also helps answer: 'What's emerging spatially'
    """
    monthly = (
        ward_df.groupby(["year_month", "micro_location"])
        .size()
        .reset_index(name="count")
    )

    pivot = monthly.pivot(index="micro_location", columns="year_month", values="count").fillna(0)
    month_cols = sorted(pivot.columns.tolist(), key=lambda x: pd.Period(x, freq="M"))

    if len(month_cols) < DRIFT_MONTHS:
        return pd.DataFrame(columns=["micro_location", "slope", "recent_total"])

    month_cols = month_cols[-DRIFT_MONTHS:]
    x = np.arange(len(month_cols))

    def slope(vals):
        vals = np.asarray(vals, dtype=float)
        if np.all(vals == vals[0]):
            return 0.0
        return np.polyfit(x, vals, 1)[0]

    pivot["slope"]        = pivot[month_cols].apply(lambda row: slope(row.values), axis=1)
    pivot["recent_total"] = pivot[month_cols].sum(axis=1)

    out = (
        pivot.reset_index()[["micro_location", "slope", "recent_total"]]
        .query("micro_location != 'UNKNOWN_LOCATION'")
        .sort_values(["slope", "recent_total"], ascending=False)
        .head(TOP_N_DRIFTING)
    )
    return out


def plot_drifting_locations(ward_df: pd.DataFrame, drifting: pd.DataFrame):
    if drifting.empty:
        return

    top_locations = drifting["micro_location"].tolist()

    monthly = (
        ward_df.groupby(["year_month", "micro_location"])
        .size()
        .reset_index(name="count")
    )
    monthly = monthly[monthly["micro_location"].isin(top_locations)].copy()
    monthly["period"] = pd.PeriodIndex(monthly["year_month"], freq="M").to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 7))
    for loc in top_locations:
        loc_df = monthly[monthly["micro_location"] == loc].sort_values("period")
        ax.plot(loc_df["period"], loc_df["count"], marker="o", label=loc)

    ax.set_title("Top 3 Drifting Locations in Ward 7 (Last 6 Months Slope)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Complaint Count")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    plt.xticks(rotation=90)
    plt.tight_layout()
    save_plot(fig, "top_3_drifting_locations_trend.png")
    print("  Saved: top_3_drifting_locations_trend.png")


def compute_categories_faster_than_city(df: pd.DataFrame, ward_df: pd.DataFrame) -> pd.DataFrame:
    """
    This is where we compare Ward 7 category growth against city-wide category growth 

    What it does:
    - Recent vs baseline growth for Ward 7 is computed 
    - Recent vs baseline growth for full city is computed 
    - Difference between Ward 7 growth and city growth is calculated 
    - Categories where Ward 7 rises faster is returned 

    Why it matters:
    - This will tell us if Ward 7 is behaving differently from the rest of the city 

    Ward 7 WOW Factor connection:
    - This answers the question: "Any category rising faster than the city baseline"
    - This answers the question: "Show where Ward 7 is behaving differently from the rest of the city"
    """
    ward_monthly = monthly_signal_counts(ward_df)
    city_monthly = monthly_signal_counts(df)

    months = city_monthly["year_month"].unique().tolist()
    recent, baseline = get_recent_and_baseline(months, RECENT_MONTHS, BASELINE_MONTHS)

    ward_recent  = ward_monthly[ward_monthly["year_month"].isin(recent)].groupby("signal")["count"].mean().rename("ward_recent")
    ward_base    = ward_monthly[ward_monthly["year_month"].isin(baseline)].groupby("signal")["count"].mean().rename("ward_baseline")
    city_recent  = city_monthly[city_monthly["year_month"].isin(recent)].groupby("signal")["count"].mean().rename("city_recent")
    city_base    = city_monthly[city_monthly["year_month"].isin(baseline)].groupby("signal")["count"].mean().rename("city_baseline")

    out = pd.concat([ward_recent, ward_base, city_recent, city_base], axis=1).fillna(0)
    out = out[out["city_baseline"] >= MIN_CITY_BASELINE]

    out["ward_pct_change"] = out.apply(lambda r: _pct_change(r["ward_recent"], r["ward_baseline"]), axis=1)
    out["city_pct_change"] = out.apply(lambda r: _pct_change(r["city_recent"], r["city_baseline"]), axis=1)
    out["growth_diff"]     = out["ward_pct_change"] - out["city_pct_change"]

    out = out.sort_values("growth_diff", ascending=False).reset_index()
    out["signal"] = out["signal"].map(lambda s: CATEGORY_MAP.get(s, s))
    return out[out["growth_diff"] > 0]


def plot_categories_faster_than_city(faster: pd.DataFrame):
    if faster.empty:
        return

    plot_df = faster.head(8).copy()
    x = np.arange(len(plot_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width / 2, plot_df["ward_pct_change"], width, label="Ward 7", color="#2563EB")
    ax.bar(x + width / 2, plot_df["city_pct_change"], width, label="City",   color="#F97316")

    ax.set_title("Categories Rising Faster Than City Baseline (Last 3 Months vs Prior 12)")
    ax.set_xlabel("Category")
    ax.set_ylabel("% Change vs Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["signal"], rotation=30, ha="right")
    ax.legend()
    save_plot(fig, "categories_faster_than_city.png")
    print("  Saved: categories_faster_than_city.png")


def compute_repeated_complaints(ward_df: pd.DataFrame) -> pd.DataFrame:
    """
    Finding repeated complaints over the last year within Ward 7 locations.

    What it does:
    - Persistent local problems are highlighted rather than just one-off events 

    Ward 7 WOW Factor connection:
    - This will answer the question: "Any building or block showing repeated complaints"
    """
    recent_cutoff = ward_df[DATE_COL].max() - pd.DateOffset(months=12)
    recent_df = ward_df[ward_df[DATE_COL] >= recent_cutoff].copy()

    out = (
        recent_df.groupby(["micro_location", TYPE_COL])
        .size()
        .reset_index(name="count")
        .query("micro_location != 'UNKNOWN_LOCATION' and count >= @MIN_REPEAT_THRESHOLD")
        .sort_values("count", ascending=False)
    )
    return out


def plot_repeated_complaints(repeated: pd.DataFrame):
    if repeated.empty:
        return

    plot_df = repeated.head(10).copy()
    plot_df["label"] = plot_df["micro_location"] + " | " + plot_df[TYPE_COL]
    plot_df = plot_df.sort_values("count", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(plot_df["label"], plot_df["count"], color="#2563EB")
    ax.set_title("Repeated Complaint Locations in Ward 7 (Last 12 Months)")
    ax.set_xlabel("Complaint Count")
    ax.set_ylabel("Location | Service Request Type")
    save_plot(fig, "repeated_complaint_locations.png")
    print("  Saved: repeated_complaint_locations.png")


def compute_city_comparison_annual(ward7: pd.DataFrame, city: pd.DataFrame) -> pd.DataFrame:
    # computes ward 7 vs city proportion per category per year
    # proportions not raw counts - ward 7 is 1 of 25 wards so raw counts are always lower
    # the real question is whether a category is overrepresented in ward 7
    # year-by-year view shows how that gap is changing over time
    rows = []
    for yr in FULL_YEARS:
        w7_yr   = ward7[ward7["Year"] == yr]
        city_yr = city[city["Year"] == yr]

        if len(w7_yr) == 0 or len(city_yr) == 0:
            continue

        for cat_key, cat_name in CATEGORY_MAP.items():
            w7_count   = (w7_yr["category_key"]   == cat_key).sum()
            city_count = (city_yr["category_key"] == cat_key).sum()

            rows.append({
                "year":      yr,
                "category":  cat_name,
                "pct_ward7": w7_count   / len(w7_yr)   * 100,
                "pct_city":  city_count / len(city_yr) * 100,
            })

    return pd.DataFrame(rows)


def plot_city_comparison_annual(comp: pd.DataFrame):
    # small multiples - one subplot per category showing ward 7 vs city % over time
    # sharey=False so each gets its own y axis scale
    # waste is ~37%, trees is ~0.6% - shared axis would flatten the small ones
    cats = list(CATEGORY_MAP.values())
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=False)
    axes = axes.flatten()

    for i, cat in enumerate(cats):
        ax  = axes[i]
        sub = comp[comp["category"] == cat].sort_values("year")

        ax.plot(sub["year"], sub["pct_ward7"], marker="o",
                label="Ward 7", color="#2563EB", linewidth=2)
        ax.plot(sub["year"], sub["pct_city"],  marker="s",
                label="Citywide", color="#F97316", linewidth=2, linestyle="--")

        ax.set_title(cat, fontsize=10, fontweight="bold")
        ax.set_xticks(FULL_YEARS)
        ax.set_xticklabels([str(y)[2:] for y in FULL_YEARS], fontsize=7)
        ax.set_ylabel("% of requests", fontsize=7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        if i == 0:
            ax.legend(fontsize=7)

    axes[-1].set_visible(False)
    fig.suptitle("Ward 7 vs Citywide — Category Share Over Time", fontsize=13)
    fig.text(0.5, 0.01, "Year", ha="center", fontsize=10)
    save_plot(fig, "ward7_vs_city_trend.png")
    print("  Saved: ward7_vs_city_trend.png")


def compute_city_ratio_snapshot(df: pd.DataFrame, ward_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ward 7 category proportions against city-wide proportions are compared 

    What it does:
    - Total counts by category for Ward 7 are computed 
    - Total counts by category for full city are computed 
    - Both proportions are converted 
    - Ward 7 / city proportion ratio is computed 

    Why it matters:
    - This will show which issue types are overrepresented within Ward 7 

    Ward 7 WOW connection:
    - This will answer: "Compare Ward 7 to the city baseline"
    - This will answer: "Show us where Ward 7 is behaving differently from the rest of the city"
    """
    ward_monthly = monthly_signal_counts(ward_df)
    city_monthly = monthly_signal_counts(df)

    ward_totals = ward_monthly.groupby("signal")["count"].sum().rename("ward_count")
    city_totals = city_monthly.groupby("signal")["count"].sum().rename("city_count")

    out = pd.concat([ward_totals, city_totals], axis=1).fillna(0)
    out["ward_prop"]        = out["ward_count"] / out["ward_count"].sum()
    out["city_prop"]        = out["city_count"] / out["city_count"].sum()
    out["ward_to_city_ratio"] = out.apply(lambda r: _safe_ratio(r["ward_prop"], r["city_prop"]), axis=1)

    out = out.sort_values("ward_to_city_ratio", ascending=False).reset_index()
    out["signal"] = out["signal"].map(lambda s: CATEGORY_MAP.get(s, s))
    return out


def plot_city_ratio_snapshot(city_cmp: pd.DataFrame):
    if city_cmp.empty:
        return

    # filter to only civic signal categories, exclude Is_Admin, Is_Parking, Is_Other
    valid_names = set(CATEGORY_MAP.values())
    plot_df = city_cmp[city_cmp["signal"].isin(valid_names)].copy()
    plot_df = plot_df.sort_values("ward_to_city_ratio", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["signal"], plot_df["ward_to_city_ratio"], color="#2563EB")
    ax.set_title("Ward 7 vs City Baseline — Proportion Ratio (2018–2025)")
    ax.set_xlabel("Ward 7 / City Ratio")
    ax.set_ylabel("Category")
    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1)
    save_plot(fig, "ward7_vs_city_ratio.png")
    print("  Saved: ward7_vs_city_ratio.png")


def compute_hotspots(ward_df: pd.DataFrame) -> pd.DataFrame:
    """
    Strong complaint hotspots in Ward 7 are identified

    What it does:
    - Looking only at recent months 
    - Counts are aggregated by micro-location 
    - Total recent complaint volume and last-month growth are measured 
    - Top 5 hotspot areas are returned 

    Why it matters:
    - This will provide us with microarea view that councillors can act on directly 

    Ward 7 WOW Factor connection:
    - This will answer the question: "Find the top 5 microareas in Ward 7 with the highest concentration or fastest growth"
    """
    recent_cutoff = ward_df[DATE_COL].max() - pd.DateOffset(months=6)
    recent_df = ward_df[ward_df[DATE_COL] >= recent_cutoff].copy()

    monthly = (
        recent_df.groupby(["year_month", "micro_location"])
        .size()
        .reset_index(name="count")
    )

    pivot = monthly.pivot(index="micro_location", columns="year_month", values="count").fillna(0)
    month_cols = sorted(pivot.columns.tolist(), key=lambda x: pd.Period(x, freq="M"))

    if not month_cols:
        return pd.DataFrame(columns=["micro_location", "total_count", "last_month_growth"])

    pivot["total_count"]      = pivot[month_cols].sum(axis=1)
    pivot["last_month_growth"] = pivot[month_cols[-1]] - pivot[month_cols[-2]] if len(month_cols) >= 2 else 0

    out = (
        pivot.reset_index()[["micro_location", "total_count", "last_month_growth"]]
        .query("micro_location != 'UNKNOWN_LOCATION'")
        .sort_values(["total_count", "last_month_growth"], ascending=False)
        .head(TOP_N_HOTSPOTS)
    )
    return out


def plot_hotspots(hotspots: pd.DataFrame):
    if hotspots.empty:
        return

    plot_df = hotspots.copy().sort_values("total_count", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["micro_location"], plot_df["total_count"], color="#2563EB")
    ax.set_title("Top 5 Hotspots in Ward 7 (Last 6 Months)")
    ax.set_xlabel("Recent Total Complaints")
    ax.set_ylabel("Micro-location")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    save_plot(fig, "top_5_hotspots.png")
    print("  Saved: top_5_hotspots.png")


def plot_ward_vs_history(history: pd.DataFrame):
    if history.empty:
        return

    plot_df = history.copy()
    # map signal keys to display names
    plot_df["signal"] = plot_df["signal"].map(lambda s: CATEGORY_MAP.get(s, s))
    plot_df = plot_df.sort_values("pct_vs_history", ascending=True)

    # dark blue = flagged as unusual (z-score > 1.0 AND pct change > 10%)
    # light blue = within normal range of variation for that category
    colors = ["#2563EB" if flag else "#93C5FD" for flag in plot_df["is_unusual"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["signal"], plot_df["pct_vs_history"], color=colors)
    ax.set_title("Ward 7 vs Its Own Historical Baseline (Last 3 Months vs Prior 12)")
    ax.set_xlabel("% Above / Below Historical Baseline")
    ax.set_ylabel("Category")
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)

    # legend explaining the two colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2563EB", label="Unusual (z-score > 1 AND pct change > 10%)"),
        Patch(facecolor="#93C5FD", label="Within normal range of variation"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    save_plot(fig, "ward7_vs_its_own_history.png")


def plot_category_heatmap(ward_df: pd.DataFrame):
    # filter to full years only - 2026 is partial and skews the color scale
    ward_full = ward_df[ward_df["Year"].isin(FULL_YEARS)]
    monthly = monthly_signal_counts(ward_full)
    if monthly.empty:
        return

    pivot = monthly.pivot(index="signal", columns="year_month", values="count").fillna(0)

    top_signals = pivot.sum(axis=1).sort_values(ascending=False).head(8).index
    pivot = pivot.loc[top_signals]

    # map signal keys to display names where available
    pivot.index = [CATEGORY_MAP.get(s, s) for s in pivot.index]

    # show every month as Jan-18 style labels rotated 90 degrees
    month_cols = list(pivot.columns)
    tick_labels = [pd.Period(m, freq="M").strftime("%b-%y") for m in month_cols]

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_title("Ward 7 Category Heatmap by Month (2018–2025)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Category")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(np.arange(len(month_cols)))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)

    fig.colorbar(im, ax=ax, label="Count")
    save_plot(fig, "ward7_category_heatmap.png")


def plot_repeated_complaints_stacked(repeated: pd.DataFrame):
    if repeated.empty:
        return

    plot_df = repeated.head(20).copy()
    pivot = plot_df.pivot_table(
        index="micro_location",
        columns=TYPE_COL,
        values="count",
        aggfunc="sum",
        fill_value=0
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(len(pivot))

    for col in pivot.columns:
        vals = pivot[col].values
        ax.bar(pivot.index, vals, bottom=bottom, label=col)
        bottom += vals

    ax.set_title("Repeated Complaints by Location and Type (Last 12 Months)")
    ax.set_xlabel("Micro-location")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")

    save_plot(fig, "repeated_complaints_stacked.png")


def compute_2026_early_warning(ward7: pd.DataFrame) -> pd.DataFrame:
    # looks at whatever months exist in 2026 and compares them to the same months
    # in prior years - gives early signals on what might be worth watching
    #
    # same-month comparison is important here - comparing against full year averages
    # would make 2026 look artificially low since it's a partial year
    #
    # groupby year first then average gives "typical jan-mar count per year"
    # rather than just summing all jan-mar rows across all years

    w26 = ward7[ward7["Year"] == PARTIAL_YEAR]
    months_in_2026 = sorted(w26["Month"].unique())

    if len(months_in_2026) == 0:
        return pd.DataFrame()

    rows = []
    for cat_key, cat_name in CATEGORY_MAP.items():
        count_2026 = (
            (w26["Month"].isin(months_in_2026)) &
            (w26["category_key"] == cat_key)
        ).sum()

        hist = ward7[
            (ward7["Year"].isin(FULL_YEARS)) &
            (ward7["Month"].isin(months_in_2026)) &
            (ward7["category_key"] == cat_key)
        ]
        hist_avg = hist.groupby("Year").size().mean()

        pct_diff = (
            (count_2026 - hist_avg) / hist_avg * 100
            if hist_avg > 0 else np.nan
        )

        rows.append({
            "Category":       cat_name,
            "2026_count":     count_2026,
            "historical_avg": round(hist_avg, 1),
            "pct_vs_hist":    round(pct_diff, 1) if not pd.isna(pct_diff) else "N/A",
        })

    return pd.DataFrame(rows).sort_values("pct_vs_hist", ascending=False)


def ward7_vs_own_history(ward_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ward 7's recent category behavior is compared to its own historical baseline 

    What it does:
    - Recent category averages are computed 
    - Baseline category averages are computed 
    - Percent difference vs Ward 7 history is calculated 
    - Z-score is computed using the std dev across the 12 baseline months
    - Only categories that cross both thresholds (z > 1.0 AND pct > 10%) are flagged as unusual
    - All categories are returned so the chart is always informative

    Why it matters:
    - Pct_change alone is not enough - a category with a low baseline can show a huge
      percent swing from just a few extra complaints
    - Z-score anchors the result to how much that category normally varies month to month
    - Requiring both gates matches the same standard used in compute_signals_annual
      and classify_signal, keeping all "unusual" claims consistent across the pipeline

    Ward 7 WOW Factor connection:
    - This helps answer: "Compare Ward 7 to its own history"
    - This helps answer: "Show us what is not normal for Ward 7"
    - Supports story examples like: "Noise complaints are 18% above the Ward 7 historical baseline"

    """

    monthly = monthly_signal_counts(ward_df)
    months = monthly["year_month"].unique().tolist()
    recent, baseline = get_recent_and_baseline(months, RECENT_MONTHS, BASELINE_MONTHS)

    recent_monthly  = monthly[monthly["year_month"].isin(recent)]
    baseline_monthly = monthly[monthly["year_month"].isin(baseline)]

    recent_avg   = recent_monthly.groupby("signal")["count"].mean().rename("recent_avg")
    baseline_avg = baseline_monthly.groupby("signal")["count"].mean().rename("baseline_avg")

    # std dev across the individual baseline months - this is the denominator for z-score
    # using per-month std, not std of a single mean, so it reflects real month-to-month
    # variability for that category rather than assuming everything is equally volatile
    baseline_std = baseline_monthly.groupby("signal")["count"].std().rename("baseline_std")

    out = pd.concat([recent_avg, baseline_avg, baseline_std], axis=1).fillna(0)
    out["pct_vs_history"] = out.apply(lambda r: _pct_change(r["recent_avg"], r["baseline_avg"]), axis=1)
    out["z_score"] = out.apply(
        lambda r: (r["recent_avg"] - r["baseline_avg"]) / r["baseline_std"]
        if r["baseline_std"] > 0 else np.nan,
        axis=1,
    )
    out["abs_change"] = out["recent_avg"] - out["baseline_avg"]

    # is_unusual flags the dual-gate (z > 1.0 AND pct > 10) without filtering the dataframe
    # all categories are returned so the chart is always informative even when nothing is alarming
    # the flag is what drives story-level claims - only is_unusual=True categories get called out
    # this matches how compute_signals_annual + classify_signal work: return everything, label it
    out["is_unusual"] = (out["z_score"] > 1.0) & (out["pct_vs_history"] > 10)

    return out.sort_values("pct_vs_history", ascending=False).reset_index()


def consecutive_month_risers(ward_df: pd.DataFrame) -> pd.DataFrame:
    """
    This will detect which categories have risen for three consecutive months within Ward 7 

    What it does:
    - Monthly category counts are built 
    - Z-score and pct_change vs the prior 12-month baseline are computed first
    - Only categories that pass the dual gate (z > 1.0 AND pct > 10%) are considered
    - Among those, checking if the last three months form a strictly increasing sequence
    - Categories with both statistically meaningful AND sustained upward drift is returned

    Why it matters:
    - Without the z-score gate, any category that ticks up by 1-2 complaints over three
      months would flag as a riser - that is noise, not a signal
    - The dual gate ensures the 3-month window is genuinely elevated above historical norms
      before we make the "drifting upward for the third consecutive month" claim
    - This matches the same standard used in compute_signals_annual and ward7_vs_own_history
      so all signal-level claims in the pipeline are held to the same bar

    Ward 7 WOW Factor connection:
    - Supports examples such as: "Waste complaints are drifting upward for the third consecutive month"

    """

    monthly = monthly_signal_counts(ward_df)
    months = monthly["year_month"].unique().tolist()
    recent_months, baseline_months = get_recent_and_baseline(months, RECENT_MONTHS, BASELINE_MONTHS)

    recent_monthly   = monthly[monthly["year_month"].isin(recent_months)]
    baseline_monthly = monthly[monthly["year_month"].isin(baseline_months)]

    recent_avg   = recent_monthly.groupby("signal")["count"].mean().rename("recent_avg")
    baseline_avg = baseline_monthly.groupby("signal")["count"].mean().rename("baseline_avg")
    baseline_std = baseline_monthly.groupby("signal")["count"].std().rename("baseline_std")

    stats_df = pd.concat([recent_avg, baseline_avg, baseline_std], axis=1).fillna(0)
    stats_df["pct_change"] = stats_df.apply(
        lambda r: _pct_change(r["recent_avg"], r["baseline_avg"]), axis=1
    )
    stats_df["z_score"] = stats_df.apply(
        lambda r: (r["recent_avg"] - r["baseline_avg"]) / r["baseline_std"]
        if r["baseline_std"] > 0 else np.nan,
        axis=1,
    )

    # only categories that are genuinely elevated can qualify as consecutive risers
    qualified = stats_df[
        (stats_df["z_score"] > 1.0) & (stats_df["pct_change"] > 10)
    ].index.tolist()

    if not qualified:
        return pd.DataFrame(columns=["signal", "m1", "m2", "m3"])

    pivot = monthly.pivot(index="signal", columns="year_month", values="count").fillna(0)
    month_cols = sorted(pivot.columns.tolist(), key=lambda x: pd.Period(x, freq="M"))

    if len(month_cols) < 3:
        return pd.DataFrame(columns=["signal", "m1", "m2", "m3"])

    last3 = month_cols[-3:]

    # filter to qualified signals first, then check for consecutive increase
    pivot_qualified = pivot.loc[pivot.index.isin(qualified)]
    out = pivot_qualified[
        (pivot_qualified[last3[0]] < pivot_qualified[last3[1]]) &
        (pivot_qualified[last3[1]] < pivot_qualified[last3[2]])
    ].reset_index()

    return out.rename(columns={last3[0]: "m1", last3[1]: "m2", last3[2]: "m3"})[["signal", "m1", "m2", "m3"]]


def generate_story(signals_annual: dict, hotspots: pd.DataFrame,
                   warn_df: pd.DataFrame, drifting_locs: pd.DataFrame,
                   repeated: pd.DataFrame, city_ratio: pd.DataFrame) -> str:
    # formats everything into a numbers-first text summary
    # classify each category and sort into three lists using annual signal detection

    rising   = [(c, s) for c, s in signals_annual.items() if classify_signal(s) == "RISING"]
    drifting = [(c, s) for c, s in signals_annual.items() if "DRIFTING" in classify_signal(s)]
    falling  = [(c, s) for c, s in signals_annual.items() if classify_signal(s) == "FALLING"]

    lines = []
    lines.append("=" * 60)
    lines.append("WARD 7 -- SIGNAL SUMMARY (2018-2025)")
    lines.append("=" * 60)

    lines.append("")
    lines.append("RISING  (2025 vs 2018-2024 baseline):")
    lines.append("-" * 40)
    if rising:
        for cat, s in sorted(rising, key=lambda x: -x[1]["pct_change"]):
            lines.append(
                f"  {cat:<14}  +{s['pct_change']:.0f}%"
                f"  |  {s['baseline_mean']:.0f} -> {s['recent_mean']:.0f} requests/yr"
            )
    else:
        lines.append("  None")

    lines.append("")
    lines.append("DRIFTING (annual trend):")
    lines.append("-" * 40)
    if drifting:
        for cat, s in sorted(drifting, key=lambda x: -abs(x[1]["pct_change"])):
            direction = "up" if "UP" in classify_signal(s) else "down"
            lines.append(
                f"  {cat:<14}  {direction} {abs(s['pct_change']):.0f}%"
                f"  |  slope: {s['slope']:+.0f} requests/yr"
            )
    else:
        lines.append("  None")

    lines.append("")
    lines.append("FALLING:")
    lines.append("-" * 40)
    if falling:
        for cat, s in sorted(falling, key=lambda x: x[1]["pct_change"]):
            lines.append(
                f"  {cat:<14}  {s['pct_change']:.0f}%"
                f"  |  {s['baseline_mean']:.0f} -> {s['recent_mean']:.0f} requests/yr"
            )
    else:
        lines.append("  None")

    lines.append("")
    lines.append("TOP DRIFTING LOCATIONS (recent months):")
    lines.append("-" * 40)
    if len(drifting_locs) > 0:
        for _, row in drifting_locs.iterrows():
            lines.append(
                f"  {row['micro_location']}   slope: {row['slope']:+.2f} complaints/month"
                f"  |  recent total: {int(row['recent_total']):,}"
            )
    else:
        lines.append("  None")

    lines.append("")
    lines.append("TOP HOTSPOTS (last 6 months):")
    lines.append("-" * 40)
    if len(hotspots) > 0:
        for _, row in hotspots.iterrows():
            lines.append(
                f"  {row['micro_location']}   {int(row['total_count']):,} complaints"
                f"  |  last month growth: {int(row['last_month_growth']):+,}"
            )
    else:
        lines.append("  No hotspot data available")

    lines.append("")
    lines.append("WARD 7 VS CITY (proportion ratio):")
    lines.append("-" * 40)
    if len(city_ratio) > 0:
        for _, row in city_ratio.head(5).iterrows():
            if pd.notna(row["ward_to_city_ratio"]):
                lines.append(
                    f"  {row['signal']:<14}  {row['ward_to_city_ratio']:.2f}x city proportion"
                )
    else:
        lines.append("  No city comparison data")

    lines.append("")
    lines.append("REPEATED COMPLAINTS (last 12 months, 3+ occurrences):")
    lines.append("-" * 40)
    if len(repeated) > 0:
        for _, row in repeated.head(5).iterrows():
            lines.append(
                f"  {row['micro_location']}   {row[TYPE_COL]}   count: {int(row['count']):,}"
            )
    else:
        lines.append("  None above threshold")

    lines.append("")
    lines.append("2026 EARLY WARNING (vs same months in prior years):")
    lines.append("-" * 40)
    if len(warn_df) > 0:
        for _, row in warn_df.iterrows():
            if row["pct_vs_hist"] == "N/A":
                continue
            pct  = float(row["pct_vs_hist"])
            flag = "  [FLAG]" if abs(pct) > 15 else ""
            sign = "+" if pct >= 0 else ""
            lines.append(
                f"  {row['Category']:<14}"
                f"  {sign}{pct:.1f}% vs hist avg"
                f"  |  2026: {int(row['2026_count']):,}"
                f"  |  hist avg: {row['historical_avg']:,}"
                f"{flag}"
            )
    else:
        lines.append("  No 2026 data available")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    DATA_DIR_OUT.mkdir(exist_ok=True)

    print("\n--- Loading data ---")
    df = load_data()

    print("\n--- Applying categories ---")
    df = apply_categories(df)

    print("\n--- Building micro-locations ---")
    df = apply_micro_location(df)

    print("\n--- Splitting Ward 7 ---")
    ward7, city = split_ward7(df)

    # annual trend chart and signal detection
    print("\n--- Computing annual YoY trends ---")
    pivot = compute_yoy_trends(ward7)
    plot_yoy_trends(pivot)

    print("\n--- Computing annual signals ---")
    signals_annual = compute_signals_annual(pivot)

    # monthly trend chart
    print("\n--- Computing monthly trends ---")
    plot_monthly_trends(ward7)

    # top rising - both annual and monthly versions
    print("\n--- Computing top rising categories ---")

    # annual rising from signal detection
    annual_rising_rows = []
    for cat, sig in signals_annual.items():
        annual_rising_rows.append({
            "signal":       cat,
            "recent_avg":   sig["recent_mean"],
            "baseline_avg": sig["baseline_mean"],
            "abs_change":   sig["recent_mean"] - sig["baseline_mean"] if not pd.isna(sig["recent_mean"]) else np.nan,
            "pct_change":   sig["pct_change"],
        })
    annual_rising_df = (
        pd.DataFrame(annual_rising_rows)
        .dropna(subset=["abs_change"])
        .sort_values("abs_change", ascending=False)
        .head(TOP_N_RISING)
    )
    plot_top_rising(annual_rising_df, "top_5_rising_annual.png", "Top 5 Rising Categories (Annual: 2025 vs 2018-2024)")
    annual_rising_df.to_csv(DATA_DIR_OUT / "top_5_rising_annual.csv", index=False)
    print("  Saved: top_5_rising_annual.csv")

    # monthly rising - last 3 months vs prior 12 months
    monthly_rising = compute_top_rising_monthly(ward7)
    plot_top_rising(monthly_rising, "top_5_rising_monthly.png", f"Top 5 Rising Categories (Monthly: last {RECENT_MONTHS} months vs prior {BASELINE_MONTHS})")
    monthly_rising.to_csv(DATA_DIR_OUT / "top_5_rising_monthly.csv", index=False)
    print("  Saved: top_5_rising_monthly.csv")

    # drifting locations
    print("\n--- Computing drifting locations ---")
    drifting_locs = compute_top_drifting_locations(ward7)
    plot_drifting_locations(ward7, drifting_locs)
    drifting_locs.to_csv(DATA_DIR_OUT / "top_3_drifting_locations.csv", index=False)
    print("  Saved: top_3_drifting_locations.csv")

    # categories rising faster than city
    print("\n--- Computing categories faster than city ---")
    faster = compute_categories_faster_than_city(df, ward7)
    plot_categories_faster_than_city(faster)
    faster.to_csv(DATA_DIR_OUT / "categories_faster_than_city.csv", index=False)
    print("  Saved: categories_faster_than_city.csv")

    # repeated complaints
    print("\n--- Computing repeated complaints ---")
    repeated = compute_repeated_complaints(ward7)
    plot_repeated_complaints(repeated)
    repeated.to_csv(DATA_DIR_OUT / "repeated_complaint_locations.csv", index=False)
    print("  Saved: repeated_complaint_locations.csv")

    # city comparison - both versions
    print("\n--- Computing city comparison ---")
    comp_annual = compute_city_comparison_annual(ward7, city)
    plot_city_comparison_annual(comp_annual)

    city_ratio = compute_city_ratio_snapshot(city, ward7)
    plot_city_ratio_snapshot(city_ratio)
    city_ratio.to_csv(DATA_DIR_OUT / "ward7_vs_city_baseline.csv", index=False)
    print("  Saved: ward7_vs_city_baseline.csv")

    # hotspots
    print("\n--- Computing hotspots ---")
    hotspots = compute_hotspots(ward7)
    plot_hotspots(hotspots)
    hotspots.to_csv(DATA_DIR_OUT / "top_5_hotspots.csv", index=False)
    print("  Saved: top_5_hotspots.csv")

    # ward 7 vs its own history
    print("\n--- Computing Ward 7 vs own history ---")
    history = ward7_vs_own_history(ward7)
    plot_ward_vs_history(history)
    history.to_csv(DATA_DIR_OUT / "ward7_vs_its_own_history.csv", index=False)
    print("  Saved: ward7_vs_its_own_history.csv")

    # consecutive month risers
    print("\n--- Computing consecutive month risers ---")
    consecutive = consecutive_month_risers(ward7)
    consecutive.to_csv(DATA_DIR_OUT / "third_consecutive_month_risers.csv", index=False)
    print("  Saved: third_consecutive_month_risers.csv")

    # additional plots
    print("\n--- Generating additional plots ---")
    plot_category_heatmap(ward7)
    print("  Saved: ward7_category_heatmap.png")

    # 2026 early warning
    print("\n--- Computing 2026 early warning ---")
    warn_df = compute_2026_early_warning(ward7)
    if len(warn_df) > 0:
        warn_df.to_csv(DATA_DIR_OUT / "early_warning_2026.csv", index=False)
        print("  Saved: early_warning_2026.csv")

    # stacked repeated complaints plot
    plot_repeated_complaints_stacked(repeated)
    print("  Saved: repeated_complaints_stacked.png")

    # story
    print("\n--- Generating story ---")
    story = generate_story(signals_annual, hotspots, warn_df, drifting_locs, repeated, city_ratio)
    print("\n" + story)

    # encoding="utf-8" needed on windows to handle special characters
    with open(OUTPUT_DIR / "ward7_story.txt", "w", encoding="utf-8") as f:
        f.write(story)
    print("\n  Saved: ward7_story.txt")

    print(f"\n-- All WOW outputs saved to: {OUTPUT_DIR}/ --")


# only runs if called directly, not if someone imports a function from this file
if __name__ == "__main__":
    main()
