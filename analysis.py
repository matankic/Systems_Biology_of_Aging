import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from scipy.stats import linregress
import os

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
# Set visual style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Directory and File Configuration
DATA_DIR = "DATA"
OUTPUT_DIR = "Figures"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_PAIRS = [
    ('DEMO_1999_2000.xpt', 'NHANES_1999_2000_MORT_2019_PUBLIC.dat'),
    ('DEMO_2001_2002.xpt', 'NHANES_2001_2002_MORT_2019_PUBLIC.dat'),
    ('DEMO_2003_2004.xpt', 'NHANES_2003_2004_MORT_2019_PUBLIC.dat'),
    ('DEMO_2005_2006.xpt', 'NHANES_2005_2006_MORT_2019_PUBLIC.dat'),
    ('DEMO_2007_2008.xpt', 'NHANES_2007_2008_MORT_2019_PUBLIC.dat'),
    ('DEMO_2009_2010.xpt', 'NHANES_2009_2010_MORT_2019_PUBLIC.dat'),
    ('DEMO_2011_2012.xpt', 'NHANES_2011_2012_MORT_2019_PUBLIC.dat'),
    ('DEMO_2013_2014.xpt', 'NHANES_2013_2014_MORT_2019_PUBLIC.dat'),
    ('DEMO_2015_2016.xpt', 'NHANES_2015_2016_MORT_2019_PUBLIC.dat'),
    ('DEMO_2017_2018.xpt', 'NHANES_2017_2018_MORT_2019_PUBLIC.dat'),
]

# Standardized Colors
COLORS = {
    'Isolated': 'firebrick', 
    'Connected': 'navy'
}

# ==========================================
# 2. DATA LOADING FUNCTIONS
# ==========================================
def parse_mortality_file(filename):
    """
    Parses the fixed-width NHANES mortality file.
    Extracts SEQN, MORTSTAT, and PERMTH_EXM.
    """
    filepath = os.path.join(DATA_DIR, filename)
    data = []
    
    if not os.path.exists(filepath):
        print(f"Warning: Missing mortality file: {filepath}")
        return pd.DataFrame()
        
    with open(filepath, 'r') as f:
        for line in f:
            try:
                # SEQN: Bytes 0-6
                seqn = int(line[0:6])
                # MORTSTAT: Byte 15
                mortstat = int(line[15:16])
                # PERMTH_EXM: Bytes 42-45
                permth_str = line[42:45].strip()
                permth = float(permth_str) if permth_str != '' else np.nan
                data.append([seqn, mortstat, permth])
            except ValueError:
                continue
                
    return pd.DataFrame(data, columns=['SEQN', 'mortstat', 'permth_exm'])

def load_and_merge_data(file_pairs):
    """
    Iterates through file pairs, loads SAS and DAT files, and merges them.
    """
    all_data = []
    print(f"Starting data load from {DATA_DIR}...")

    for demo_file, mort_file in file_pairs:
        demo_path = os.path.join(DATA_DIR, demo_file)
        
        if os.path.exists(demo_path):
            try:
                # Load Demographics
                demo = pd.read_sas(demo_path)
                demo.columns = [c.upper() for c in demo.columns] 
                
                # Select required columns
                cols_to_keep = ['SEQN', 'RIDAGEYR', 'RIAGENDR']
                if 'DMDHHSIZ' in demo.columns:
                    cols_to_keep.append('DMDHHSIZ')
                
                # Check for missing columns (e.g., if variable names changed in specific cycle)
                if not all(col in demo.columns for col in cols_to_keep):
                    print(f"  Skipping {demo_file}: Missing required columns.")
                    continue

                demo = demo[cols_to_keep]
                
                # Load Mortality
                mort = parse_mortality_file(mort_file)
                
                if not mort.empty:
                    # Ensure join key is numeric
                    demo['SEQN'] = pd.to_numeric(demo['SEQN'], errors='coerce')
                    mort['SEQN'] = pd.to_numeric(mort['SEQN'], errors='coerce')
                    
                    merged = pd.merge(demo, mort, on='SEQN')
                    all_data.append(merged)
                    
            except Exception as e:
                print(f"  Error reading {demo_file}: {e}")
        else:
            print(f"  Missing demo file: {demo_path}")

    if not all_data:
        return pd.DataFrame()
        
    return pd.concat(all_data, ignore_index=True)

# ==========================================
# 3. DATA PROCESSING
# ==========================================
raw_df = load_and_merge_data(FILE_PAIRS)

if raw_df.empty:
    raise RuntimeError("No data loaded. Ensure 'DATA' folder contains the correct files.")

# Remove duplicates
df = raw_df.drop_duplicates(subset=['SEQN'], keep='first')

# Type Conversion
numeric_cols = ['mortstat', 'permth_exm', 'RIDAGEYR', 'RIAGENDR', 'DMDHHSIZ']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Filter: Age 50+, remove invalid entries
df = df[df['RIDAGEYR'] >= 50].copy()
df = df.dropna(subset=numeric_cols)

# Define Social Groups (Household Size == 1 vs > 1)
df['Group'] = np.where(df['DMDHHSIZ'] == 1, 'Isolated', 'Connected')

# Decode Gender (1=Male, 2=Female in NHANES)
df['Gender'] = df['RIAGENDR'].map({1: 'Male', 2: 'Female'})

# Calculate Age at Event (for Left Truncation)
df['follow_up_years'] = df['permth_exm'] / 12.0
df['age_at_event'] = df['RIDAGEYR'] + df['follow_up_years']

print("=" * 30)
print(f"Data Processing Complete.")
print(f"Final N: {len(df)}")
print(df.groupby(['Gender', 'Group']).size())
print("=" * 30)

# ==========================================
# 4. SURVIVAL ANALYSIS
# ==========================================

# --- Figure 1: Overall Survival Probability ---
plt.figure(figsize=(10, 6), dpi=140)
kmf = KaplanMeierFitter()

plt.xlim(50, 100)
plt.ylim(0, 1.05)

for group in ['Connected', 'Isolated']:
    mask = df['Group'] == group
    
    kmf.fit(
        durations=df.loc[mask, 'age_at_event'], 
        event_observed=df.loc[mask, 'mortstat'], 
        entry=df.loc[mask, 'RIDAGEYR'],   
        label=group
    )
    kmf.plot_survival_function(color=COLORS[group], linewidth=2.5)

plt.title(f'Overall Survival Probability by Social Isolation (N={len(df)})', fontsize=14, fontweight='bold')
plt.xlabel('Age (Years)', fontsize=12)
plt.ylabel('Survival Probability S(t)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(frameon=True)
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1.png'))


# --- Figure 2: Survival by Gender (Left Truncated) ---
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 6), dpi=120)

for i, gender in enumerate(['Male', 'Female']):
    ax = axes[i]
    df_g = df[df['Gender'] == gender]
    
    for group in ['Connected', 'Isolated']:
        mask = df_g['Group'] == group
        
        kmf.fit(
            durations=df_g.loc[mask, 'age_at_event'], 
            event_observed=df_g.loc[mask, 'mortstat'], 
            entry=df_g.loc[mask, 'RIDAGEYR'], # Handling left truncation
            label=group
        )
        kmf.plot_survival_function(ax=ax, color=COLORS[group], linewidth=2.5)

    ax.set_title(f'{gender} Survival Curve', fontsize=14, fontweight='bold')
    ax.set_xlabel('Age')
    if i == 0: ax.set_ylabel('Survival Probability')
    ax.set_xlim(50, 100)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2.png'))



# --- Figure 3: Gompertz Analysis (Log-Cumulative Hazard) ---
plt.figure(figsize=(14, 6))
naf = NelsonAalenFitter()
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120, sharey=True)

start_age, end_age = 50, 95

for i, gender in enumerate(['Male', 'Female']):
    ax = axes[i]
    df_g = df[df['Gender'] == gender]
    
    for group in ['Connected', 'Isolated']:
        mask = df_g['Group'] == group
        
        naf.fit(
            durations=df_g.loc[mask, 'age_at_event'], 
            event_observed=df_g.loc[mask, 'mortstat'],
            entry=df_g.loc[mask, 'RIDAGEYR'],
            label=group
        )
        
        # Calculate Log Cumulative Hazard
        H_t = naf.cumulative_hazard_
        H_t = H_t.loc[start_age:end_age]
        log_H_t = np.log(H_t + 1e-5)
        
        ax.plot(log_H_t.index, log_H_t.values, color=COLORS[group], label=group, linewidth=2.5)
        
        # Calculate Slope (Alpha) for annotation
        if len(log_H_t) > 10:
            x = log_H_t.index.values
            y = log_H_t.iloc[:, 0].values
            slope, _, _, _, _ = linregress(x, y)
            
            # Annotate plot
            y_pos = 0.9 - (0.1 if group == 'Isolated' else 0)
            ax.text(0.05, y_pos, f"{group} $\\alpha$={slope:.3f}", 
                    transform=ax.transAxes, color=COLORS[group], fontweight='bold')

    ax.set_title(f'{gender} Gompertz Analysis', fontsize=14, fontweight='bold')
    ax.set_xlabel('Age')
    if i == 0: ax.set_ylabel('Log Cumulative Hazard ln(H(t))')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3.png'))



# --- Figure 4: Baseline Age Distribution ---
plt.figure(figsize=(8, 5), dpi=120)
sns.boxplot(
    data=df, x='Gender', y='RIDAGEYR', hue='Group', 
    palette=COLORS
)

plt.title('Distribution of Baseline Ages', fontsize=14, fontweight='bold')
plt.ylabel('Age at Survey Entrance')
plt.xlabel('')
plt.grid(True, alpha=0.3)
plt.legend(title='Status')
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4.png'))


# Print mean ages for verification
print("Mean Age by Group:")
print(df.groupby(['Gender', 'Group'])['RIDAGEYR'].mean())


# --- Figure 5: Survival Advantage (Difference Plot) ---
plt.figure(figsize=(10, 6))

timeline = np.arange(50, 96, 1)
gender_colors = {'Male': 'navy', 'Female': 'crimson'}

for gender in ['Male', 'Female']:
    df_g = df[df['Gender'] == gender]
    
    # 1. Fit Connected
    mask_c = df_g['Group'] == 'Connected'
    kmf_c = KaplanMeierFitter()
    kmf_c.fit(
        durations=df_g.loc[mask_c, 'age_at_event'], 
        event_observed=df_g.loc[mask_c, 'mortstat'], 
        entry=df_g.loc[mask_c, 'RIDAGEYR']
    )
    # Extract survival at specific timeline
    # Note: survival_function_at_times returns a Series with the timeline as index
    survival_c = kmf_c.survival_function_at_times(timeline)
    
    # 2. Fit Isolated
    mask_i = df_g['Group'] == 'Isolated'
    kmf_i = KaplanMeierFitter()
    kmf_i.fit(
        durations=df_g.loc[mask_i, 'age_at_event'], 
        event_observed=df_g.loc[mask_i, 'mortstat'], 
        entry=df_g.loc[mask_i, 'RIDAGEYR']
    )
    survival_i = kmf_i.survival_function_at_times(timeline)
    
    # 3. Calculate Difference
    survival_difference = survival_c - survival_i
    
    # 4. Plot
    plt.plot(timeline, survival_difference, label=f'{gender} Advantage', 
             color=gender_colors[gender], linewidth=3)

plt.title('Survival Advantage of Connection (S_connected - S_isolated)', fontsize=14, fontweight='bold')
plt.xlabel('Age (Years)', fontsize=12)
plt.ylabel('Survival Probability Difference', fontsize=12)
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(-0.05, 0.30)
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5.png'))


print("Analysis Complete.")
