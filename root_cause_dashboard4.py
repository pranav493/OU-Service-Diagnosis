# ✅ root_cause_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import altair as alt
import io



st.set_page_config(page_title="Root Cause Drilldown Dashboard", layout="wide")

# Set Streamlit theme via config.toml (not in Python code)
# To apply the theme, create a file at .streamlit/config.toml in your project directory with:
#
# [theme]
# base="light"
# secondaryBackgroundColor="#87ceeb"
#
# You cannot set Streamlit's theme colors directly from Python code.
# The above config.toml settings will apply the light base and sky blue secondary background.
st.title(" OU: Service Diagnosis")
st.markdown("Analyze Casefill %, detect bad months, weeks & understanding drivers with SHAP")
# ---- 1️⃣ Load Data ----
@st.cache_data
def load_data():
    df = pd.read_excel("National level - sourcing info 2.xlsx", sheet_name="Sheet1")
    df = df[df['Casefill %'] != '-']
    df['Casefill %'] = pd.to_numeric(df['Casefill %'], errors='coerce')
    df = df[df['Casefill %'] <= 200]
    df = df.dropna(subset=['Casefill %'])
    
    df = df.replace('NaN', np.nan)
    df = df.dropna()
    return df

df = load_data()

# ---- 2️⃣ Clean & Feature Engineering ----
# Select final features you gave:
features = [
     'Inventory', 'Inventory T-1', 'Inventory T-2', 'Inventory T-3',
    'Demand', 'Demand T-1', 'Demand T-2', 'Demand T-3',
    'Production', 'Production T-1', 'Production T-2', 'Production T-3',
    'Forecast 4WK', 'AVC', 'AVS', 'DOS', 'Demand Volatality',
    'Production Volatality', 'Inventory Volatality', 
    'Uncovered Demand', 'Bias', 'Inventory Turnover'
]

# ✅ If CV & Fill ratio not there, create:

if 'Production Volatality' not in df.columns:
    df['Production Volatality'] = df[['Production','Production T-1','Production T-2','Production T-3']].std(axis=1) / df[['Production','Production T-1','Production T-2','Production T-3']].mean(axis=1)
if 'Inventory Volatality' not in df.columns:
    df['Inventory Volatality'] = df[['Inventory','Inventory T-1','Inventory T-2','Inventory T-3']].std(axis=1) / df[['Inventory','Inventory T-1','Inventory T-2','Inventory T-3']].mean(axis=1)
if 'Uncovered Demand' not in df.columns:
    df['Uncovered Demand'] = df['Demand'] - (df['Production'] + df['Inventory'])
    df['Uncovered Demand'] = df['Uncovered Demand'].clip(lower=0)
if 'Inventory Turnover' not in df.columns:
    df['Inventory Turnover'] = df['Production'] / df['Inventory']
    df['Inventory Turnover'] = df['Inventory Turnover'].replace([np.inf, -np.inf], np.nan).fillna(0)


# ✅ Coerce object columns & handle inf/NaN
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df[features] = df[features].replace([np.inf, -np.inf], np.nan)
df[features] = df[features].fillna(0)

# ---- 3️⃣ Sidebar Filters ----
st.sidebar.title("🔍 Filter Data")
fiscal_month = st.sidebar.selectbox("Fiscal Month", ["All"] + sorted(df['Fiscal Month'].unique()))
base_product = st.sidebar.selectbox("Base Product", ["All"] + sorted(df['Base Product'].unique()))
gmi_division = st.sidebar.selectbox("GMI Division", ["All"] + sorted(df['GMI Division'].unique()))
gph_family = st.sidebar.selectbox("GPH Family", ["All"] + sorted(df['GPH Family'].unique()))
gph_category = st.sidebar.selectbox("GPH Category", ["All"] + sorted(df['GPH Category'].unique()))
production_source = st.sidebar.selectbox("Production Source", ["All"] + sorted(df['Production Source'].unique()))
# ---- 4️⃣ Apply Filters ----
filtered = df.copy()
if fiscal_month != "All":
    filtered = filtered[filtered['Fiscal Month'] == fiscal_month]
if base_product != "All":
    filtered = filtered[filtered['Base Product'] == base_product]
if gmi_division != "All":
    filtered = filtered[filtered['GMI Division'] == gmi_division]
if gph_family != "All":
    filtered = filtered[filtered['GPH Family'] == gph_family]
if gph_category != "All":
    filtered = filtered[filtered['GPH Category'] == gph_category]
if production_source != "All":
    filtered = filtered[filtered['Production Source'] == production_source]    
   



if len(filtered) < 0:
    st.warning("⚠️ Too few rows for robust RCA — widen your filters.")
    st.stop()

# ---- 5️⃣ Monthly RCA (Correct Formula) ----


# ✅ 1️⃣ Define custom fiscal month order (June → May

# ✅ 2️⃣ ---- Monthly RCA ----


# -------------------------------
# ✅ 1️⃣ MONTHLY RCA - FOCUSED VIEW
# -------------------------------
import altair as alt

# -------------------------------
# ✅ 1️⃣ MONTHLY RCA - Top Zoom
# -------------------------------
monthly = (
    filtered.groupby('Fiscal Month').agg({
        'Delivery': 'sum',
        'Demand': 'sum'
    }).reset_index()
)

month_order = ['JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC', 'JAN', 'FEB', 'MAR', 'APR', 'MAY']
monthly['Fiscal Month'] = pd.Categorical(monthly['Fiscal Month'], categories=month_order, ordered=True)
monthly = monthly.sort_values('Fiscal Month')

monthly['Casefill %'] = 100 * (monthly['Delivery'] / monthly['Demand'])
overall_casefill = 100 * (filtered['Delivery'].sum() / filtered['Demand'].sum())

monthly['Flag'] = np.where(monthly['Casefill %'] < overall_casefill, 'Below Target', 'Above Target')
low_months = monthly[monthly['Flag'] == 'Below Target']['Fiscal Month'].tolist()

# ---- 6️⃣ Improved Monthly Chart ----
y_min = max(min(monthly['Casefill %']) - 5, 0)
y_max = min(max(monthly['Casefill %']) + 5, 105)

bar = alt.Chart(monthly).mark_bar().encode(
    x=alt.X('Fiscal Month:N', sort=month_order, title='Fiscal Month'),
    y=alt.Y('Casefill %:Q', scale=alt.Scale(domain=[y_min, y_max], clamp=True)),
    color=alt.Color('Flag:N', scale=alt.Scale(domain=['Above Target', 'Below Target'], range=['#87ceeb', 'navy'])),
    tooltip=['Fiscal Month', alt.Tooltip('Casefill %:Q', format='.2f')]
)

line = alt.Chart(monthly).mark_line(
    color='black',
    point=alt.OverlayMarkDef(filled=True, size=70)
).encode(
    x=alt.X('Fiscal Month:N', sort=month_order),
    y=alt.Y('Casefill %:Q'),
    tooltip=['Fiscal Month', alt.Tooltip('Casefill %:Q', format='.2f')]
)

chart = (bar + line).properties(
    width=800, height=400,
    title=' Monthly Casefill %'
).configure_axisX(
    labelAngle=-30,
    labelPadding=10,
    labelFontSize=12,
    grid=False  # Remove vertical grid lines
).configure_axisY(
    grid=False  # Remove horizontal grid lines
)

st.altair_chart(chart, use_container_width=True)
st.write(f"Casefill: `{overall_casefill:.2f}%`")
if low_months:
    st.info(f"Below-average months: {', '.join(low_months)}")
else:
    st.success(" All months above or equal to overall.")

# ---- 7️⃣ Per Month RCA ----
if low_months:
    st.write("##  Root Cause for Each Low Month")

    for month in low_months:
        st.write(f"###  {month} - Top Drivers")
        month_data = filtered[filtered['Fiscal Month'] == month]
        if month_data.empty:
            st.info(f"Not enough data for {month}")
            continue

        X_m = month_data[features]
        y_m = pd.to_numeric(month_data['Casefill %'], errors='coerce').fillna(0)

        if len(X_m) > 5:
            model_m = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
            model_m.fit(X_m, y_m)

            explainer_m = shap.Explainer(model_m, X_m)
            shap_vals_m = explainer_m(X_m)
            mean_abs_shap = np.abs(shap_vals_m.values).mean(axis=0)
            shap_pct = 100 * mean_abs_shap / mean_abs_shap.sum()
            shap_df = (
                pd.Series(shap_pct, index=features)
                .sort_values(ascending=False)
                .head(5)
                .sort_values(ascending=True)  # ✅ So bars appear biggest on top in barh
                .reset_index()
            )
            shap_df.columns = ['Feature', 'SHAP Contribution %']


            # ✅ Add tidy horizontal bar plot
            fig, ax = plt.subplots(figsize=(8, 4))  # Slightly smaller
            bars = ax.barh(
                shap_df['Feature'],
                shap_df['SHAP Contribution %'],
                color="#87ceeb"  # light blue
            )
            ax.set_title(f"Top Drivers for {month}", fontsize=12)
            ax.set_xlabel('Contribution (%)', fontsize=10)
            ax.set_ylabel('Feature', fontsize=10)
            ax.xaxis.set_major_formatter(mtick.PercentFormatter())

            # Labels inside bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width / 2, bar.get_y() + bar.get_height() / 2,
                        f'{width:.1f}%', va='center', ha='center',
                        color='white', fontweight='bold', fontsize=6)

            plt.tight_layout()
            st.pyplot(fig)

        else:
            st.info(f"Not enough rows for RCA for {month}")


    
   


# ---- 7️⃣ XGBoost RCA ----
X = filtered[features]
y = pd.to_numeric(filtered['Casefill %'], errors='coerce').fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

r2_train = r2_score(y_train, model.predict(X_train))
r2_test = r2_score(y_test, model.predict(X_test))


# ---- 8️⃣ SHAP ----
# Calculate mean absolute SHAP values
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

# Convert to % contribution
shap_importance = pd.Series(mean_abs_shap, index=features)
shap_importance_pct = 100 * shap_importance / shap_importance.sum()
shap_importance_pct = shap_importance_pct.sort_values(ascending=True)  # ascending for horizontal barh

# Show top 10
top_shap_pct = shap_importance_pct.tail(10)  # last 10 because sorted ascending

st.write("###  Top Drivers (SHAP % Contribution)")


# Better bar plot with dynamic title and light blue color
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(
    list(top_shap_pct.index),
    list(top_shap_pct.values),
    color='#87ceeb'  # light blue
)
# Dynamic title based on filters
title_parts = []
if gmi_division != "All":
    title_parts.append(gmi_division)
if gph_family != "All":
    title_parts.append(gph_family)
if gph_category != "All":
    title_parts.append(gph_category)
if production_source != "All":
    title_parts.append(production_source)
dynamic_title = "Top Drivers"
if title_parts:
    dynamic_title += " for " + " | ".join(title_parts)
ax.set_title(dynamic_title, fontsize=14)
ax.set_xlabel('Contribution (%)')

# Download/copy option

# Set y-axis label and format x-axis as percent before saving
ax.set_ylabel('Feature')
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()  # Call after all labels/text are set

# Add % labels in the middle of the bars
for bar in bars:
    width = bar.get_width()
    ax.text(width / 2, bar.get_y() + bar.get_height() / 2,
            f'{width:.1f}%', va='center', ha='center', color='white', fontweight='bold')

# Save a clean version of the figure (remove Streamlit's extra padding)
buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches='tight', facecolor='white', transparent=False)
buf.seek(0)
st.download_button(
    label="Download Plot as PNG",
    data=buf.getvalue(),
    file_name="top_shap_drivers.png",
    mime="image/png"
)

# Format x-axis as %
ax.xaxis.set_major_formatter(mtick.PercentFormatter())

st.pyplot(fig)


# ✅ 1️⃣ Calculate True Casefill
true_casefill = 100 * (filtered['Delivery'].sum() / filtered['Demand'].sum())
overall_casefill = 100 

casefill_gap = overall_casefill - true_casefill

# ✅ 2️⃣ Get Top SHAP Drivers
top_n = 5
top_features = shap_importance_pct.sort_values(ascending=False).head(top_n)

# ✅ 3️⃣ Build Driver Table: mean, overall mean, delta, effect
driver_rows = []
for feature in top_features.index:
    shap_pct = top_features[feature]
    filtered_mean = filtered[feature].sum()
    overall_mean = df[feature].sum()
    delta = filtered_mean - overall_mean

    # Qualitative read
    if delta < 0:
        sign = "🔻 lower than expected"
    else:
        sign = "🔺 higher than expected"

    # Estimate share of Casefill drop
    feature_gap = (shap_pct / 100) * casefill_gap

    driver_rows.append({
        "Feature": feature,
        "SHAP % Contribution": f"{shap_pct:.1f}%",
        "Effect": sign,
        "Estimated Casefill Loss Due to Feature": round(feature_gap, 2)
    })

driver_df = pd.DataFrame(driver_rows)

# ✅ 4️⃣ Show all nicely
st.subheader(" Service Loss Drivers")
st.write(f" Casefill: `{true_casefill:.2f}%` vs Targeted Casefill `{overall_casefill:.2f}%` ⇒ Gap `{casefill_gap:.2f}%`")

st.write("🔍 **Top Features Impacting This Loss:**")
st.dataframe(driver_df)

# ✅ 5️⃣ Visual: stacked bar estimate
fig, ax = plt.subplots(figsize=(8, 4))
driver_df.set_index("Feature")["Estimated Casefill Loss Due to Feature"].plot(
    kind="barh",
    color="tomato",
    ax=ax
)


# ✅ 6️⃣ Inference
st.success(
    "**Takeaway:** These features explain why your Casefill is lower.\n\n"
    "If you can address the top drivers: boost inventory, improve forecast accuracy, or reduce demand variability, "
    f"you can recover up to **~{casefill_gap:.2f}%** Casefill.\n\n"
)

st.write(f"✅ **XGBoost R²:** Train `{r2_train:.3f}` | Test `{r2_test:.3f}`")