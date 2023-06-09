import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

# set page title
st.title('Texas 2020 Forced Coal and Nuclear')

# add table with text
st.write('### Parameters')
param_table = pd.DataFrame()
param_table['Parameter'] = ['Year', 'Forced coal generation', 'Forced nuclear generation', 'Unit commitment', 'Energy share requirement']
param_table['Value'] = ['2020', '3000 MW', 'Yes - MUST_RUN', 'No', 'No']
st.table(param_table)


######################################################################################################################
# PRICES
######################################################################################################################
# Load data

# Get the directory path of the script
script_directory = os.path.dirname(os.path.abspath(__file__))
# Construct the file path relative to the script directory
file_path_2020prices = os.path.join(script_directory, '2020prices.csv')
file_path_prices = os.path.join(script_directory, 'prices.csv')
file_path_Load_data = os.path.join(script_directory, 'Load_data.csv')

# Read the CSV file
df1 = pd.read_csv(file_path_2020prices)
df2 = pd.read_csv(file_path_prices)
dfload = pd.read_csv(file_path_Load_data)

# Create a new column with the time index for the prices
df2['Time_Index'] = range(min(df1['Time_Index']), max(df1['Time_Index'])+1)

# Add a new column with the date
df1['Date'] = pd.to_datetime(df1['Time_Index'], unit='h', origin=pd.Timestamp('2020-01-01'))
df2['Date'] = pd.to_datetime(df2['Time_Index'], unit='h', origin=pd.Timestamp('2020-01-01'))
dfload['Date'] = pd.to_datetime(dfload['Time_Index'], unit='h', origin=pd.Timestamp('2020-01-01'))

# Create two columns
col1, col2 = st.columns(2)

# Create two date pickers for the start and end dates
with col1:
    start_date1 = st.date_input('Select a start date', df1['Date'].min(), key=1)
    end_date1 = st.date_input('Select an end date', df1['Date'].max(), key=2)
    start_date1 = pd.to_datetime(start_date1)
    end_date1 = pd.to_datetime(end_date1)
    # Filter the data based on the selected dates
    filtered_df1_1 = df1[(df1['Date'] >= start_date1) & (df1['Date'] <= end_date1)]
    filtered_df2_1 = df2[(df2['Date'] >= start_date1) & (df2['Date'] <= end_date1)]
    filtered_dfload_1 = dfload[(dfload['Date'] >= start_date1) & (dfload['Date'] <= end_date1)]

with col2:
    start_date2 = st.date_input('Select a start date', df2['Date'].min(), key=3)
    end_date2 = st.date_input('Select an end date', df2['Date'].max(), key=4)
    start_date2 = pd.to_datetime(start_date2)
    end_date2 = pd.to_datetime(end_date2)
    # Filter the data based on the selected dates
    filtered_df1_2 = df1[(df1['Date'] >= start_date2) & (df1['Date'] <= end_date2)]
    filtered_df2_2 = df2[(df2['Date'] >= start_date2) & (df2['Date'] <= end_date2)]
    filtered_dfload_2 = dfload[(dfload['Date'] >= start_date2) & (dfload['Date'] <= end_date2)]


# Create plot1
fig1, ax1 = plt.subplots()
ax1.plot(filtered_df1_1['Date'], filtered_df1_1['Price'])
ax1.plot(filtered_df2_1['Date'], filtered_df2_1['1'])
ax1.legend(['Empirical Prices', 'GenX'])
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.set_title('Electricity Prices Over Time')
plt.xticks(rotation=45, ha='right')

# Create plotload_1
fig1load, ax1load = plt.subplots()
ax1load.plot(filtered_dfload_1['Date'], filtered_dfload_1['Load_MW_z1'])
ax1load.legend(['Load'])
ax1load.set_xlabel('Date')
ax1load.set_ylabel('MW')
ax1load.set_title('Load Over Time')
plt.xticks(rotation=45, ha='right')

# Create plot2
fig2, ax2 = plt.subplots()
ax2.plot(filtered_df1_2['Date'], filtered_df1_2['Price'])
ax2.plot(filtered_df2_2['Date'], filtered_df2_2['1'])
ax2.legend(['Empirical Prices', 'GenX'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Price')
ax2.set_title('Electricity Prices Over Time')
plt.xticks(rotation=45, ha='right')

# Create plotload_2
fig2load, ax2load = plt.subplots()
ax2load.plot(filtered_dfload_2['Date'], filtered_dfload_2['Load_MW_z1'])
ax2load.legend(['Load'])
ax2load.set_xlabel('Date')
ax2load.set_ylabel('MW')
ax2load.set_title('Load Over Time')
plt.xticks(rotation=45, ha='right')

# Display the plots
col1.pyplot(fig1)
col1.pyplot(fig1load)
col2.pyplot(fig2)
col2.pyplot(fig2load)


######################################################################################################################
# CORRELATION
######################################################################################################################
# Calculate correlation coefficients every 24 steps
step_size = 24
corr_pearson = []
corr_spearman = []
for i in range(0, len(df1), step_size):
    y1 = df1.iloc[i:i+step_size]['Price']
    y2 = df2.iloc[i:i+step_size]['1']
    p_corr, _ = pearsonr(y1, y2)
    s_corr, _ = spearmanr(y1, y2)
    corr_pearson.append(p_corr)
    corr_spearman.append(s_corr)

# Calculate the mean absolute error between y1 and y2
mae = mean_absolute_error(y1, y2)

# Create a dataframe with the correlation coefficients
corr = pd.DataFrame()
corr['Time_Index'] = range(1, len(corr_pearson) + 1)
corr['Spearman'] = corr_spearman
corr['Pearson'] = corr_pearson
start_date = pd.to_datetime('2020-01-01')
corr['Date'] = pd.date_range(start=start_date, periods=len(corr), freq='D')

# Set the index to be the date
corr.set_index('Date', inplace=True)
corr.index = pd.to_datetime(corr.index)
corr.drop('Time_Index', axis=1, inplace=True)
corr.dropna(inplace=True)

# Create a table with the average prices, the average variance of the prices, the median prices
data_table = pd.DataFrame()
data_table['Data'] = ['Empirical', 'GenX']
data_table['Average Price'] = [np.mean(y1), np.mean(y2)]
data_table['Average Variance'] = [np.var(y1), np.var(y2)]
data_table['Median Price'] = [np.median(y1), np.median(y2)]
data_table.set_index('Data', inplace=True)

# Create a table with the average correlation coefficients, the mean absolute error
metrics_table = pd.DataFrame()
metrics_table['Metrics'] = ['Pearson', 'Spearman', 'Mean Absolute Error']
metrics_table['Average'] = [np.mean(corr_pearson), np.mean(corr_spearman), mae]
metrics_table.set_index('Metrics', inplace=True)

# Create and display a frequency distribution of df1['Price'] and df2['1'] on the same plot. Do so by puting the two in a new dataframe and using the plotly express histogram function
df = pd.DataFrame()
df['Empirical'] = df1['Price']
df['GenX'] = df2['1']
# If df['Empirical'] and df['Genx'] values are above max_price, set them to max_price
max_price = 200
df['Empirical'] = df['Empirical'].apply(lambda x: max_price if x > max_price else x)
df['GenX'] = df['GenX'].apply(lambda x: max_price if x > max_price else x)

figfreqdis = px.histogram(df, x=df.columns, marginal="rug", title="Frequency Distribution of Prices")
figfreqdis.update_layout(
    xaxis=dict(title="Price"),
    yaxis=dict(title="Count"),
    dragmode="pan",
    barmode='overlay',
    margin=dict(l=20, r=20, t=30, b=20), height=500, width=800,
    font=dict(color='black'),
)
figfreqdis.update_layout(legend=dict(title='Method', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

# Create plot3
fig3 = px.line(corr, x=corr.index, y=corr.columns, title="Daily Correlation Curves")
fig3.update_layout(
    xaxis=dict(title="Date", range=[start_date, corr.index[-1]], rangemode='tozero'),
    yaxis=dict(title="Correlation"),
    dragmode="pan",
    margin=dict(l=20, r=20, t=30, b=20), height=500, width=800,
    font=dict(color='black'),
)
fig3.update_layout(legend=dict(title='Method', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

st.table(metrics_table)
st.table(data_table)
st.plotly_chart(figfreqdis, use_container_width=True)
st.plotly_chart(fig3, use_container_width=True)


######################################################################################################################
# GENERATION
######################################################################################################################

# read csv
file_path_power = os.path.join(script_directory, 'power.csv')
file_path_IntGenbyFuel2020_hourly = os.path.join(script_directory, 'IntGenbyFuel2020_hourly.csv')

dfgenX = pd.read_csv(file_path_power)
dfgenE = pd.read_csv(file_path_IntGenbyFuel2020_hourly)
dfgenX = dfgenX.iloc[2:] # remove first two rows, as first row is zone and second is annual sum
dfgenX['Time_Index'] = range(1, len(dfgenX) + 1)
dfgenE['Time_Index'] = range(1, len(dfgenE) + 1)
dfgenX['Date'] = pd.to_datetime(dfgenX['Time_Index'], unit='h', origin=pd.Timestamp('2020-01-01'))
dfgenE['Date'] = pd.to_datetime(dfgenE['Time_Index'], unit='h', origin=pd.Timestamp('2020-01-01'))
dfgenX.set_index("Date", inplace=True)
dfgenE.set_index("Date", inplace=True)


# Create plotgenX, create total column
natural_gasX_cols = [col for col in dfgenX.columns if "natural_gas" in col]
coalX_cols = [col for col in dfgenX.columns if "coal" in col]
nuclearX_cols = [col for col in dfgenX.columns if "nuclear" in col]
windX_cols = [col for col in dfgenX.columns if "wind" in col]
biomassX_cols = [col for col in dfgenX.columns if "biomass" in col]
solarX_cols = [col for col in dfgenX.columns if "solar" in col]
hydroX_cols = [col for col in dfgenX.columns if "hydroelectric" in col]
otherX_cols = [col for col in dfgenX.columns if "battery" in col]
natural_gasE_cols = [col for col in dfgenE.columns if "Gas" in col]
coalE_cols = [col for col in dfgenE.columns if "Coal" in col]
nuclearE_cols = [col for col in dfgenE.columns if "Nuclear" in col]
windE_cols = [col for col in dfgenE.columns if "Wind" in col]
biomassE_cols = [col for col in dfgenE.columns if "Biomass" in col]
solarE_cols = [col for col in dfgenE.columns if "Solar" in col]
hydroE_cols = [col for col in dfgenE.columns if "Hydro" in col]
otherE_cols = [col for col in dfgenE.columns if "Other" in col]
totalX_cols = natural_gasX_cols + coalX_cols + nuclearX_cols + windX_cols + biomassX_cols + solarX_cols + hydroX_cols + otherX_cols
totalE_cols = natural_gasE_cols + coalE_cols + nuclearE_cols + windE_cols + biomassE_cols + solarE_cols + hydroE_cols + otherE_cols

dfgen_summed = pd.DataFrame({
    "nuclearX": dfgenX[nuclearX_cols].sum(axis=1),
    "nuclearE": dfgenE[nuclearE_cols].sum(axis=1),
    "natural_gasX": dfgenX[natural_gasX_cols].sum(axis=1),
    "natural_gasE": dfgenE[natural_gasE_cols].sum(axis=1),
    "coalX": dfgenX[coalX_cols].sum(axis=1),
    "coalE": dfgenE[coalE_cols].sum(axis=1),
    "solarX": dfgenX[solarX_cols].sum(axis=1),
    "solarE": dfgenE[solarE_cols].sum(axis=1),
    "windX": dfgenX[windX_cols].sum(axis=1),
    "windE": dfgenE[windE_cols].sum(axis=1),
    "biomassX": dfgenX[biomassX_cols].sum(axis=1),
    "biomassE": dfgenE[biomassE_cols].sum(axis=1),
    "hydroX": dfgenX[hydroX_cols].sum(axis=1),
    "hydroE": dfgenE[hydroE_cols].sum(axis=1),
    "otherX": dfgenX[otherX_cols].sum(axis=1),
    "otherE": dfgenE[otherE_cols].sum(axis=1),
    "totalX": dfgenX[totalX_cols].sum(axis=1),
    "totalE": dfgenE[totalE_cols].sum(axis=1)
})

fig4 = go.Figure()

# Add traces for the first y-axis (left side)
for column in dfgen_summed.columns:
    fig4.add_trace(go.Scatter(x=dfgen_summed.index, y=dfgen_summed[column], name=column))
    for trace in fig4.data:
        trace.update(visible='legendonly')

# Add trace for the second y-axis (right side)
for column in corr.columns:
    fig4.add_trace(go.Scatter(x=corr.index, y=corr[column], name=column, visible='legendonly', yaxis='y2'))

fig4.update_layout(
    title="Power Generation by Resource",
    xaxis=dict(title="Date"),
    yaxis=dict(title="Generation (MW)"),
    yaxis2=dict(title="Correlation", overlaying="y", side="right", rangemode='tozero'),
    dragmode="pan",
    margin=dict(l=20, r=20, t=30, b=20),
    height=500,
    width=800
)

st.plotly_chart(fig4, use_container_width=True)

######################################################################################################################
# Calculate the difference, average diff, standard dev, between each generation type
dfgen_diff_avg_std = pd.DataFrame()
dfgen_diff_avg_std['nuclear_diff'] = dfgen_summed['nuclearE'] - dfgen_summed['nuclearX']
dfgen_diff_avg_std['nuclear_avg'] = dfgen_diff_avg_std['nuclear_diff'].mean()
dfgen_diff_avg_std['nuclear_std'] = dfgen_diff_avg_std['nuclear_diff'].std()
dfgen_diff_avg_std['natural_gas_diff'] = dfgen_summed['natural_gasE'] - dfgen_summed['natural_gasX']
dfgen_diff_avg_std['natural_gas_avg'] = dfgen_diff_avg_std['natural_gas_diff'].mean()
dfgen_diff_avg_std['natural_gas_std'] = dfgen_diff_avg_std['natural_gas_diff'].std()
dfgen_diff_avg_std['coal_diff'] = dfgen_summed['coalE'] - dfgen_summed['coalX']
dfgen_diff_avg_std['coal_avg'] = dfgen_diff_avg_std['coal_diff'].mean()
dfgen_diff_avg_std['coal_std'] = dfgen_diff_avg_std['coal_diff'].std()
dfgen_diff_avg_std['solar_diff'] = dfgen_summed['solarE'] - dfgen_summed['solarX']
dfgen_diff_avg_std['solar_avg'] = dfgen_diff_avg_std['solar_diff'].mean()
dfgen_diff_avg_std['solar_std'] = dfgen_diff_avg_std['solar_diff'].std()
dfgen_diff_avg_std['wind_diff'] = dfgen_summed['windE'] - dfgen_summed['windX']
dfgen_diff_avg_std['wind_avg'] = dfgen_diff_avg_std['wind_diff'].mean()
dfgen_diff_avg_std['wind_std'] = dfgen_diff_avg_std['wind_diff'].std()
dfgen_diff_avg_std['biomass_diff'] = dfgen_summed['biomassE'] - dfgen_summed['biomassX']
dfgen_diff_avg_std['biomass_avg'] = dfgen_diff_avg_std['biomass_diff'].mean()
dfgen_diff_avg_std['biomass_std'] = dfgen_diff_avg_std['biomass_diff'].std()
dfgen_diff_avg_std['hydro_diff'] = dfgen_summed['hydroE'] - dfgen_summed['hydroX']
dfgen_diff_avg_std['hydro_avg'] = dfgen_diff_avg_std['hydro_diff'].mean()
dfgen_diff_avg_std['hydro_std'] = dfgen_diff_avg_std['hydro_diff'].std()
dfgen_diff_avg_std['other_diff'] = dfgen_summed['otherE'] - dfgen_summed['otherX']
dfgen_diff_avg_std['other_avg'] = dfgen_diff_avg_std['other_diff'].mean()
dfgen_diff_avg_std['other_std'] = dfgen_diff_avg_std['other_diff'].std()
dfgen_diff_avg_std['total_diff'] = dfgen_summed['totalE'] - dfgen_summed['totalX']
dfgen_diff_avg_std['total_avg'] = dfgen_diff_avg_std['total_diff'].mean()
dfgen_diff_avg_std['total_std'] = dfgen_diff_avg_std['total_diff'].std()

fig5 = go.Figure()

# Add traces for the first y-axis (left side)
for column in dfgen_diff_avg_std.columns:
    fig5.add_trace(go.Scatter(x=dfgen_diff_avg_std.index, y=dfgen_diff_avg_std[column], name=column))
    for trace in fig5.data:
        trace.update(visible='legendonly')

# Add trace for the second y-axis (right side)
for column in corr.columns:
    fig5.add_trace(go.Scatter(x=corr.index, y=corr[column], name=column, visible='legendonly', yaxis='y2'))

fig5.update_layout(
    title="Power Generation Difference by Resource",
    xaxis=dict(title="Date"),
    yaxis=dict(title="Generation Difference (MW)"),
    yaxis2=dict(title="Correlation", overlaying="y", side="right", rangemode='tozero'),
    dragmode="pan",
    margin=dict(l=20, r=20, t=30, b=20),
    height=500,
    width=800
)

st.plotly_chart(fig5, use_container_width=True)

######################################################################################################################
# Calculate the average generation for each resource
avg_generation_table = pd.DataFrame()

# Iterate over the columns of dfgen_summed
for column in dfgen_summed.columns:
    # Split the column name by the last character
    fuel_type, generation_type = column[:-1], column[-1:]

    # Check if the generation type is 'E' or 'X' and add a row accordingly
    if generation_type == 'E':
        avg_generation_table.loc['E', fuel_type] = dfgen_summed[column].mean()
    elif generation_type == 'X':
        avg_generation_table.loc['X', fuel_type] = dfgen_summed[column].mean()

# # Add a column for the total generation
# avg_generation_table['Total'] = avg_generation_table.sum(axis=1)

# Calculate the delta in percentage between 'X' and 'E' generations
x_generation = avg_generation_table.loc['X']
e_generation = avg_generation_table.loc['E']
delta = (x_generation - e_generation) / e_generation * 100
delta[x_generation.abs() < 1e-6] = 0  # Set delta to 0 where 'X' generation is close to zero
avg_generation_table.loc['Delta (%)'] = delta

# Format the values in the table to have no decimal places
avg_generation_table = avg_generation_table.round(0).astype(int).astype(str)

# Display the table in Streamlit
st.table(avg_generation_table)
