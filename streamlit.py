import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('2019_empirical_electricity_prices.csv')

# Create plot
fig, ax = plt.subplots()
ax.plot(data['Time_Index'], data['Prices'])
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.set_title('Electricity Prices Over Time')

# Display plot
st.pyplot(fig)
