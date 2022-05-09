
# coding: utf-8


#Import Libraries

import streamlit as st
import pandas as pd
import numpy as np
import chart_studio.plotly as plotly
import plotly.figure_factory as ff
from plotly import graph_objs as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly

#coding For Sidebar
st.sidebar.subheader("About the Predictor")

st.sidebar.subheader("What is 'ds' and 'y' columns?")
st.sidebar.info("Import the time series csv file. It should have two columns labelled as 'ds' and 'y'.The 'ds' column should be of datetime format  by Pandas. The 'y' column must be numeric representing the measurement to be forecasted.")


st.sidebar.subheader("Created by")
st.sidebar.info("Name")

st.sidebar.subheader("Mentor:")
st.sidebar.info("Mentor Name")


#main Coding For Predictor
st.title('Gold Price Forecast')


DATA_URL =('Gold_data.csv')

month = st.slider('Months of prediction:',1,12)
period = month * 30


@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    return data
	


data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text('Loading data... done!')

def plot_fig():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data.date, y=data['price'], name="Price_History",line_color='blueviolet'))
	#fig.add_trace(go.Scatter(x=data.date, y=data['price'], name="Price_History",line_color='dimgray'))
	fig.layout.update(title_text='Time Series data with Rangeslider',xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	return fig

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
	
# plotting the figure of Actual Data
plot_fig()

# preparing the data for Facebook-Prophet.

data_pred = data[['date','price']]
data_pred=data_pred.rename(columns={"date": "ds", "price": "y"})

# code for facebook prophet prediction

m = Prophet()
m.fit(data_pred)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

#plot forecast
fig1 = plot_plotly(m, forecast)
if st.checkbox('Show forecast data'):
    st.subheader('forecast data')
    st.write(forecast)
st.write('Forecasting gold value for Gold_data for a period of: '+str(month)+'Month/Months')
st.plotly_chart(fig1)

#plot component wise forecast
st.write("Component wise forecast")
fig2 = m.plot_components(forecast)
st.write(fig2)
	
    

