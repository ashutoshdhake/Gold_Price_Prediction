import streamlit as st
import datetime

import pandas as pd
import warnings
from pickle import load

warnings.filterwarnings('ignore')


df = pd.read_csv('Gold_data.csv')


# load the model from disk
loaded_model = load(open('gold_prediction_ARIMA_model.sav', 'rb'))


def main():
    st.title('Model Deployment: Timeseries Forecasting')
    
    s = datetime.date(2021,12,22)
    e = st.date_input("Enter the ending Date to Predict the Gold Prices")
    diff=( (e-s).days+1)
    
    
    if st.button("PREDICT"):
        index_future_dates=pd.date_range(start= s ,end= e)
        final_forecast=loaded_model.forecast(steps=diff)[0]
        pred =  pd.DataFrame(final_forecast)
        pred.index=index_future_dates.rename("date")
        pred.columns=['Price']
        
        st.dataframe(pred)
        
        st.line_chart(pred)
        
        
       
      
        
if __name__ == '__main__':
    main() 