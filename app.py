import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import plotly.express as px
import datetime


st.set_page_config(
   page_title="Stock Trend Prediction",
   page_icon="📈",
   layout="wide",
   initial_sidebar_state="expanded",
)

st.title('Stock Dashboard')

ticker = st.sidebar.text_input('Ticker', 'SBIN.NS')
start_date = st.sidebar.date_input('Start Date',datetime.date(2000, 1, 1), max_value = datetime.date(2020, 1, 1))
end_date = st.sidebar.date_input('End Date',min_value = datetime.date(2008, 1, 1))
n = st.sidebar.number_input('Enter number of days for future prediction',value=30)

data = yf.download(ticker, start=start_date, end=end_date)

import plotly.graph_objects as go
figure = go.Figure(data=[go.Candlestick(x=data.index,
                                        open=data["Open"], 
                                        high=data["High"],
                                        low=data["Low"], 
                                        close=data["Close"])])
figure.update_layout(title = ticker, 
                     xaxis_rangeslider_visible=False)
st.plotly_chart(figure)


data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.80)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.80):int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

model = load_model('keras_model1.h5')

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

y_predicted = scaler.inverse_transform(y_predicted)
y_test=y_test.reshape(-1,1)
y_test = scaler.inverse_transform(y_test)



fundamental_data, news, prediction = st.tabs(["Fundamental Data", "Top News", "Prediction"])
with fundamental_data:
    st.write('Dataset')
    st.write(data)

    
    ma100 = data.Close.rolling(100).mean()
    newnames = {'Close':'Close', 'wide_variable_1': 'ma100'}
    
    fig2 = px.line(data, x=data.index, y=[data['Close'],ma100], title='Closing Price vs Time Chart with 100MA',color_discrete_sequence=[ '#2E91E5', '#B82E2E'])
    fig2.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     )
                  )
    
    st.plotly_chart(fig2)
    

    ma200 = data.Close.rolling(200).mean()
    newnames = {'Close':'Close', 'wide_variable_1': 'ma100','wide_variable_2': 'ma200'}
    
    fig3 = px.line(data, x=data.index, y=[data['Close'],ma100,ma200], title='Closing Price vs Time Chart with 200MA',color_discrete_sequence=[ '#2E91E5', '#B82E2E','#16FF32'])
    fig3.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     )
                  )
    
    st.plotly_chart(fig3)


with news:
    company = yf.Ticker(ticker)
    news_df = company.news
    df = pd.DataFrame(news_df)
    st.write(f"Latest news articles for {ticker}:")
    st.write(df.loc[:, ["title", "link", "publisher"]].to_html(render_links=True), unsafe_allow_html=True)
    
with prediction:
    

    
    
    dataframe = pd.DataFrame(data['Close'][int(len(data)*0.80):int(len(data))])

    y_test_df = pd.DataFrame(y_test, columns = ['ytest'])
    y_predicted_df = pd.DataFrame(y_predicted, columns = ['ypredicted'])

    newnames = {'wide_variable_0': 'Original Price','wide_variable_1': 'Predicted Price'}
    
    fig4 = px.line(dataframe, x=dataframe.index, y=[y_test_df['ytest'],y_predicted_df['ypredicted']], title='Predictions vs Original',color_discrete_sequence=[ '#2E91E5', '#B82E2E'])

    fig4.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     )
                  )
    
    st.plotly_chart(fig4)

    dataframe.insert(loc=1,column='Predicted Values',value=y_predicted)
    dataframe.rename(columns = {'Close':'Actual Values'}, inplace = True)
    st.write(dataframe)

    i=0
    temp=[]
    temp=np.array(temp)
    x_future=data_testing.tail(100)
    x_future=scaler.fit_transform(x_future)
    x_test_future=[]
    x_test_future.append(x_future[0:100])
    x_test_future=np.array(x_test_future)
    while(i<n):
        y_predicted_future = model.predict(x_test_future)
        temp=np.append(temp,y_predicted_future)
        x_test_future=np.append(x_test_future,y_predicted_future)
        x_test_future=x_test_future[1:]
        x_test_future=x_test_future.reshape(1,100,1)
        i=i+1
    temp=temp.reshape(-1,1)
    temp = scaler.inverse_transform(temp)    
    
    
    temp_df = pd.DataFrame(temp, columns = ['0'])
    newnames = {'0': 'Future Value'}
    fig5 = px.line(temp_df, x=temp_df.index, y=[temp_df['0']], title='Next N days future prediction',labels={"index": "Days"})
    fig5.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     )
                  )
    
    st.plotly_chart(fig5)



