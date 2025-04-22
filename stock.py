import streamlit as st    
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


st.title('Stock price prediction app')
st.sidebar.header("user selection")
stock_ticker=st.sidebar.text_input("Enter stock Ticker (e.g., RELIANCE.BO):","RELIANCE.BO")
start_date=st.sidebar.date_input("start Date",pd.to_datetime("2020-01-01"))
end_date=st.sidebar.date_input("end Date",pd.to_datetime("2025-04-22"))
stock_data=yf.download(stock_ticker,start=start_date,end=end_date)
a=st.sidebar.button("VIEW DATA")
if a:
    st.write("stock Data")
    st.write(stock_data)
stock_data.reset_index(inplace=True)

stock_data['Days']=(stock_data.index-stock_data.index.min())
stock_data=stock_data[['Days','Date','Close']]
x=stock_data.drop(['Close','Date'],axis=1)
y=stock_data['Close']
model=LinearRegression()
model.fit(x,y)
#st.write(x)
#st.write(y)
st.sidebar.subheader("feature Prediction Input")
select_date=st.sidebar.date_input("select date",pd.to_datetime("2025-01-01"))
days=(pd.to_datetime(select_date)-stock_data['Date'].min()).days
#st.write(days)
y_pred=model.predict([[days]])
#st.write(y_pred)

    
c=st.sidebar.button("View PREDICTION")
if c:
    st.subheader(f'PREDICTED PRICE For {select_date} is :       {y_pred[0]}')




fig, ax = plt.subplots()
ax.plot(stock_data['Date'],stock_data["Close"])
plt.figure(figsize=(6,4))
b=st.sidebar.button("VIEW GRAPH")
if b:
    st.pyplot(fig)


#plt.bar(stock_data["Date"],stock_data["Close"],label='close')

hide_menu = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)
