# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 13:15:29 2021

@author: Akhil
"""
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import datetime as dt
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
from matplotlib.pyplot import plot

warnings.filterwarnings("ignore")

def main():       
       #front end elements of the web page 
       html_temp = """ 
       <div style ="background-color:#1EDFFD;padding:3px"> 
       <h2 style ="color:black;text-align:center;">
       Crypto Portfolio Optimizer</h2> 
       </div> 
       """
       st.header(("Predicting the optimal portfolio using Modern Portfolio Theory"))
       st.markdown("There’s no such thing as the perfect investment, but crafting a strategy that offers high returns and relatively low risk is priority for modern investors. While this hallmark seems rather straightforward today, this strategy actually didn’t exist until the latter half of the 20th century. In 1952, an economist named Harry Markowitz wrote his dissertation on “Portfolio Selection”, a paper that contained theories which transformed the landscape of portfolio management—a paper which would earn him the Nobel Prize in Economics nearly four decades later. As the philosophical antithesis of traditional stock selection, his Modern Portfolio Theory (MPT) continues to be a popular investment strategy, and this portfolio management tool—if used correctly—can result in a diverse, profitable investment portfolio") 
       st.markdown(" Instead of focusing on the risk of each individual asset, Markowitz demonstrated that a diversified portfolio is less volatile than the total sum of its individual parts. While each asset itself might be quite volatile, the volatility of the entire portfolio can actually be quite low. More than 60 years after its introduction, the fundamentals of MPT ring true. Let’s delve into this popular portfolio management strategy, and discover what makes the principles of this revolutionary theory so effective")
      #display the front end aspect
       st.markdown(html_temp, unsafe_allow_html = True) 

      #following lines create boxes in which user can enter  data  required to make prediction
       st.write('Select the purchased coins from down the list:')
       
       list_coins = []
       option_1 = st.checkbox('Bitcoin (BTC)')
       option_2 = st.checkbox('Etherium (ETH)')
       option_3 = st.checkbox('Cardano (CAR)')
       option_4 = st.checkbox('Doge (DOGE)')
       option_5 = st.checkbox('LiteCoin (LTC)')
       option_6 = st.checkbox('Solana (SOL)')
       option_7 = st.checkbox('Polka Dot (DOT)')
       
       st.markdown('---------------------------------------------')
       df = pd.read_pickle('Coins.pkl')
       if option_1:
           list_coins.append("BTC-USD")
           st.markdown("**Stock price changes in Bitcoin**")
           temp = df.iloc[:,0]
           st.area_chart(temp)    
       if option_2:
           list_coins.append("ETH-USD")
           st.markdown("**Stock price changes in Ethereum**")
           temp = df.iloc[:,1]
           st.area_chart(temp)
       if option_3:
           list_coins.append("ADA-USD")
           st.markdown("**Stock price changes in Cardano**")
           temp = df.iloc[:,2]
           st.area_chart(temp)
       if option_4:
           list_coins.append("DOGE-USD")
           st.markdown("**Stock price changes in Doge**")
           temp = df.iloc[:,3]
           st.area_chart(temp)
       if option_5:
           list_coins.append("LTC-USD")
           st.markdown("**Stock price changes in LiteCoin**")
           temp = df.iloc[:,4]
           st.area_chart(temp)
       if option_6:
           list_coins.append("SOL1-USD")
           st.markdown("**Stock price changes in Solana**")
           temp = df.iloc[:,5]
           st.area_chart(temp)
       if option_7:
           list_coins.append("DOT1-USD")
           st.markdown("**Stock price changes in Polka Dot**")
           temp = df.iloc[:,6]
           st.area_chart(temp)
  
       
       st.markdown('---------------------------------------------')
       list_amount = [ ]
       amount_input  = st.text_input("Enter the amount invested in the coins in the same order seperated by a comma")
       L = amount_input.split(', ')
       L = L[0].split(',')
       for i in range(len(L)):
           list_amount.append(L[i])
           
       st.markdown('---------------------------------------------')
           
       df = df[list_coins]
       cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
       corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()
       ind_er = df.resample('Y').last().pct_change().mean()
       ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))
       assets = pd.concat([ind_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
       assets.columns = ['Returns', 'Volatility']
       
       if len(list_coins)>0:
           st.markdown('*The table for the expected returns and the volatility of the chosen coins based on the historical data*')
           st.dataframe(assets)
           
       p_ret = [] # Define an empty array for portfolio returns
       p_vol = [] # Define an empty array for portfolio volatility
       p_weights = [] # Define an empty array for asset weights

       num_assets = len(df.columns)
       num_portfolios = 10000

       for portfolio in range(num_portfolios):
           weights = np.random.random(num_assets)
           weights = weights/np.sum(weights)
           p_weights.append(weights)
           returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its weights 
           p_ret.append(returns)
           var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
           sd = np.sqrt(var) # Daily standard deviation
           ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
           p_vol.append(ann_sd)
           
       
       data = {'Returns':p_ret, 'Volatility':p_vol}
       for counter, symbol in enumerate(df.columns.tolist()):
           #print(counter, symbol)
           data[symbol+' weight'] = [w[counter] for w in p_weights]
       portfolios  = pd.DataFrame(data)
       if len(list_coins)>0:
           st.subheader('Safest Portfolio Proportion')
           min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
           st.dataframe(min_vol_port)
           
           st.subheader("Most Risky Portfolio")
           rf = 0.01 # risk factor
           optimal_risky_port = portfolios.iloc[((portfolios['Returns']-rf)/portfolios['Volatility']).idxmax()]
           st.dataframe(optimal_risky_port)
           
           st.markdown("----------------------------------------")
           st.subheader("The Efficient Forntier Curve Looks like this")
           x = portfolios['Volatility'].tolist()
           y = portfolios['Returns'].tolist()
           fig = plt.figure()
           plt.scatter(x, y, c=y, cmap="RdYlGn", s=10)
           plt.xlabel("Volatility")
           plt.ylabel("Returns")
           st.pyplot(fig)
           
       
           
       result =""
      
if __name__=='__main__': 
        main()
    
