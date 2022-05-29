# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:02:01 2022

@author: marlo
"""
import streamlit as st
import pickle
import pandas as pd
import numpy as np
np.random.seed(42)

st.title("Análise de Clientes com Cartão Crédito para Aumento de Limite - Banking** ")
st.subheader("Arquivo XLSX - Abaixo os Códigos dos Clientes com ou sem predição de default")
st.write('\n\n')

data = st.file_uploader('Escolha o dataset (.xlsx)', type = 'xlsx')
option = st.selectbox('Selecione qual filtro deseja:',('Clientes Pronpenso Default','Clientes sem propensão de Default'))
st.write('You selected:', option)
quant = st.number_input( 'Quantidade Máxima de Clientes à Analisar')
st.write('The current number is ', quant)
if st.button('Analisar'):
    df2 = pd.read_excel(data)
    todrop = ['Codigo']
    df3 = df2.drop(todrop, axis=1)
    a = len(df2)
    b = int(quant)
    with open('full_pipeline3.pkl','rb') as f:
        full_pipeline3 =  pickle.load(f)
        default_X_prep3 = full_pipeline3.fit_transform(df3)
    with open('modelpronto.pkl','rb') as f:
        modelpronto1 =  pickle.load(f)
        y_pred = modelpronto1.predict(default_X_prep3)
        for i in range(1,b):
            if option == 'Clientes sem propensão de Default':
                if y_pred[i]==0:
                    df15 = ([df2.Codigo[i]])
                    st.success(df15)
            else:
                if y_pred[i]==1:
                   df15 = ([df2.Codigo[i]])
                   st.success(df15)
else:
    st.write("Press the above button..")
 
     