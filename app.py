import streamlit as st
import pandas as pd
import pickle

@st.cache_resource()
def load_model():
    file =  open('model.sav', 'rb')
    gsearch = pickle.load(file)
    file.close()
    return gsearch

model = load_model()

inputs = {}
for var in ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']:
    response = st.number_input(var, value=None, placeholder="Type a number...")
    inputs[var] = [response]


df = pd.DataFrame(inputs)
all_data = df.dropna()
if all_data.shape[0] > 0:
    prediction = model.best_estimator_.predict(df)

    st.write('Your probability of deposit is:')
    st.write(prediction)