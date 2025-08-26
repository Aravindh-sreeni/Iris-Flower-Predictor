import pandas as pd
import numpy as np
import streamlit as st
from os import path
import pickle


st.title("Flower species predictor")
petal_length=st.number_input("petal length",placeholder="please enter a valid petal length value between 1 and7",min_value=1.0,max_value=6.9,value=None)
petal_width=st.number_input("petal width",placeholder="please enter a valid petal width value between 0 and 2.5",min_value=0.1,max_value=2.5,value=None)
sepal_length=st.number_input("sepal length",placeholder="please enter a valid sepal length value between 4.3 and 8",min_value=4.3,max_value=7.9,value=None)
sepal_width=st.number_input("sepal width",placeholder="please enter a valid sepal width value between 2 and 4.5",min_value=2.0,max_value=4.4,value=None)


#prepare the dataframe for prediction
user_input = pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],
                          columns=['sepal_length','sepal_width','petal_length','petal_width'])
#using the .pkl file, creating an ML model named "iris predictor"
model_path = path.join("Model","iris_classifier.pkl")
with open (model_path,'rb') as file:
    iris_predictor=pickle.load(file)

dict_species={0:'setosa',1:'versicolor',2:'verginica'}

if st.button("predict species"):
    if((petal_length==None)or(petal_width==None)
        or(sepal_length==None)or(sepal_width==None)):
        #Null be exexcuted when any of the values is not entered properly
        st.write("please enter a valid value")
    else:
        #preiction can be done here.we are expecting a dataframe
        predicted_species = iris_predictor.predict(user_input)
        #predictied_species[0] will give us the value in the dataframe
        #we use that value to find the coresponding species from the dictionary 'dict_species'
        st.write("the species is ",dict_species[predicted_species[0]])

st.write(user_input)