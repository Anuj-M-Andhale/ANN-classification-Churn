import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import streamlit as st


class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__() 
        self.model = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),

            
            nn.Linear(64, 1)           
                                    
            
        )

    def forward(self, x):
        return self.model(x)
        
    
model = ANN()
## Load Model
model.load_state_dict(torch.load("ann_model.pth"))

## Load pickle files
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)
with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.title("Customer Churn Prediction")
# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox(' Is Active Member', [0, 1])


# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform(
    [[geography]]
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Combine numerical + encoded categorical features
input_data = pd.concat(
    [
        input_data.reset_index(drop=True),
        geo_encoded_df.reset_index(drop=True)
    ],
    axis=1
)

# Scale
input_df_scaled = scaler.transform(input_data)

# Convert to PyTorch tensor
input_tensor = torch.tensor(
    input_df_scaled,
    dtype=torch.float32
)

# Predict
model.eval()

with torch.no_grad():
    prediction = model(input_tensor)
    pred_probs = torch.sigmoid(prediction).item()
st.write(f'churn probability : {pred_probs:.2f}')
if pred_probs >= 0.5:
    st.write("Customer is Likely to churn")
else:
    st.write("Customer is not likely to churn")