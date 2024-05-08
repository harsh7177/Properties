import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import KNNImputer

def impute(df):
    imputer = KNNImputer(n_neighbors=5)
    num_df = df.drop(['status','area'],axis=1)
    imp_df = pd.DataFrame(imputer.fit_transform(num_df), columns=['BHK','Size(Sq/ft)','bath','Price','Price/Sqft'])
    status_dummi=pd.get_dummies(df['status'])
    final_df = pd.concat([imp_df, status_dummi], axis=1)
    final_df=final_df.dropna()
    return final_df

def predict_house_price(df,input_values):
    models = {
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor()
    }
   
    
    f_df=impute(df)
    f_df = f_df[['Price', 'BHK', 'Size(Sq/ft)', 'bath', 'Ready to move', 'Under Construction']]
    input_dict = {
        'BHK': input_values[0],
        'Size(Sq/ft)': input_values[1],
        'bath': input_values[2],
        'Ready to move': input_values[3],
        'Under Construction': input_values[4]
    }
    X_user = pd.DataFrame([input_dict])
    predictions = {'Random Forest': [], 'Gradient Boosting': []}
    errors = {'Random Forest': [], 'SVM': []}
    for name, model in models.items():
        X = f_df.drop(columns=['Price'])
        y = f_df['Price']
        model.fit(X, y)
        y_pred = model.predict(X_user)
        predictions[name] = y_pred[0]
        y_pred_all = model.predict(X)
        mse = mean_squared_error(y, y_pred_all)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred_all)
        errors[name] = {'Mean Squared Error': mse, 'Root Mean Squared Error': rmse, 'R-squared': r2}
    average_predictions = np.mean(list(predictions.values()))
    return average_predictions,errors

def predict_page(suburb):
    st.write(suburb)
    df=pd.read_csv('noida_properties.csv',index_col=0)
    
    df=df[df['area']==suburb]
    
    bhk = st.number_input("BHK (Bedrooms, Hall, Kitchen)", min_value=1, step=1)
    size_sq_ft = st.number_input("Size (Sq. ft.)", min_value=1)
    bath = st.number_input("Number of Bathrooms", min_value=1, step=1)
    ready_to_move = st.selectbox("Ready to Move?", ["Yes", "No"])
    under_construction = st.selectbox("Under Construction?", ["Yes", "No"])
    ready_to_move = 1 if ready_to_move == "Yes" else 0
    under_construction = 1 if under_construction == "Yes" else 0
    if st.button("Predict"):
        # Perform prediction or any other action here
        st.success("Inputs submitted successfully!")
        st.write(f"BHK: {bhk}")
        st.write(f"Size (Sq. ft.): {size_sq_ft}")
        st.write(f"Number of Bathrooms: {bath}")
        st.write(f"Ready to Move?: {'Yes' if ready_to_move else 'No'}")
        st.write(f"Under Construction?: {'Yes' if under_construction else 'No'}")
        prediction,errors=predict_house_price(df,[bhk,size_sq_ft,bath,ready_to_move,under_construction])
        st.write(f"Predicted Price for your home will be :- {round(prediction,2)}")
  
  

   
    
    
    
