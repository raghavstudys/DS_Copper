import streamlit as st
import numpy as np
import pickle

# Set page configuration
st.set_page_config(
    page_title="PhonePe Pulse",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load resources
@st.cache_resource
def load_resources():
    with open('/Users/shanthakumark/Downloads/rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('/Users/shanthakumark/Downloads/country_to_transformed_.pkl', 'rb') as f:
        country_transformed = pickle.load(f)
    with open('/Users/shanthakumark/Downloads/item_to_transformed_.pkl', 'rb') as f:
        items_transformed = pickle.load(f)
    with open('/Users/shanthakumark/Downloads/status_to_transformed_.pkl', 'rb') as f:
        status_to_transformed = pickle.load(f)
    with open('/Users/shanthakumark/Downloads/data.pkl', 'rb') as data:
        df = pickle.load(data)
    with open('/Users/shanthakumark/Downloads/rf_model_class_1.pkl', 'rb') as f:
        rf_class_model_1 = pickle.load(f)
    with open('/Users/shanthakumark/Downloads/X_sample.pkl', 'rb') as f:
        X_sample_class = pickle.load(f)
    with open('/Users/shanthakumark/Downloads/transformed_to_status.pkl', 'rb') as f:
        trans_status_class = pickle.load(f)
    
    # Reverse the status_to_transformed dictionary
    trans_status_class_reversed = {v: k for k, v in status_to_transformed.items()}
    
    return rf_model, country_transformed, items_transformed, status_to_transformed, df, rf_class_model_1, X_sample_class, trans_status_class_reversed

# Load all resources
rf_model, country_transformed, items_transformed, status_to_transformed, df, rf_class_model_1, X_sample_class, trans_status_class_reversed = load_resources()

# Create tabs
tab1, tab2 = st.tabs(["Selling Price Prediction", "Status Prediction"])

# Tab 1: Selling Price Prediction
with tab1:
    # Define custom CSS for styling
    st.markdown(
        """
        <style>
        .header {
            background-color: #b87333;
            color: white;
            padding: 11px;
            border-radius: 4px;
            text-align: center;
            font-size:30px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="header">Predicting Selling Price $</div>', unsafe_allow_html=True)
    st.write('--------------------------------------------------------------------------------------------------------------| DS COPPER MODELLING |--------------------------------------------------------------------------------------------------------------', unsafe_allow_html=True)
    st.markdown('')

    col1, col2 = st.columns([1, 1])

    with col1:
        pred_s_1 = st.container(height=489)
        with pred_s_1:
            quantity_tons = st.number_input("Enter the quantity tons of copper:", min_value=0.0, max_value=10000.0)
            thickness = st.number_input("Enter thickness of copper:", min_value=0.0, max_value=10.0)
            width = st.number_input("Enter the width of copper:", min_value=0.0, max_value=2000.0)
            status = st.selectbox("Enter the status", options=list(status_to_transformed.keys()))
    
    with col2:
        pred_s_2  = st.container(height=489)
        with pred_s_2:
            country = st.selectbox("Select the Country", options=list(country_transformed.keys()))
            item_type = st.selectbox("Select the item type", options=list(items_transformed.keys()))
            application = st.selectbox("Select the application type", options=df['application'].unique())
            product_ref = st.selectbox("Select the Product Ref", options=df['product_ref'].unique())
            delivery_time = st.number_input("Enter the delivery time:", min_value=0, max_value=365)

    if st.button('Predict'):
        try:
            status_encode = status_to_transformed.get(status)
            country_encode = country_transformed.get(country)
            item_type_encode = items_transformed.get(item_type)

            y = [quantity_tons, thickness, width, status_encode, country_encode, item_type_encode, application, product_ref, delivery_time]
            final_y = np.array(y).reshape(1, -1)
            
            pred = rf_model.predict(final_y)
            predicted_price = pred[0]

            formatted_price = f"${predicted_price:,.2f}"
            st.metric(label="Predicted Selling Price:", value=formatted_price)
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Tab 2: Status Prediction
with tab2:
    # Define custom CSS for styling
    st.markdown(
        """
        <style>
        .header {
            background-color: #b87333;
            color: white;
            padding: 11px;
            border-radius: 4px;
            text-align: center;
            font-size:30px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="header">Predicting Status</div>', unsafe_allow_html=True)
    st.write('--------------------------------------------------------------------------------------------------------------| DS COPPER MODELLING |--------------------------------------------------------------------------------------------------------------', unsafe_allow_html=True)
    st.markdown('')

    col_class1, col_class2 = st.columns([1, 1])

    with col_class1:
        class_c1 = st.container(height=400)
        with class_c1:
            quantity_tons_class = st.number_input("Enter the quantity tons:", min_value=0.0, max_value=10000.0)
            thickness_class = st.number_input("Enter thickness:", min_value=0.0, max_value=10.0)
            width_class = st.number_input("Enter the width:", min_value=0.0, max_value=2000.0)
            country_class = st.selectbox("Country", options=list(country_transformed.keys()))
    
    with col_class2:
        class_c2 = st.container(height=400)
        with class_c2:
            item_type_class = st.selectbox("Item type", options=list(items_transformed.keys()))
            application_class = st.selectbox("Application type", options=df['application'].unique())
            product_ref_class = st.selectbox("Product Ref", options=df['product_ref'].unique())
            delivery_time_class = st.number_input("Delivery time:", min_value=0, max_value=365)

    if st.button('Predict Status'):
        try:
            country_encode_class = country_transformed.get(country_class)
            item_type_encode_class = items_transformed.get(item_type_class)

            X_Class = [quantity_tons_class, thickness_class, width_class, country_encode_class, item_type_encode_class, application_class, product_ref_class, delivery_time_class]
            x_class = np.array(X_Class).reshape(1, -1)
            
            pred = rf_class_model_1.predict(x_class)
            pred_status = pred[0]

            # Reverse the prediction to get the status string
            final_pred = trans_status_class_reversed.get(pred_status, "Unknown status")
            st.metric(label = "Predicted Status:",value=final_pred)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
