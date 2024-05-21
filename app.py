import streamlit as st
from joblib import load

st.set_page_config(page_icon="favicon.webp",page_title="Breast Cancer Predicition")




# Load KNN model
@st.cache_data
def load_knn_model():
    knn_model = load('knn_model.pkl')
    return knn_model

# Main function
def main():
    st.title('Breast Cancer Diagnosis Prediction')



    # Load KNN model
    knn_model = load_knn_model()

    # Sidebar for user inputs
    st.sidebar.header('User Input')
    # Add input fields for all 30 features
    features = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
        'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
        'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
        'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
        'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
        'symmetry_worst', 'fractal_dimension_worst'
    ]
    user_inputs = {}
    for feature in features:
        user_inputs[feature] = st.sidebar.number_input(feature.capitalize(), min_value=0.0)

    # Prediction
    if st.button('Predict'):
        # Prepare input data
        input_data = [user_inputs[feature] for feature in features]
        # Perform prediction based on user inputs
        prediction = knn_model.predict([input_data])[0]
        diagnosis = 'Malignant' if prediction == 1 else 'Benign'
        st.write(f'The predicted diagnosis is: {diagnosis}')


    # Footer
    footer_html = """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px 0;
    }
    </style>
<div class="footer">Made by: <a href="https://www.linkedin.com/in/abdellah-abrkaoui-815186229/">ABRKAOUI ABDELLAH</a> & <a href="https://www.linkedin.com/in/ilyas-chrami-682551239/">CHRAMI ILYAS</a></div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

    st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)


if __name__ == '__main__':
    main()
