
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(
    repo_id="avijeetV/Tourism-Package-Prediction",
    filename="best_tourism_package_model_v1.joblib",
    repo_type="model"
)

model = joblib.load(model_path)

# Streamlit UI
st.title("Wellness Tourism Package Sale Prediction")
st.write("""
This application predicts whether a customer is likely to purchase the
Wellness Tourism Package based on their demographic and interaction details.
""")

age = st.number_input("Age", min_value=18, max_value=100, value=35)

typeofcontact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

citytier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

durationofpitch = st.number_input(
    "Duration of Pitch",
    min_value=1,
    max_value=100,
    value=15
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

numberofpersonvisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2
)

numberoffollowups = st.number_input(
    "Number of Follow-ups",
    min_value=0,
    max_value=10,
    value=3
)

productpitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

preferredpropertystar = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

maritalstatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

numberoftrips = st.number_input(
    "Number of Trips per Year",
    min_value=0,
    max_value=20,
    value=2
)

passport = st.selectbox(
    "Passport Available",
    [0, 1]
)

pitchsatisfactionscore = st.slider(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

owncar = st.selectbox(
    "Owns a Car",
    [0, 1]
)

numberofchildrenvisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=5,
    value=0
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

monthlyincome = st.number_input(
    "Monthly Income",
    min_value=0,
    value=20000
)

# Create DataFrame
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": citytier,
    "DurationOfPitch": durationofpitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": numberofpersonvisiting,
    "NumberOfFollowups": numberoffollowups,
    "ProductPitched": productpitched,
    "PreferredPropertyStar": preferredpropertystar,
    "MaritalStatus": maritalstatus,
    "NumberOfTrips": numberoftrips,
    "Passport": passport,
    "PitchSatisfactionScore": pitchsatisfactionscore,
    "OwnCar": owncar,
    "NumberOfChildrenVisiting": numberofchildrenvisiting,
    "Designation": designation,
    "MonthlyIncome": monthlyincome
}])
# Debug: show the input
st.write(input_data)
if st.button("Predict"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            f"The customer is likely to purchase the Wellness Tourism Package "
            f"(Probability: {probability:.2%})."
        )
    else:
        st.error(
            f"The customer is unlikely to purchase the Wellness Tourism Package "
            f"(Probability: {probability:.2%})."
        )
