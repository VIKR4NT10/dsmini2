
import streamlit as st
import pandas as pd
import math

from sklearn.linear_model import LinearRegression


def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in KM

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

df = pd.read_csv("food_delivery_cleaned.csv")

df["Distance_km"] = df.apply(
    lambda row: calculate_distance(
        row["Restaurant_latitude"],
        row["Restaurant_longitude"],
        row["Delivery_location_latitude"],
        row["Delivery_location_longitude"]
    ),
    axis=1
)



X = df[
    [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Distance_km",
        "Type_of_order",
        "Type_of_vehicle"
    ]
]

y = df["Time_taken(min)"]


model = LinearRegression()
model.fit(X, y)


order_type_map = {
    "Snack": 0,
    "Meal": 1,
    "Drinks": 2,
    "Buffet": 3
}

vehicle_type_map = {
    "Bike": 0,
    "Scooter": 1,
    "Cycle": 2,
    "Electric Scooter": 3
}


st.title("Food Delivery Time Predictor 🚚")
st.write("Predict delivery time using delivery distance")

#User Inputs
# ------------------------------------------------------------

age = st.number_input(
    "Delivery Person Age",
    min_value=18,
    max_value=60,
    value=25
)

rating = st.number_input(
    "Delivery Person Rating",
    min_value=1.0,
    max_value=5.0,
    value=4.2
)

distance = st.number_input(
    "Distance Between Restaurant and Customer (KM)",
    min_value=0.5,
    max_value=50.0,
    value=5.0
)

selected_order_type = st.selectbox(
    "Type of Order",
    list(order_type_map.keys())
)

selected_vehicle_type = st.selectbox(
    "Type of Vehicle",
    list(vehicle_type_map.keys())
)

order_type = order_type_map[selected_order_type]
vehicle_type = vehicle_type_map[selected_vehicle_type]


if st.button("Predict Delivery Time"):

    input_data = pd.DataFrame(
        [[
            age,
            rating,
            distance,
            order_type,
            vehicle_type
        ]],
        columns=[
            "Delivery_person_Age",
            "Delivery_person_Ratings",
            "Distance_km",
            "Type_of_order",
            "Type_of_vehicle"
        ]
    )

    predicted_time = model.predict(input_data)[0]

    # Prevent unrealistic negative values
    predicted_time = max(predicted_time, 5)

    st.subheader("Prediction Result")

    st.success(
        f"Estimated Delivery Time: {predicted_time:.2f} minutes"
    )