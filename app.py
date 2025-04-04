import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, roc_curve, auc
import plotly.graph_objects as go
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt

# Load the training and test datasets
train_file_path = 'injury_train.csv'
test_file_path = 'injury_test.csv'

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Combine train and test data for encoding
combined_data = pd.concat([train_data, test_data])

# Encode categorical variables
label_encoder_injury = LabelEncoder()
label_encoder_gender = LabelEncoder()
label_encoder_type = LabelEncoder()

combined_data['Injury_Encoded'] = label_encoder_injury.fit_transform(combined_data['Injury'])
combined_data['Gender'] = label_encoder_gender.fit_transform(combined_data['Gender'])
combined_data['Type'] = label_encoder_type.fit_transform(combined_data['Type'])

# Split back to train and test data after encoding
train_data = combined_data.loc[train_data.index]
test_data = combined_data.loc[test_data.index]

# Define features and target variable
X_train = train_data[['Callorie', 'Age', 'Weight', 'Fitness_Level', 'Gender', 'Type', 'Injury_Encoded']]
y_train = train_data['Recovery_Period']
X_test = test_data[['Callorie', 'Age', 'Weight', 'Fitness_Level', 'Gender', 'Type', 'Injury_Encoded']]
y_test = test_data['Recovery_Period']

# Normaize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Gradient Boosting model
model = GradientBoostingRegressor(random_state=42)

# Perform hyperparameter tuning with RandomizedSearchCV
param_distributions = {
    'n_estimators': randint(50, 200),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 7)
}
random_search = RandomizedSearchCV(model, param_distributions, n_iter=50, cv=5, scoring='neg_mean_squared_error', random_state=42)
random_search.fit(X_train_scaled, y_train)
best_model = random_search.best_estimator_

# Train the model
best_model.fit(X_train_scaled, y_train)

# Predict on the evaluation set
y_pred = best_model.predict(X_test_scaled)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Define a threshold for classification (e.g., median recovery period)
threshold = y_train.median()

# Convert regression predictions to binary classification
y_test_class = (y_test > threshold).astype(int)
y_pred_class = (y_pred > threshold).astype(int)

# Calculate classification metrics
precision = precision_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test_class, y_pred)
roc_auc = auc(fpr, tpr)

# Streamlit UI
st.title("Injury Recovery Period Prediction")

st.write("### Model Performance")
st.write(f"Gradient Boosting - MSE: {mse:.2f}, MAE: {mae:.2f}, R² Score: {r2:.2f}")
st.write(f"Precision: {precision:.2f}, Recall: {recall:.2f}, AUC: {roc_auc:.2f}")

# Plot ROC curve
fig_roc = plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
st.pyplot(fig_roc)

# Decode labels for hover text
gender_mapping = {index: label for index, label in enumerate(label_encoder_gender.classes_)}
type_mapping = {index: label for index, label in enumerate(label_encoder_type.classes_)}
injury_mapping = {index: label for index, label in enumerate(label_encoder_injury.classes_)}

@st.cache_data
def plot_results(_test_data, y_test, y_pred):
    hover_texts = []
    
    for idx, row in _test_data.iterrows():
        injury_encoded = row['Injury_Encoded']
        gender = row['Gender']
        injury_text = []
        
        if isinstance(injury_encoded, list):
            for i, enc in enumerate(injury_encoded):
                injury_text.append(f"Injury {i+1}: {injury_mapping.get(enc, 'Unknown')}")
        else:
            injury_text.append(f"Injury: {injury_mapping.get(injury_encoded, 'Unknown')}")
        
        hover_text = (
            f"Index: {idx}<br>"
            + "<br>".join(injury_text)
            + f"<br>Type: {type_mapping.get(row['Type'], 'Unknown')}<br>"
            + f"Gender: {gender_mapping.get(gender, 'Unknown')}<br>"
            + f"Actual Recovery Period: {y_test.iloc[idx]}"
        )
        
        hover_texts.append(hover_text)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        marker=dict(color='blue', size=10),
        text=hover_texts,
        hoverinfo='text'
    ))
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        marker=dict(color='red'),
        name='Ideal'
    ))
    fig.update_layout(
        title='Gradient Boosting - Actual vs Predicted',
        xaxis_title='Actual Recovery Period',
        yaxis_title='Predicted Recovery Period',
        template='plotly_white',
        showlegend=False,
        hovermode='closest'
    )
    return fig

# Plot results for the best model
scatter_fig = plot_results(test_data, y_test, y_pred)
st.plotly_chart(scatter_fig)

# Show details on click
selected_point = st.selectbox('Select a point to see details', options=test_data.index.unique(), format_func=lambda x: f'Index {x}')
if selected_point is not None:
    selected_injury = test_data.loc[test_data.index == selected_point, 'Injury'].values[0]
    selected_recovery_period = y_test.loc[y_test.index == selected_point].values[0]
    st.write(f"### Details for Selected Point (Index {selected_point})")
    st.write(f"- Injury: {selected_injury}")
    st.write(f"- Actual Recovery Period: {selected_recovery_period} days")

# Function to get user input
def get_user_input():
    st.write("### Enter details to predict recovery period")
    user_injury = st.selectbox("Injury", label_encoder_injury.classes_)
    user_gender = st.selectbox("Gender", ['M', 'F'])
    user_type = st.selectbox("Type", ['Major', 'Minor'])
    user_age = st.number_input("Age (0-100)", min_value=0, max_value=100, step=1)
    user_weight = st.number_input("Weight (0-200)", min_value=0, max_value=200, step=1)
    user_calorie = st.number_input("Callorie (0-5000)", min_value=0, max_value=5000, step=1)
    user_fitness_level = st.number_input("Fitness Level (0-10)", min_value=0, max_value=10, step=1)

    return {
        'Callorie': user_calorie,
        'Age': user_age,
        'Weight': user_weight,
        'Fitness_Level': user_fitness_level,
        'Gender': 1 if user_gender == 'M' else 0,
        'Type': 1 if user_type == 'Major' else 0,
        'Injury_Encoded': label_encoder_injury.transform([user_injury])[0]
    }

# Function to check if the input data exists in the dataset
def check_existing_data(user_input, data):
    existing_record = data[
        (data['Callorie'] == user_input['Callorie']) &
        (data['Age'] == user_input['Age']) &
        (data['Weight'] == user_input['Weight']) &
        (data['Fitness_Level'] == user_input['Fitness_Level']) &
        (data['Gender'] == user_input['Gender']) &
        (data['Type'] == user_input['Type']) &
        (data['Injury_Encoded'] == user_input['Injury_Encoded'])
    ]
    if not existing_record.empty:
        return existing_record
    else:
        return None

# Get user input
user_input = get_user_input()

if st.button("Predict"):
    # Check if the input data exists in the training or test dataset
    existing_record_train = check_existing_data(user_input, train_data)
    existing_record_test = check_existing_data(user_input, test_data)
    
    if existing_record_train is not None:
        st.write("Found matching record in the training dataset")
        user_prediction = existing_record_train['Recovery_Period'].values[0]
    elif existing_record_test is not None:
        st.write("Found matching record in the test dataset")
        user_prediction = existing_record_test['Recovery_Period'].values[0]
    else:
        user_data = pd.DataFrame([user_input])
        user_data_scaled = scaler.transform(user_data)
        user_prediction = best_model.predict(user_data_scaled)[0]
    
    st.write(f"Predicted Recovery Period: {user_prediction:.2f} days")