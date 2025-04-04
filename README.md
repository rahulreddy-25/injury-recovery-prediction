# injury-recovery-prediction
This Streamlit app predicts injury recovery periods using a Gradient Boosting Regressor, incorporating features such as calorie intake, age, weight, fitness level, gender, type, and encoded injury type. It includes performance metrics, an interactive scatter plot, and a user input prediction feature.

Features
Model Performance Metrics: Displays MSE, MAE, R² Score, Precision, Recall, and AUC for the model.
ROC Curve: Plots the ROC curve to visualize the model's classification performance.
Actual vs Predicted Scatter Plot: Interactive Plotly chart showing actual vs. predicted recovery periods.
User Input Prediction: Accepts user input for new predictions and checks for existing records in the dataset.
Installation
Clone the repository:
git clone https://github.com/your_username/injury-recovery-prediction.git

Navigate to the project directory:
cd injury-recovery-prediction

Install the required dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py

Usage
Load the training and test datasets (injury_train.csv and injury_test.csv).
The app encodes categorical variables, scales features, and performs hyperparameter tuning using RandomizedSearchCV.
Users can view model performance metrics, plot the ROC curve, and visualize actual vs. predicted recovery periods.
Users can input new data to predict recovery periods and check for existing records in the dataset.
Files
app.py: Main Streamlit application code.
requirements.txt: List of required Python packages.
Contributing
Contributions are welcome! Please submit a pull request or open an issue for any changes or suggestions.
