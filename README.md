# Smart_Irrigation_AICTE_Shell
AICTE Internship Cycle 2 — Smart Irrigation project using soil moisture and weather data. Python + Machine Learning in Jupyter Notebook.
# Smart Irrigation Advisor

## Requirements and Setup

### 1. Clone or Download the Project
- Download the project files or clone the GitHub repository containing the source code and necessary files.

### 2. Install Python (if not installed)
- Ensure Python 3.7 or higher is installed on your system.
- Download from: https://www.python.org/downloads/

### 3. Install Required Python Packages
Open a terminal or command prompt and run the following command to install all required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### 4. Running the Application Locally

- Place your sensor CSV file (with columns `sensor_0` to `sensor_19`) in the project folder.  
- Run the Streamlit app by executing this command in your terminal at the project directory:

```bash
streamlit run app.py
```

- This will open a local web page where you can upload your sensor CSV, view irrigation predictions, and interact with the app.

***

## Project Explanation

### Overview

The **Smart Irrigation Advisor** project aims to improve water use efficiency in farming by predicting parcel-wise irrigation needs based on soil and environmental sensor data.

### Key Components

- **Data**:  
  Uses 20 sensor readings per sample representing environmental and soil parameters such as soil moisture, temperature, humidity, and more.

- **Machine Learning Model**:  
  A multi-label classification model based on a Random Forest algorithm wrapped in a MultiOutputClassifier predicts whether irrigation is needed for each of three farm parcels.

- **Data Processing**:  
  Data is cleaned, unnecessary columns dropped, and features scaled using MinMaxScaler before being fed into the model.

- **Model Training and Evaluation**:  
  The model is trained on a labeled dataset and evaluated using metrics including precision, recall, and F1-score, ensuring reliable irrigation predictions.

- **Deployment**:  
  The trained model and scaler are saved together in a pipeline (`Farm_Irrigation_System.pkl`), which is loaded by the Streamlit app to provide real-time irrigation recommendations on uploaded sensor data.

- **Interactive Web Application**:  
  Built using Streamlit, the app enables users to:  
  - Upload sensor CSV files easily.  
  - View parcel-specific irrigation advice with intuitive warning and success icons.  
  - Analyze feature importance and explore trends over time.  
  - Perform what-if sensor value adjustments to observe changes in irrigation recommendations.

### Benefits

- Promotes sustainable farming by reducing water wastage.  
- Helps farmers make informed irrigation decisions tailored to specific farm parcels.  
- Enhances resource optimization using AI-powered insights delivered through an easy-to-use web interface.

***

Feel free to refer to the source code files for implementation details and the provided sample CSV template to test the app.
