import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Preparation ---
data_dict = {
    'Added Coir %': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5, 1.6, 1.7,
                     1.75],
    'CBR @ 2.5mm penetration %': [2.54, 2.906, 3.272, 3.638, 4.004, 4.37, 4.946, 5.522, 5.81, 6.028, 6.464, 6.9, 7.484,
                                  8.068, 8.36, 8.288, 8.144, 8, 7.708, 7.416, 7.27],
    'CBR @ 5.0mm penetration %': [2.18, 2.478, 2.776, 3.074, 3.372, 3.67, 4.042, 4.414, 4.6, 4.902, 5.506, 6.11, 6.386,
                                  6.662, 6.8, 6.764, 6.692, 6.62, 6.524, 6.428, 6.38],
    'MDD in gm/cc': [1.746, 1.7538, 1.7616, 1.7694, 1.7772, 1.785, 1.7982, 1.8114, 1.818, 1.8234, 1.8342, 1.845, 1.849,
                     1.853, 1.855, 1.8524, 1.8472, 1.842, 1.834, 1.826, 1.822],
    'OMC in %': [None, 16.72, 16.44, 16.16, 15.88, 15.6, 14.88, 14.16, 13.8, 13.36, 12.48, 11.6, 11.28, 10.96, 10.8,
                 10.96, 11.28, 11.6, 11.92, 12.24, 12.4]
}

data = pd.DataFrame(data_dict)

st.sidebar.download_button(label="Download CSV", data=data.to_csv().encode('utf-8'), file_name='cbr_data.csv',
                           mime='text/csv')
# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = data[['Added Coir %', 'MDD in gm/cc', 'OMC in %']]
X = imputer.fit_transform(X)

# Define targets for 2.5mm and 5.0mm CBR penetration
y_25 = data['CBR @ 2.5mm penetration %']
y_50 = data['CBR @ 5.0mm penetration %']

# Split the data into training and testing sets
X_train_25, X_test_25, y_train_25, y_test_25 = train_test_split(X, y_25, test_size=0.2, random_state=42)
X_train_50, X_test_50, y_train_50, y_test_50 = train_test_split(X, y_50, test_size=0.2, random_state=42)

# --- Model Training ---
model_25 = LinearRegression()
model_25.fit(X_train_25, y_train_25)

model_50 = LinearRegression()
model_50.fit(X_train_50, y_train_50)

# --- Streamlit App Layout ---
st.title("CBR Prediction")

# Sidebar Layout
st.sidebar.header("User Input")
st.header('CBR Prediction for Coir')
# Input Fields for Added Coir %, MDD, and OMC
added_coir = st.sidebar.number_input('Added Coir %', min_value=0.0, max_value=100.0, step=0.01)
mdd = st.sidebar.number_input('MDD in gm/cc', min_value=0.0, max_value=3.0, step=0.0001, format="%.3f")
omc = st.sidebar.number_input('OMC in %', min_value=0.0, max_value=100.0, step=0.01)

# Predict Button
if st.sidebar.button('Predict'):
    features = [[added_coir, mdd, omc]]
    features = imputer.transform(features)  # Impute missing values
    cbr_25 = model_25.predict(features)[0]
    cbr_50 = model_50.predict(features)[0]
    st.write(f'### Predicted CBR @ 2.5mm: {cbr_25:.2f}')
    st.write(f'### Predicted CBR @ 5.0mm: {cbr_50:.2f}')

# --- Comparison with Original Laboratory Values ---
st.sidebar.header("Comparison with Original Laboratory Values")

st.write("### Comparison with Original Laboratory Values")
data['Predicted CBR @ 2.5mm'] = model_25.predict(X)
data['Predicted CBR @ 5.0mm'] = model_50.predict(X)
comparison_df = data[['Added Coir %', 'MDD in gm/cc', 'OMC in %', 'CBR @ 2.5mm penetration %', 'Predicted CBR @ 2.5mm',
                      'CBR @ 5.0mm penetration %', 'Predicted CBR @ 5.0mm']]
st.write(comparison_df)

st.title("CBR Predictor and Data Visualization")

st.sidebar.header("Visualization Options")
plot_choice = st.sidebar.selectbox("Choose Plot Type",
                                   ["Line Chart", "Bar Chart", "Scatter Plot", "Pie Chart", "Heatmap"])

if plot_choice == "Line Chart":
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(comparison_df['Added Coir %'], comparison_df['CBR @ 2.5mm penetration %'], 'bo-',
               label='Original CBR @ 2.5mm')
    ax[0].plot(comparison_df['Added Coir %'], comparison_df['Predicted CBR @ 2.5mm'], 'ro-',
               label='Predicted CBR @ 2.5mm')
    ax[0].set_xlabel('Added Coir %')
    ax[0].set_ylabel('CBR @ 2.5mm')
    ax[0].legend()

    ax[1].plot(comparison_df['Added Coir %'], comparison_df['CBR @ 5.0mm penetration %'], 'bo-',
               label='Original CBR @ 5.0mm')
    ax[1].plot(comparison_df['Added Coir %'], comparison_df['Predicted CBR @ 5.0mm'], 'ro-',
               label='Predicted CBR @ 5.0mm')
    ax[1].set_xlabel('Added Coir %')
    ax[1].set_ylabel('CBR @ 5.0mm')
    ax[1].legend()
    st.pyplot(fig)

elif plot_choice == "Bar Chart":
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    comparison_df.plot(kind='bar', x='Added Coir %', y=['CBR @ 2.5mm penetration %', 'Predicted CBR @ 2.5mm'], ax=ax[0])
    comparison_df.plot(kind='bar', x='Added Coir %', y=['CBR @ 5.0mm penetration %', 'Predicted CBR @ 5.0mm'], ax=ax[1])
    st.pyplot(fig)

elif plot_choice == "Scatter Plot":
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].scatter(comparison_df['Added Coir %'], comparison_df['CBR @ 2.5mm penetration %'],
                  label='Original CBR @ 2.5mm')
    ax[0].scatter(comparison_df['Added Coir %'], comparison_df['Predicted CBR @ 2.5mm'], label='Predicted CBR @ 2.5mm')
    ax[0].set_xlabel('Added Coir %')
    ax[0].set_ylabel('CBR @ 2.5mm')
    ax[0].legend()

    ax[1].scatter(comparison_df['Added Coir %'], comparison_df['CBR @ 5.0mm penetration %'],
                  label='Original CBR @ 5.0mm')
    ax[1].scatter(comparison_df['Added Coir %'], comparison_df['Predicted CBR @ 5.0mm'], label='Predicted CBR @ 5.0mm')
    ax[1].set_xlabel('Added Coir %')
    ax[1].set_ylabel('CBR @ 5.0mm')
    ax[1].legend()
    st.pyplot(fig)

elif plot_choice == "Pie Chart":
    fig, ax = plt.subplots(figsize=(8, 8))
    sizes = [comparison_df['CBR @ 2.5mm penetration %'].mean(), comparison_df['Predicted CBR @ 2.5mm'].mean(),
             comparison_df['CBR @ 5.0mm penetration %'].mean(), comparison_df['Predicted CBR @ 5.0mm'].mean()]
    labels = ['Original CBR @ 2.5mm', 'Predicted CBR @ 2.5mm', 'Original CBR @ 5.0mm', 'Predicted CBR @ 5.0mm']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

elif plot_choice == "Heatmap":
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = comparison_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# --- Data Operations ---
st.sidebar.header("Data Operations")
data_operation = st.sidebar.selectbox("Choose Data Operation",
                                      ["Data.head()", "Data.tail()", "df.describe()", "df.info()", "df.shape",
                                       "df.columns", "df.index", "df.groupby()", "df.pivot_table()", "df.corr()",
                                       "df.median()", "df.mode()", "df.min()", "df.max()", "df.sum()", "df.var()",
                                       "df.std()", "df.matrics()"])

if data_operation == "Data.head()":
    st.write(data.head())
elif data_operation == "Data.tail()":
    st.write(data.tail())
elif data_operation == "df.describe()":
    st.write(data.describe())
elif data_operation == "df.info()":
    st.write(data.info())
elif data_operation == "df.shape":
    st.write(data.shape)
elif data_operation == "df.columns":
    st.write(data.columns)
elif data_operation == "df.index":
    st.write(data.index)
elif data_operation == "df.groupby()":
    st.write(data.groupby("Added Coir %").mean())
elif data_operation == "df.pivot_table()":
    st.write(data.pivot_table(values="CBR @ 2.5mm penetration %", columns="Added Coir %", aggfunc="mean"))
elif data_operation == "df.corr()":
    st.write(data.corr())
elif data_operation == "df.median()":
    st.write(data.median())
elif data_operation == "df.mode()":
    st.write(data.mode())
elif data_operation == "df.min()":
    st.write(data.min())
elif data_operation == "df.max()":
    st.write(data.max())
elif data_operation == "df.sum()":
    st.write(data.sum())
elif data_operation == "df.var()":
    st.write(data.var())
elif data_operation == "df.std()":
    st.write(data.std())

# --- Upload Other Datasets ---


# Sidebar for page navigation
page = st.sidebar.radio("Select Page", ("Upload CSV Data",))

st.title("CBR Predictor and Data Visualization")
st.write("Upload a CSV file to perform data operations and CBR predictions.")


# Function for experiments with data
def experiment_with_data(data):
    # Check if the necessary columns exist in the uploaded data
    if 'Added Coir %' in data.columns and 'CBR @ 2.5mm penetration %' in data.columns and 'CBR @ 5.0mm penetration %' in data.columns:

        # Data Preprocessing: Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = data[['Added Coir %', 'MDD in gm/cc', 'OMC in %']].fillna(0)  # Handling missing data
        X = imputer.fit_transform(X)

        # Define targets
        y_25 = data['CBR @ 2.5mm penetration %']
        y_50 = data['CBR @ 5.0mm penetration %']

        # Split data into train/test sets
        X_train_25, X_test_25, y_train_25, y_test_25 = train_test_split(X, y_25, test_size=0.2, random_state=42)
        X_train_50, X_test_50, y_train_50, y_test_50 = train_test_split(X, y_50, test_size=0.2, random_state=42)

        # Create and train the model for CBR @ 2.5mm and CBR @ 5.0mm penetration
        model_25 = LinearRegression()
        model_25.fit(X_train_25, y_train_25)

        model_50 = LinearRegression()
        model_50.fit(X_train_50, y_train_50)

        # Make predictions for the entire dataset
        data['Predicted CBR @ 2.5mm'] = model_25.predict(X)
        data['Predicted CBR @ 5.0mm'] = model_50.predict(X)

        # Visualizations
        st.write("### Data Analysis and Visualizations")
        st.write("Comparison of Original vs Predicted CBR Values")

        # Interactive Plot Selection
        plot_type = st.selectbox("Select Plot Type", ["Line Chart", "Bar Chart", "Scatter Plot", "Heatmap"])

        if plot_type == "Line Chart":
            fig, ax = plt.subplots(2, 1, figsize=(10, 10))
            ax[0].plot(data['Added Coir %'], data['CBR @ 2.5mm penetration %'], label='Original CBR @ 2.5mm',
                       color='blue')
            ax[0].plot(data['Added Coir %'], data['Predicted CBR @ 2.5mm'], label='Predicted CBR @ 2.5mm', color='red')
            ax[1].plot(data['Added Coir %'], data['CBR @ 5.0mm penetration %'], label='Original CBR @ 5.0mm',
                       color='blue')
            ax[1].plot(data['Added Coir %'], data['Predicted CBR @ 5.0mm'], label='Predicted CBR @ 5.0mm', color='red')
            ax[0].set_xlabel('Added Coir %')
            ax[0].set_ylabel('CBR @ 2.5mm')
            ax[0].legend()
            ax[1].set_xlabel('Added Coir %')
            ax[1].set_ylabel('CBR @ 5.0mm')
            ax[1].legend()
            st.pyplot(fig)

        elif plot_type == "Bar Chart":
            fig, ax = plt.subplots(2, 1, figsize=(10, 10))
            data.plot(kind='bar', x='Added Coir %', y=['CBR @ 2.5mm penetration %', 'Predicted CBR @ 2.5mm'], ax=ax[0])
            data.plot(kind='bar', x='Added Coir %', y=['CBR @ 5.0mm penetration %', 'Predicted CBR @ 5.0mm'], ax=ax[1])
            st.pyplot(fig)

        elif plot_type == "Scatter Plot":
            fig, ax = plt.subplots(2, 1, figsize=(10, 10))
            ax[0].scatter(data['Added Coir %'], data['CBR @ 2.5mm penetration %'], label='Original CBR @ 2.5mm')
            ax[0].scatter(data['Added Coir %'], data['Predicted CBR @ 2.5mm'], label='Predicted CBR @ 2.5mm')
            ax[1].scatter(data['Added Coir %'], data['CBR @ 5.0mm penetration %'], label='Original CBR @ 5.0mm')
            ax[1].scatter(data['Added Coir %'], data['Predicted CBR @ 5.0mm'], label='Predicted CBR @ 5.0mm')
            ax[0].set_xlabel('Added Coir %')
            ax[0].set_ylabel('CBR @ 2.5mm')
            ax[1].set_xlabel('Added Coir %')
            ax[1].set_ylabel('CBR @ 5.0mm')
            ax[0].legend()
            ax[1].legend()
            st.pyplot(fig)

        elif plot_type == "Heatmap":
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = data.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        # Experiment with different sliders
        st.write("### Experiment with Data")

        # Slider for selecting a range of values for added coir percentage
        coir_range = st.slider("Select Range for Added Coir %", min_value=float(data['Added Coir %'].min()),
                               max_value=float(data['Added Coir %'].max()), step=0.1)
        filtered_data = data[data['Added Coir %'] <= coir_range]
        st.write("Filtered Data", filtered_data)

    else:
        st.write("Uploaded data does not have the expected columns.")


# Page 2 - Upload CSV Data
if page == "Upload CSV Data":
    # File uploader component
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Load the data from the uploaded CSV file
        data = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.write("### Uploaded Data", data)

        # Run experiments on the uploaded data
        experiment_with_data(data)
    else:
        st.write("Please upload a CSV file to proceed.")
# Footer section with your name and photo




st.logo("sati.jpg", size="large", link=None, icon_image=None)

st.image("akshya.jpg", caption=None, width=100)


st.markdown("""
    <div style='display: flex; align-items: center; justify-content: center;'>
        <div style='text-align: center;'>
            <p>Made with ❤️ by Akshay Mangriya (0108CE20ME27) M-Tech Scholar Transportation Engineering </p>

    </div>
""", unsafe_allow_html=True)