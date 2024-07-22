import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
dataset = pd.read_csv(url, names=column_names, na_values="?", comment='\t', sep=" ", skipinitialspace=True)
dataset.dropna(inplace=True)

# Convert MPG to km/l
dataset['km_per_liter'] = dataset['MPG'] * 0.425144

# Add a column for "Liters of Fuel" (for demonstration, let's assume 1 cylinder is equivalent to 0.5 liters of fuel)
dataset['Liters_of_Fuel'] = dataset['Cylinders'] * 0.5

# Splitting the data into features and labels
X = dataset[['Liters_of_Fuel', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']]
y = dataset['MPG']  # Using MPG directly for prediction

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building the model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=[X_train.shape[1]]))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MAE: {test_mae:.2f} MPG")


# Function to predict fuel efficiency
def predict_fuel_efficiency(features):
    # Convert features to numpy array and reshape for prediction
    features = np.array(features).reshape(1, -1)

    # Scale the features using the same scaler used during training
    features = scaler.transform(features)

    # Make prediction using the trained model
    prediction = model.predict(features)
    return prediction[0][0]


# Tkinter user interface
class Application(tk.Tk):
    def _init_(self):
        super()._init_()
        self.title("Fuel Efficiency Predictor")
        self.geometry("800x600")
        self.configure(bg="#1e1e1e")
        self.canvas = tk.Canvas(self, width=800, height=600)
        self.canvas.pack(fill="both", expand=True)
        self.create_gradient()
        self.show_login_page()

    def create_gradient(self):
        for i in range(256):
            r = 255 - i
            b = i
            self.canvas.create_line(0, i * 2.34, 800, i * 2.34, fill=f'#{r:02x}00{b:02x}')

    def center_window(self, width=800, height=600):
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = int((screen_width / 2) - (width / 2))
        y = int((screen_height / 2) - (height / 2))
        self.geometry(f"{width}x{height}+{x}+{y}")

    def show_login_page(self):
        self.clear_screen()
        self.center_window()
        self.login_frame = tk.Frame(self.canvas, bg="red")
        self.canvas.create_window(400, 300, window=self.login_frame, anchor="center")

        tk.Label(self.login_frame, text="Login", font=("Helvetica", 24, 'bold'), bg="red", fg="white").pack(pady=20)
        tk.Label(self.login_frame, text="Username or Email or Phone", bg="red", fg="white").pack(pady=5)
        self.login_user_entry = tk.Entry(self.login_frame, width=30)
        self.login_user_entry.pack(pady=5)

        tk.Label(self.login_frame, text="Password", bg="red", fg="white").pack(pady=5)
        self.login_pass_entry = tk.Entry(self.login_frame, show="*", width=30)
        self.login_pass_entry.pack(pady=5)

        tk.Button(self.login_frame, text="Login", command=self.authenticate_user, width=20, bg="#333333",
                  fg="white").pack(pady=20)
        tk.Button(self.login_frame, text="Create New Account", command=self.show_create_account_page, width=20,
                  bg="#333333", fg="white").pack(pady=5)

    def show_create_account_page(self):
        self.clear_screen()
        self.center_window()
        self.create_account_frame = tk.Frame(self.canvas, bg="blue")
        self.canvas.create_window(400, 300, window=self.create_account_frame, anchor="center")

        tk.Label(self.create_account_frame, text="Create New Account", font=("Helvetica", 24, 'bold'), bg="blue",
                 fg="white").pack(pady=20)

        tk.Label(self.create_account_frame, text="Username", bg="blue", fg="white").pack(pady=5)
        self.new_user_entry = tk.Entry(self.create_account_frame, width=30)
        self.new_user_entry.pack(pady=5)

        tk.Label(self.create_account_frame, text="Email", bg="blue", fg="white").pack(pady=5)
        self.new_email_entry = tk.Entry(self.create_account_frame, width=30)
        self.new_email_entry.pack(pady=5)

        tk.Label(self.create_account_frame, text="Phone Number", bg="blue", fg="white").pack(pady=5)
        self.new_phone_entry = tk.Entry(self.create_account_frame, width=30)
        self.new_phone_entry.pack(pady=5)

        tk.Label(self.create_account_frame, text="Password", bg="blue", fg="white").pack(pady=5)
        self.new_pass_entry = tk.Entry(self.create_account_frame, show="*", width=30)
        self.new_pass_entry.pack(pady=5)

        tk.Label(self.create_account_frame, text="Confirm Password", bg="blue", fg="white").pack(pady=5)
        self.new_confirm_pass_entry = tk.Entry(self.create_account_frame, show="*", width=30)
        self.new_confirm_pass_entry.pack(pady=5)

        tk.Button(self.create_account_frame, text="Create Account", command=self.create_account, width=20, bg="#333333",
                  fg="white").pack(pady=20)
        tk.Button(self.create_account_frame, text="Back to Login", command=self.show_login_page, width=20, bg="#333333",
                  fg="white").pack(pady=5)

    def clear_screen(self):
        for widget in self.winfo_children():
            widget.destroy()
        self.canvas = tk.Canvas(self, width=800, height=600)
        self.canvas.pack(fill="both", expand=True)
        self.create_gradient()

    def authenticate_user(self):
        # Authentication logic (this example assumes any input is valid)
        self.show_prediction_page()

    def create_account(self):
        # Account creation logic (this example assumes any input is valid)
        self.show_login_page()

    def show_prediction_page(self):
        self.clear_screen()
        self.center_window()
        self.prediction_frame = tk.Frame(self.canvas, bg="red")
        self.canvas.create_window(400, 300, window=self.prediction_frame, anchor="center")

        tk.Label(self.prediction_frame, text="Fuel Efficiency Prediction", font=("Helvetica", 24, 'bold'), bg="red",
                 fg="white").pack(pady=20)

        self.entries = []
        labels = ['Liters of Fuel', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

        for label in labels:
            tk.Label(self.prediction_frame, text=label, bg="red", fg="white").pack(pady=5)
            entry = tk.Entry(self.prediction_frame, width=30)
            entry.pack(pady=5)
            self.entries.append(entry)

        tk.Button(self.prediction_frame, text="Predict", command=self.predict, width=20, bg="#333333", fg="white").pack(
            pady=20)
        tk.Button(self.prediction_frame, text="Exit", command=self.exit, width=20, bg="#333333", fg="white").pack(
            pady=5)

    def predict(self):
        try:
            features = [float(entry.get()) for entry in self.entries]
            prediction = predict_fuel_efficiency(features)
            messagebox.showinfo("Prediction", f"Predicted fuel efficiency: {prediction:.2f} MPG")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for all fields")

    def exit(self):
        self.destroy()


# Run the application
if _name_ == "_main_":
    app = Application()
    app.mainloop()
