# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

#Step 1: Data Acquisition
#Load the dataset
dataset_path = 'shopping_trends_updated.csv'
data = pd.read_csv(dataset_path)

#Step 2: Data Analysis and Preparation
#Assume 'Subscription Status' is the target variable
X = data.drop('Subscription Status', axis=1)
y = data['Subscription Status']

#Convert categorical columns to numerical using Label Encoding
label_encoder = LabelEncoder()
categorical_cols = ['Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season', 'Shipping Type', 'Payment Method', 'Frequency of Purchases']
for col in categorical_cols:
    X[col] = label_encoder.fit_transform(X[col])

#Handle 'Yes' and 'No' in other columns
yes_no_cols = ['Discount Applied', 'Promo Code Used']
for col in yes_no_cols:
    X[col] = X[col].map({'Yes': 1, 'No': 0})

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Convert 'Yes' and 'No' to 1 and 0 for the target variable
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

#Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Step 4: Analysis and Preparation of Potential Models
#Model 1: Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

#Model 2: Support Vector Machine
svm_model = SVC(probability=True) # Enable probability for SVM ROC curve
svm_model.fit(X_train, y_train)

#Model 3: Simple Neural Network
nn_model = Sequential([
Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Step 5: Implementation and Training of the Selected Models
#Random Forest and Support Vector Machine models are already trained
#Train the Neural Network
nn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

#Step 6: Evaluation of the Selected Models
#Evaluate the Random Forest model
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy}")

#Evaluate the Support Vector Machine model
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"Support Vector Machine Accuracy: {svm_accuracy}")

#Evaluate the Neural Network model
nn_predictions = nn_model.predict(X_test)
nn_predictions = nn_predictions > 0.5 # Convert probabilities to binary predictions
nn_accuracy = accuracy_score(y_test, nn_predictions)
print(f"Neural Network Accuracy: {nn_accuracy}")

# Display classification reports and confusion matrices
print("Classification Report (Random Forest):\n", classification_report(y_test, rf_predictions))
print("Confusion Matrix (Random Forest):\n", confusion_matrix(y_test, rf_predictions))
print("========================")
print("Classification Report (Support Vector Machine):\n", classification_report(y_test, svm_predictions))
print("Confusion Matrix (Support Vector Machine):\n", confusion_matrix(y_test, svm_predictions))
print("========================")
print("Classification Report (Neural Network):\n", classification_report(y_test, nn_predictions))
print("Confusion Matrix (Neural Network):\n", confusion_matrix(y_test, nn_predictions))
print("========================")
#Step 6.1: Visualizations
#Visualize the confusion matrices
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'/graphs{model_name}_confusion_matrix.png', bbox_inches='tight')
    plt.show()

#Visualize Random Forest Confusion Matrix
rf_cm = confusion_matrix(y_test, rf_predictions)
plot_confusion_matrix(rf_cm, 'Random Forest')

#Visualize Support Vector Machine Confusion Matrix
svm_cm = confusion_matrix(y_test, svm_predictions)
plot_confusion_matrix(svm_cm, 'Support Vector Machine')

#Visualize Neural Network Confusion Matrix
nn_cm = confusion_matrix(y_test, nn_predictions)
plot_confusion_matrix(nn_cm, 'Neural Network')

#Visualize classification reports
def plot_classification_report(y_true, y_pred, model_name):
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    plt.figure(figsize=(8, 4))
    sns.heatmap(df_report.iloc[:-1, :3], annot=True, cmap='Blues', cbar=False)
    plt.title(f'{model_name} Classification Report')
    plt.savefig(f'/graphs/{model_name}_classification_report.png', bbox_inches='tight')
    plt.show()

#Visualize Random Forest Classification Report
plot_classification_report(y_test, rf_predictions, 'Random Forest')

#Visualize Support Vector Machine Classification Report
plot_classification_report(y_test, svm_predictions, 'Support Vector Machine')

#Visualize Neural Network Classification Report
plot_classification_report(y_test, nn_predictions, 'Neural Network')

#Visualize ROC curves
def plot_roc_curve(y_true, y_prob, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(f'/graphs/{model_name}_roc_curve.png', bbox_inches='tight')
    plt.show()

#Visualize Random Forest ROC Curve
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, rf_probabilities, 'Random Forest')

#Visualize Support Vector Machine ROC Curve
svm_probabilities = svm_model.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, svm_probabilities, 'Support Vector Machine')

#Visualize Neural Network ROC Curve
nn_probabilities = nn_model.predict(X_test)
plot_roc_curve(y_test, nn_probabilities, 'Neural Network')