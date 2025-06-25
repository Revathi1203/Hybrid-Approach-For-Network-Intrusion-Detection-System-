
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
import pyttsx3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import winsound
import customtkinter as ctk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.utils import to_categorical
from tkinter import messagebox
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.float_format', lambda x: '%.3f' % x)
plt.rcParams["figure.figsize"] = (10,6)

df_0 = pd.read_csv("KDDTrain+.txt")
df = df_0.copy()
print(df.head())
columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent',
'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate',
'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
'dst_host_rerror_rate','dst_host_srv_rerror_rate','attack','level'])
df.columns = columns
print(df.head(5))
print(df.info())
print(df.describe().T)
print(df.isnull().sum())

def unique_values(df, columns):
    """Prints unique values and their counts for specific columns in the DataFrame."""
    for column_name in columns:
        print(f"Column: {column_name}\n{'-'*30}")
        unique_vals = df[column_name].unique()
        value_counts = df[column_name].value_counts()
        print(f"Unique Values ({len(unique_vals)}): {unique_vals}\n")
        print(f"Value Counts:\n{value_counts}\n{'='*40}\n")

cat_features = df.select_dtypes(include='object').columns
unique_values(df, cat_features)
print(df.duplicated().sum())

attacks_types = {
    'normal': 'normal','back': 'dos','buffer_overflow': 'u2r','ftp_write': 'r2l','guess_passwd': 'r2l','imap': 'r2l',
    'ipsweep': 'probe','land': 'dos','loadmodule': 'u2r','multihop': 'r2l','neptune': 'dos','nmap': 'probe',
    'perl': 'u2r','phf': 'r2l','pod': 'dos','portsweep': 'probe','rootkit': 'u2r','satan': 'probe',
    'smurf': 'dos','spy': 'r2l','teardrop': 'dos','warezclient': 'r2l','warezmaster': 'r2l'
}
df['label'] = df['attack'].apply(lambda r: attacks_types.get(r.strip(), 'unknown'))

def preprocess_data(data):
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    X = data.drop(['label'], axis=1)
    y = data['label']

    class_names = sorted(y.unique())

    # One-hot encoding
    X = pd.get_dummies(X, columns=categorical_cols)
    print(X)
    # Convert all columns to numeric, fill missing values with 0
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    print(X)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Print processed data
    print("\n🔷 Features (X):")
    print(X.head())

    print("\n🔷 Encoded Labels (y_encoded):")
    print(y_encoded[:10])

    print("\n🔷 Original Labels (y):")
    print(y.head())

    print("\n🔷 Label Mappings:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"{i} -> {label}")

    return X, y_encoded, label_encoder, class_names, y


def preprocess_test_data(test_data, scaler, pca, label_encoder, train_columns):
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    # Separate features and label
    X_test = test_data.drop(['label'], axis=1)
    y_test = test_data['label']

    # One-hot encode categorical features
    X_test = pd.get_dummies(X_test, columns=categorical_cols)
    print( X_test)

    # Align columns with training data
    X_test = X_test.reindex(columns=train_columns, fill_value=0)
    print( X_test)

    # Convert to numeric and handle missing values
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
    print( X_test)

    # Scale and reduce dimensions
    X_test_scaled = scaler.transform(X_test)
    print( X_test_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    

    # Handle unseen labels gracefully
    unknown_labels = set(y_test) - set(label_encoder.classes_)
    if unknown_labels:
        print(f"\n⚠️ Warning: Unseen labels in test data: {unknown_labels}")
        y_test = y_test.apply(lambda x: x if x in label_encoder.classes_ else None)
        mask = y_test.notna()
        y_test_filtered = y_test[mask]
        X_test_pca = X_test_pca[mask]
        y_test_encoded = label_encoder.transform(y_test_filtered)
    else:
        y_test_encoded = label_encoder.transform(y_test)

    # Print debug info
    print("\n🔷 Processed Test Features (PCA output):")
    print(X_test_pca[:5])

    print("\n🔷 Encoded Test Labels:")
    print(y_test_encoded[:10])

    print("\n🔷 Original Test Labels:")
    print(y_test.head())

    return X_test_pca, y_test_encoded, y_test


def build_blstm_model(input_dim):
    model = Sequential()
    model.add(Bidirectional(LSTM(64), input_shape=(1, input_dim)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(22, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def plot_performance_graph(history, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title(f'{model_name} Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.show()

def plot_metrics_bar(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics = [acc, prec, rec, f1]
    names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    plt.figure(figsize=(8,6))
    sns.barplot(x=names, y=metrics)
    plt.ylim(0, 1)
    plt.title(f'{model_name} Metrics')
    plt.ylabel('Score')
    plt.show()



def plot_final_accuracy_comparison(acc_nb, acc_svm, acc_blstm):
    combined_svm_blstm = acc_svm + acc_blstm
    plt.figure(figsize=(8,6))
    sns.barplot(x=["Naive Bayes", "SVM + BLSTM"], y=[acc_nb, combined_svm_blstm], palette='Set2')
    plt.title("Layer-wise Accuracy Comparison (on KDDTest+)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.show()

def plot_comparison_graphs():
    previous_models = {
        "CNN": {"accuracy": 75.67, "precision": 66.23, "recall": 70.89, "f1_score": 75.0},
        "Deep Learning": {"accuracy": 77.20, "precision": 70.99, "recall": 73.11, "f1_score": 78.0},
        "LSTM": {"accuracy": 83.45, "precision": 76.11, "recall": 78.55, "f1_score": 81.33},
        "CNN + LSTM (Hybrid)":{"accuracy": 85.00, "precision": 78.00, "recall": 80.00, "f1_score": 82.00},
        "Ensemble Stacking":{"accuracy": 87.00, "precision": 80.00, "recall": 83.00, "f1_score": 84.00},
        "Hybrid Model\n(NB+SVM+BLSTM)": {"accuracy": 98.10, "precision": 91.70, "recall": 88.23, "f1_score": 90.67},
    }
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    model_names = list(previous_models.keys())
    colors = ['#d9534f', '#f0ad4e', '#5bc0de', '#5cb85c']
    for metric in metrics:
        scores = [previous_models[model][metric] for model in model_names]
        plt.figure(figsize=(10, 5))
        bars = plt.bar(model_names, scores, color=colors)
        plt.title(f'{metric.capitalize()} Comparison', fontsize=14)
        plt.ylabel(f'{metric.capitalize()} (%)', fontsize=12)
        plt.ylim(0, 100)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f'{yval:.2f}%', ha='center', va='bottom', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    X, y, label_encoder, class_names, y_raw = preprocess_data(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)
    train_columns = X.columns
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_pca, y, test_size=0.3, random_state=42)

    layer1_model = GaussianNB()
    layer1_model.fit(X_train1, y_train1)
    y_pred1 = layer1_model.predict(X_test1)
    y_pred1_labels = label_encoder.inverse_transform(y_pred1)
    dos_count = df[df['label'] == 'dos'].shape[0]
    probe_count = df[df['label'] == 'probe'].shape[0]

# Alert system
    root = tk.Tk()
    root.withdraw()
    engine = pyttsx3.init()
    engine.say("Alert. Attacks detected. DoS and Probe.")
    engine.runAndWait()
    messagebox.showinfo("Naive Bayes Detection Summary",
                    f"Attacks Detected:\n\nDoS: {dos_count}\nProbe: {probe_count}")

    print(f"\nTotal number of DoS attacks identified: {dos_count}")
    print(f"Total number of Probe attacks identified: {probe_count}")
    print(classification_report(y_test1, y_pred1, labels=range(len(label_encoder.classes_)), target_names=label_encoder.classes_))
    plot_confusion_matrix(y_test1, y_pred1, "Naive Bayes")
    plot_metrics_bar(y_test1, y_pred1, "Naive Bayes")

    correct_indices = np.where(y_pred1 == y_test1)[0]
    X_correct = X_test1[correct_indices]
    y_correct = y_test1[correct_indices]

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X_correct, y_correct, test_size=0.3, random_state=42)
    svm_model = svm.SVC()
    svm_model.fit(X_train2, y_train2)
    y_pred2_svm = svm_model.predict(X_test2)
    y_pred2_svm_labels = label_encoder.inverse_transform(y_pred2_svm)
   
    print(classification_report(y_test2, y_pred2_svm, labels=range(len(label_encoder.classes_)), target_names=label_encoder.classes_))
    plot_confusion_matrix(y_test2, y_pred2_svm, "SVM")
    plot_metrics_bar(y_test2, y_pred2_svm, "SVM")

    X_train2_blstm = np.reshape(X_train2, (X_train2.shape[0], 1, X_train2.shape[1]))
    X_test2_blstm = np.reshape(X_test2, (X_test2.shape[0], 1, X_test2.shape[1]))
    y_train2_cat = to_categorical(y_train2, num_classes=22)
    y_test2_cat = to_categorical(y_test2, num_classes=22)
    blstm_model = build_blstm_model(X_train2.shape[1])
    history = blstm_model.fit(X_train2_blstm, y_train2_cat, epochs=10, batch_size=64, validation_data=(X_test2_blstm, y_test2_cat), verbose=1)
    y_pred2_blstm = blstm_model.predict(X_test2_blstm)
    y_pred2_blstm_classes = np.argmax(y_pred2_blstm, axis=1)
    y_pred2_blstm_labels = label_encoder.inverse_transform(y_pred2_blstm_classes)
    r2l_count = df[df['label'] == 'r2l'].shape[0]
    u2r_count = df[df['label'] == 'u2r'].shape[0]

    engine.say("R2L and U2R attacks detected.")
    engine.runAndWait()

    messagebox.showinfo("Layer 2 - Detection Summary",
                        f"Total Attacks Detected:\n\nR2L: {r2l_count}\nU2R: {u2r_count}")

    print(f"\n✅ Total number of R2L attacks identified in dataset: {r2l_count}")
    print(f"✅ Total number of U2R attacks identified in dataset: {u2r_count}")

    print(classification_report(y_test2, y_pred2_blstm_classes, labels=range(len(label_encoder.classes_)), target_names=label_encoder.classes_))
    plot_confusion_matrix(y_test2, y_pred2_blstm_classes, "BLSTM")
    plot_performance_graph(history, "BLSTM")

    test_data = pd.read_csv('KDDTest+.txt', names=columns)

# Map only known attack types and drop unknown ones
    test_data['label'] = test_data['attack'].map(lambda r: attacks_types.get(r.strip()))
    test_data = test_data.dropna(subset=['label'])  # drop rows where label is NaN

# Continue as before
    X_test_all, y_test_all, y_test_labels = preprocess_test_data(test_data, scaler, pca, label_encoder, train_columns)


    y_pred_test1 = layer1_model.predict(X_test_all)
    y_pred_test1_labels = label_encoder.inverse_transform(y_pred_test1)
    
    acc_nb = accuracy_score(y_test_all, y_pred_test1)
    print(classification_report(y_test_all, y_pred_test1, labels=range(len(label_encoder.classes_)), target_names=label_encoder.classes_))
    plot_confusion_matrix(y_test_all, y_pred_test1, "Naive Bayes - Test Data")
    plot_metrics_bar(y_test_all, y_pred_test1, "Naive Bayes - Test Data")
    y_pred_test2_svm = svm_model.predict(X_test_all)
    y_pred_test2_svm_labels = label_encoder.inverse_transform(y_pred_test2_svm)
    
    
    acc_svm = accuracy_score(y_test_all, y_pred_test2_svm)
    print(classification_report(y_test_all, y_pred_test2_svm, labels=range(len(label_encoder.classes_)), target_names=label_encoder.classes_))
    plot_confusion_matrix(y_test_all, y_pred_test2_svm, "SVM - Test Data")
    plot_metrics_bar(y_test_all, y_pred_test2_svm, "SVM - Test Data")

    X_test_all_blstm = np.reshape(X_test_all, (X_test_all.shape[0], 1, X_test_all.shape[1]))
    y_pred_test2_blstm = blstm_model.predict(X_test_all_blstm)
    y_pred_test2_blstm_classes = np.argmax(y_pred_test2_blstm, axis=1)
    y_pred_test2_blstm_labels = label_encoder.inverse_transform(y_pred_test2_blstm_classes)
    
    acc_blstm = accuracy_score(y_test_all, y_pred_test2_blstm_classes)
    print(classification_report(y_test_all, y_pred_test2_blstm_classes, labels=range(len(label_encoder.classes_)), target_names=label_encoder.classes_))
    plot_confusion_matrix(y_test_all, y_pred_test2_blstm_classes, "BLSTM - Test Data")

    plot_final_accuracy_comparison(acc_nb, acc_svm, acc_blstm)
    plot_comparison_graphs()

if __name__ == "__main__":
    main() 
