import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):

    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    
    # Convert failure types to binary (0: no failure, 1: failure)
    df['Failure'] = df['Type'].apply(lambda x: 0 if x == 'No Failure' else 1)
    
    # Feature engineering
    df['Temp_Diff'] = df['Process temperature [K]'] - df['Air temperature [K]']
    df['Torque_RPM_Ratio'] = df['Torque [Nm]'] / (df['Rotational speed [rpm]'] + 1e-6)
    
    # Rolling statistics
    for col in ['Air temperature [K]', 'Process temperature [K]', 'Torque [Nm]']:
        df[f'{col}_Rolling_Mean'] = df[col].rolling(window=5, min_periods=1).mean()
        df[f'{col}_Rolling_Std'] = df[col].rolling(window=5, min_periods=1).std()
    
    # Scale numerical features
    scaler = MinMaxScaler()
    numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 
                      'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]',
                      'Temp_Diff', 'Torque_RPM_Ratio', 
                      'Air temperature [K]_Rolling_Mean', 'Air temperature [K]_Rolling_Std',
                      'Process temperature [K]_Rolling_Mean', 'Process temperature [K]_Rolling_Std',
                      'Torque [Nm]_Rolling_Mean', 'Torque [Nm]_Rolling_Std']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, scaler

if __name__ == "__main__":
    df = load_data("data/ai4i2020.csv")
    df_processed, scaler = preprocess_data(df)
    df_processed.to_csv("data/processed_data.csv", index=False)
    print("Data preprocessing complete. Saved to data/processed_data.csv")