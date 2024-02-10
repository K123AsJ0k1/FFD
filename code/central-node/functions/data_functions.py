from flask import current_app

def split_data():
    USED_COLUMNS = current_app.config['USED_COLUMNS']
    
    data_path = 'data/Formated_Fraud_Detection_Data.csv'
