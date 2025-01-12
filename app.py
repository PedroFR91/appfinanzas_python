from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
import datetime  # Importar el módulo datetime
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Filtrar datos válidos con 'DATE' y '%P&L' no nulos
def clean_data(df):
    df_cleaned = df.dropna(subset=['DATE', '%P&L'])
    df_cleaned = df_cleaned.copy()
    df_cleaned.loc[:, 'DATE'] = pd.to_datetime(df_cleaned['DATE'], format='%Y-%m-%d', errors='coerce')
    df_cleaned.loc[:, '%P&L'] = pd.to_numeric(df_cleaned['%P&L'], errors='coerce') / 100
    df_cleaned.loc[:, '$P&L'] = pd.to_numeric(df_cleaned['$P&L'], errors='coerce')
    df_cleaned.loc[:, 'AC PROFIT'] = pd.to_numeric(df_cleaned['AC PROFIT'], errors='coerce')

    return df_cleaned

# Calcular métricas clave
def calculate_metrics(df):
    total_trades = len(df)
    total_tps = len(df[df['TP/SL'] == 'TP'])
    total_sls = len(df[df['TP/SL'] == 'SL'])
    total_bes = len(df[df['TP/SL'] == 'BE'])
    winrate = (total_tps / total_trades) * 100 if total_trades > 0 else 0
    positive_pnl = df[df['$P&L'] > 0]['$P&L'].sum()
    negative_pnl = abs(df[df['$P&L'] < 0]['$P&L'].sum())
    profit_factor = positive_pnl / negative_pnl if negative_pnl > 0 else 0
    return {
        "total_trades": total_trades,
        "total_tps": total_tps,
        "total_sls": total_sls,
        "total_bes": total_bes,
        "winrate": round(winrate, 2),
        "profit_factor": round(profit_factor, 2),
        "positive_pnl": round(positive_pnl, 2),
        "negative_pnl": round(negative_pnl, 2),
    }

def calculate_charts_data(df):
    # Calcular el P&L acumulado
    df_sorted = df.sort_values('DATE')
    df_sorted['cumulative_pnl'] = df_sorted['$P&L'].cumsum()

    # Obtener fechas y valores para el gráfico de P&L acumulado
    cumulative_pnl_data = {
        "dates": df_sorted['DATE'].apply(lambda x: x.strftime('%Y-%m-%d')).tolist(),
        "values": df_sorted['cumulative_pnl'].tolist(),
    }

    # Contar la distribución de TP, SL y BE para el gráfico de distribución de trades
    trade_distribution_data = {
        "tp": len(df[df['TP/SL'] == 'TP']),
        "sl": len(df[df['TP/SL'] == 'SL']),
        "be": len(df[df['TP/SL'] == 'BE']),
    }

    return {
        "cumulative_pnl": cumulative_pnl_data,
        "trade_distribution": trade_distribution_data
    }

# Análisis de rachas
def analyze_streaks(df, condition):
    streaks = (df['TP/SL'] == condition).astype(int).groupby(df['TP/SL'].ne(condition).cumsum()).cumsum()
    max_streak = streaks.max()
    if max_streak == 0:
        return None, None, None, 0
    max_streak_end_idx = streaks.idxmax()
    max_streak_start_idx = max_streak_end_idx - max_streak + 1
    start_date = df.iloc[max_streak_start_idx]['DATE']
    end_date = df.iloc[max_streak_end_idx]['DATE']
    total_pnl = df.iloc[max_streak_start_idx:max_streak_end_idx + 1]['$P&L'].sum()
    return max_streak, start_date, end_date, total_pnl

# Análisis adicionales
def analyze_day_performance(df):
    day_metrics = df.groupby('DAY').agg(
        total_pnl=pd.NamedAgg(column='$P&L', aggfunc='sum'),
        total_operations=pd.NamedAgg(column='DAY', aggfunc='count'),
        winrate=pd.NamedAgg(column='TP/SL', aggfunc=lambda x: (x == 'TP').sum() / len(x) * 100)
    )
    day_metrics['percentage_of_total_pnl'] = (abs(day_metrics['total_pnl']) / abs(day_metrics['total_pnl'].sum())) * 100
    return day_metrics.reset_index().to_dict(orient='records')

def analyze_hour_performance(df):
    df = df.copy()
    df.loc[:, 'HOUR'] = pd.to_datetime(df['OPEN'], format='%H:%M:%S', errors='coerce').dt.hour

    hour_metrics = df.groupby('HOUR').agg(
        total_pnl=pd.NamedAgg(column='$P&L', aggfunc='sum'),
        total_operations=pd.NamedAgg(column='HOUR', aggfunc='count'),
        winrate=pd.NamedAgg(column='TP/SL', aggfunc=lambda x: (x == 'TP').sum() / len(x) * 100)
    )
    hour_metrics['percentage_of_total_pnl'] = (abs(hour_metrics['total_pnl']) / abs(hour_metrics['total_pnl'].sum())) * 100
    return hour_metrics.reset_index().to_dict(orient='records')

def analyze_session_performance(df):
    session_metrics = df.groupby('SESSION').agg(
        total_pnl=pd.NamedAgg(column='$P&L', aggfunc='sum'),
        total_operations=pd.NamedAgg(column='SESSION', aggfunc='count'),
        winrate=pd.NamedAgg(column='TP/SL', aggfunc=lambda x: (x == 'TP').sum() / len(x) * 100)
    )
    return session_metrics.reset_index().to_dict(orient='records')

def analyze_assets(df):
    asset_metrics = df.groupby('ASSET').agg(
        total_operations=pd.NamedAgg(column='ASSET', aggfunc='count'),
        tp=pd.NamedAgg(column='TP/SL', aggfunc=lambda x: (x == 'TP').sum()),
        sl=pd.NamedAgg(column='TP/SL', aggfunc=lambda x: (x == 'SL').sum()),
        be=pd.NamedAgg(column='TP/SL', aggfunc=lambda x: (x == 'BE').sum()),
        winrate=pd.NamedAgg(column='TP/SL', aggfunc=lambda x: (x == 'TP').sum() / len(x) * 100)
    )
    return asset_metrics.reset_index().to_dict(orient='records')

# Serializar JSON
def convert_to_serializable(data):
    if isinstance(data, pd.Timestamp):
        return data.strftime('%Y-%m-%d')
    elif isinstance(data, datetime.time):  # Manejar objetos time
        return data.strftime('%H:%M:%S')
    elif isinstance(data, (np.integer, int)):
        return int(data)
    elif isinstance(data, (np.floating, float)):
        return float(data)
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_dict() if isinstance(data, pd.DataFrame) else data.tolist()
    elif isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(i) for i in data]
    return data

# Endpoint home
@app.route("/")
def home():
    return jsonify({"message": "Bienvenido a la API de Finanzas"}), 200

# Endpoint para procesar archivo Excel
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'userId' not in request.form:
        return jsonify({'error': 'Archivo o userId no proporcionado'}), 400

    file = request.files['file']
    user_id = request.form.get('userId')  # Obtener el userId del formulario

    if not user_id:
        return jsonify({'error': 'userId es obligatorio'}), 400

    if file.filename == '':
        return jsonify({'error': 'El archivo está vacío'}), 400

    try:
        # Leer el archivo Excel en un DataFrame
        df = pd.read_excel(file)

        # Procesar el DataFrame
        df_cleaned = clean_data(df)

        # Generar las entradas individuales
        entries = df_cleaned.to_dict(orient='records')
        entries = [
            {
                "userId": user_id,  # Añadir el userId a cada entrada
                "date": row['DATE'],
                "day": row['DAY'],
                "open": row['OPEN'],
                "close": row['CLOSE'],
                "asset": row['ASSET'],
                "session": row.get('SESSION'),
                "buySell": row.get('BUY_SELL'),
                "lots": row.get('LOTS'),
                "tpSlBe": row.get('TP/SL'),
                "pnl": row.get('$P&L'),
                "pnlPercentage": row.get('%P&L'),
                "ratio": row.get('RATIO'),
                "risk": row.get('RISK'),
                "temporalidad": row.get('TEMPORALIDAD'),
            }
            for _, row in df_cleaned.iterrows()
        ]

        # Enviar las `entries` al backend general
        try:
            backend_url = "https://wqpxtxrkme.eu-west-2.awsapprunner.com/data"
            print(f"Enviando entradas al backend: {backend_url}")
            response = requests.post(backend_url, json={"entries": entries})
            print(f"Respuesta del backend: {response.status_code}")
            if response.status_code != 201:
                return jsonify({'error': 'Error al guardar los datos', 'details': response.text}), 500
        except Exception as e:
            return jsonify({'error': 'Error al enviar datos al backend', 'details': str(e)}), 500

        return jsonify({"message": "Datos procesados correctamente"}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
