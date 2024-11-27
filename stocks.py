import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkcalendar import Calendar
import matplotlib.pyplot as plt

def get_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return data

def preprocess_data(data):
    data['pct_change'] = data.pct_change() * 100
    data['SMA_7'] = data['Close'].rolling(window=7).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    data['volatility'] = data['pct_change'].rolling(window=7).std()
    data['target'] = np.where(data['pct_change'] > 0, 1, 0)
    data = data.dropna()
    return data

def train_model(data):
    X = data[['Close', 'SMA_7', 'SMA_30', 'volatility']]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuratezza del modello: {accuracy * 100:.2f}%")
    return model

def simulate_future_data(data, last_date, future_date, model):
    """Simula i dati futuri dal giorno successivo all'ultimo dato reale fino alla data futura."""
    current_date = last_date + timedelta(days=1)
    future_data = []

    last_row = data.iloc[-1]
    simulated_close = last_row['Close']

    while current_date <= future_date:
        # Simulazione del prezzo futuro
        simulated_pct_change = np.random.normal(0, last_row['volatility'])  # Rumore casuale basato sulla volatilità
        simulated_close += simulated_close * (simulated_pct_change / 100)

        # Aggiungi dati simulati
        future_data.append({
            'Date': current_date,
            'Close': simulated_close,
            'SMA_7': simulated_close,  # Semplificazione
            'SMA_30': simulated_close,
            'volatility': last_row['volatility']
        })

        current_date += timedelta(days=1)

    return pd.DataFrame(future_data)

def make_prediction_and_plot():
    ticker = ticker_var.get()
    model_choice = model_var.get()
    selected_date = calendar.get_date()

    if not ticker:
        messagebox.showerror("Errore", "Seleziona un'azione per fare la previsione.")
        return

    try:
        selected_date_dt = datetime.strptime(selected_date, "%m/%d/%Y")
    except ValueError as e:
        messagebox.showerror("Errore", f"Errore nella conversione della data: {e}")
        return

    try:
        if model_choice == 'Generale':
            model = joblib.load("general_model.pkl")
        else:
            model = joblib.load(f"{ticker}_model.pkl")
    except FileNotFoundError:
        messagebox.showerror("Errore", f"Modello non trovato per {ticker}.")
        return

    try:
        data = pd.read_csv(f"{ticker}_processed.csv", parse_dates=['Date'], index_col='Date')
    except FileNotFoundError:
        messagebox.showerror("Errore", f"Non è stato trovato il file per {ticker}.")
        return

    today = datetime.now()
    if selected_date_dt > today:
        # Simula i dati futuri
        future_data = simulate_future_data(data, data.index[-1], selected_date_dt, model)
        data = pd.concat([data, future_data.set_index('Date')])

        # Previsione per l'ultimo giorno simulato
        X_future = future_data[['Close', 'SMA_7', 'SMA_30', 'volatility']].iloc[-1].values.reshape(1, -1)
        prediction = model.predict(X_future)

        result = f"Previsione per {selected_date_dt.strftime('%Y-%m-%d')}: L'azione {'aumenterà' if prediction[0] == 1 else 'diminuirà'}."
        messagebox.showinfo("Predizione", result)
    else:
        if selected_date_dt not in data.index:
            messagebox.showerror("Errore", "Data non disponibile nei dati storici.")
            return

        X = data.loc[selected_date_dt, ['Close', 'SMA_7', 'SMA_30', 'volatility']].values.reshape(1, -1)
        prediction = model.predict(X)
        result = f"Data: {selected_date_dt.strftime('%Y-%m-%d')}\nL'azione {'aumenterà' if prediction[0] == 1 else 'diminuirà'}."
        messagebox.showinfo("Predizione", result)

    # Genera il grafico mensile con la previsione inclusa
    plot_monthly_data(data, ticker, selected_date_dt)

def plot_monthly_data(data, ticker, selected_date_dt):
    # Aggiungere i dati mensili
    data['Month'] = data.index.to_period('M')
    monthly_data = data.groupby('Month').mean()

    plt.figure(figsize=(12, 6))
    plt.plot(monthly_data.index.to_timestamp(), monthly_data['Close'], marker='o', label='Prezzo Medio')
    plt.fill_between(
        monthly_data.index.to_timestamp(),
        monthly_data['Close'] - monthly_data['volatility'],
        monthly_data['Close'] + monthly_data['volatility'],
        color='skyblue', alpha=0.2, label='Volatilità'
    )

    # Converti la data selezionata in un Period per confrontarla
    selected_month = pd.Timestamp(selected_date_dt).to_period('M')
    if selected_month in monthly_data.index:
        plt.scatter(selected_month.to_timestamp(), monthly_data.loc[selected_month, 'Close'], color='red', s=100, label='Mese della previsione')

    plt.title(f"Andamento Mensile per {ticker}", fontsize=16)
    plt.xlabel("Mese", fontsize=12)
    plt.ylabel("Prezzo (€)", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Tickers di interesse
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "JPM", "V"]

# Interfaccia Grafica
root = tk.Tk()
root.title("Predizione Azioni")

ticker_label = tk.Label(root, text="Seleziona Azione:")
ticker_label.pack()

ticker_var = tk.StringVar()
ticker_menu = ttk.Combobox(root, textvariable=ticker_var, values=tickers)
ticker_menu.pack()

model_label = tk.Label(root, text="Seleziona Modello:")
model_label.pack()

model_var = tk.StringVar()
model_menu = ttk.Combobox(root, textvariable=model_var, values=["Modello singolo", "Generale"])
model_menu.pack()

date_label = tk.Label(root, text="Seleziona Data:")
date_label.pack()

calendar = Calendar(root, selectmode='day', date_pattern='mm/dd/yyyy')
calendar.pack()

predict_button = tk.Button(root, text="Fare la previsione e mostra grafico", command=make_prediction_and_plot)
predict_button.pack()

root.mainloop()
