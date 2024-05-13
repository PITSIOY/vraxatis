import pandas as pd
import streamlit as st

def load_data(file_path):
    data = None
    try:
        # Δοκιμάστε να φορτώσετε τα δεδομένα από αρχείο CSV
        data = pd.read_csv(file_path)
    except Exception as e:
        # Αν αποτύχει, δοκιμάστε να φορτώσετε τα δεδομένα από αρχείο Excel
        try:
            data = pd.read_excel(file_path, engine='openpyxl')
        except Exception as e:
            st.error(f"Σφάλμα φόρτωσης δεδομένων: {e}")
    return data

def main():
    st.title("Φόρτωση Δεδομένων")

    # Προσθέστε ένα πεδίο επιλογής αρχείου
    uploaded_file = st.file_uploader("Επιλέξτε ένα αρχείο CSV ή Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Φορτώστε τα δεδομένα αν το αρχείο είναι μη κενό
        df = load_data(uploaded_file)
        if df is not None:
            st.write("Τα δεδομένα φορτώθηκαν επιτυχώς:")
            st.write(df)
        else:
            st.error("Δεν ήταν δυνατή η φόρτωση των δεδομένων.")

if __name__ == "__main__":
    main()
