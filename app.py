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

def validate_data(df):
    if df is None:
        return False, "Δεν ήταν δυνατή η φόρτωση των δεδομένων."

    # Ελέγξτε αν ο πίνακας έχει τουλάχιστον 2 στήλες (F χαρακτηριστικά + 1 label)
    if df.shape[1] < 2:
        return False, "Ο πίνακας δεδομένων πρέπει να έχει τουλάχιστον δύο στήλες (F χαρακτηριστικά + 1 label)."

    # Ελέγξτε αν η τελευταία στήλη περιέχει την ετικέτα (label)
    if not pd.api.types.is_numeric_dtype(df.iloc[:, -1]) and not pd.api.types.is_string_dtype(df.iloc[:, -1]):
        return False, "Η τελευταία στήλη πρέπει να περιέχει τις ετικέτες (labels) των δειγμάτων."

    return True, "Τα δεδομένα είναι έγκυρα."

def main():
    st.title("Φόρτωση Δεδομένων")

    # Προσθέστε ένα πεδίο επιλογής αρχείου
    uploaded_file = st.file_uploader("Επιλέξτε ένα αρχείο CSV ή Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Φορτώστε τα δεδομένα αν το αρχείο είναι μη κενό
        df = load_data(uploaded_file)
        is_valid, message = validate_data(df)
        if is_valid:
            st.write("Τα δεδομένα φορτώθηκαν επιτυχώς:")
            st.write(df)
        else:
            st.error(message)

if __name__ == "__main__":
    main()
