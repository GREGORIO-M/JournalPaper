import os
import pandas as pd
from datetime import datetime
import time

DEVICE_PATH = "/dev/mpdccp_acpf_data"
OUTPUT_FILE = "subflow_data.csv"
CHUNK_SIZE = 1000  # Adjust as needed

def get_subflow_info():
    try:
        with open(DEVICE_PATH, "r") as f:
            subflow_data = f.readlines()
        return subflow_data
    except FileNotFoundError:
        print(f"Error: File not found at {DEVICE_PATH}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def create_dataframe(subflow_data):
    if not subflow_data:
        return pd.DataFrame(columns=['Timestamp', 'sock', 'name', 'cwnd','in_flight', 'srtt', 'prio', 'subflow_queue', 'meta_queue'])

    data = []
    for line in subflow_data:
        parts = line.strip().split()
        if len(parts) > 2 and parts[1] == "sock":  # Ensure at least "timestamp sock ..."
            try:
                row = {'Timestamp': float(parts[0])}  # Could be more accurate
                i = 1  # Start from the element after the timestamp
                while i < len(parts) - 1:
                    key = parts[i]
                    value = parts[i + 1]

                    if key in ('cwnd','in_flight', 'srtt', 'prio', 'subflow_queue', 'meta_queue'):

    
                        try:
                            row[key] = int(value)
                        except ValueError:  # Handle cases where the value might not be an int
                            print(f"Warning: Could not convert {key} value '{value}' to int in line: {line.strip()}. Skipping.")
                            row[key] = None  # Or some other default value
                    elif key == 'name':
                        row[key] = value
                    elif key == 'sock':
                        row[key] = value
                    i += 2

                data.append(row)
            except (ValueError, IndexError) as e:
                print(f"Error parsing line: {line.strip()}. Skipping. Error: {e}")
                continue
        else:
            print(f"Skipping malformed line: {line.strip()}")

    df = pd.DataFrame(data)
    return df

def update_data():
    if not os.path.exists(OUTPUT_FILE):  # Create file with headers only once
        pd.DataFrame(columns=['Timestamp', 'sock', 'name', 'cwnd','in_flight', 'srtt', 'prio', 'subflow_queue', 'meta_queue']).to_csv(OUTPUT_FILE, mode='w', header=True, index=False)

    while True:
        subflow_info = get_subflow_info()
        df = create_dataframe(subflow_info)

        try:#This try-except is for CPF
            if not df.empty:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')  # Convert to datetime objects
    
                # Append in chunks to avoid holding large dataframes in memory
                for i in range(0, len(df), CHUNK_SIZE):
                    chunk = df[i:i + CHUNK_SIZE]
                    chunk.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
    
            time.sleep(0.000001)
        except:
            pass
def set_cwnd_frac(subflow_fracs):
    #Sets cwnd_frac for multiple subflows.
    try:
        message = ""  # Start with an empty message
        for sock_addr, new_frac in subflow_fracs.items():
            message += f"sock {sock_addr} cwnd_frac: {new_frac}\n" # Append each subflow's settings
        with open(DEVICE_PATH, "w") as f:
            f.write(message)  # Write the *complete* message at once
        print("cwnd_fracs set for subflows.")
        return True
    except FileNotFoundError:
        print(f"Device not found: {DEVICE_PATH}")
        return False
    except Exception as e:
        print(f"Error writing to device: {e}")
        return False

if __name__ == "__main__":
    update_data() # This will run in the background, continuously updating the CSV