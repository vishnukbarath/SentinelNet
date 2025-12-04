import pandas as pd

# Update path exactly to your file
csv_path = r"C:\Users\vishn\Documents\SentinelNet\data\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"

df = pd.read_csv(csv_path)
print("\n================ COLUMN NAMES ================\n")
for col in df.columns:
    print(col)

print("\n==============================================\n")
print(f"Total Columns: {len(df.columns)}")
