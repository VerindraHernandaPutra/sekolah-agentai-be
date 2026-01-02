import pandas as pd
import glob
import os

# Tentukan folder
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_folder = os.path.join(base_dir, "list")
output_file = os.path.join(base_dir, "combined.csv")

# Ambil semua file CSV di folder list
csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

if not csv_files:
    print("Tidak ada file CSV ditemukan di folder list/")
    exit()

dataframes = []

# Kolom yang harus diperlakukan sebagai string meskipun terlihat numerik
force_string_cols = [
    "rt", "rw", "nomorTelepon"
]

for file in csv_files:
    print(f"Membaca file: {os.path.basename(file)}")

    # Baca CSV, paksa kolom tertentu jadi string agar tidak berubah jadi float
    df = pd.read_csv(file, dtype={col: str for col in force_string_cols})

    # Hilangkan spasi berlebih di nama kolom
    df.columns = df.columns.str.strip()

    # Pastikan semua kolom yang dipaksa string tidak berubah jadi NaN atau float
    for col in force_string_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    dataframes.append(df)

# Gabungkan semua dataframe
combined_df = pd.concat(dataframes, ignore_index=True)

# Hapus baris duplikat jika ada
combined_df.drop_duplicates(inplace=True)

# Simpan hasil gabungan
combined_df.to_csv(output_file, index=False)

print(f"Proses selesai! File gabungan disimpan di: {output_file}")
