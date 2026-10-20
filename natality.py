import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import requests
from pathlib import Path

# libreria requests per effettuare il download dei file dal repository online
#https://www.nber.org/research/data/vital-statistics-natality-birth-data


DATA_DIR = Path(__file__).resolve().parent / "data"
BASE_URL = "https://data.nber.org/nvss/natality/dta/2024"


def get_dataset_path(filename):
	DATA_DIR.mkdir(exist_ok=True)
	data_path = DATA_DIR / filename
	legacy_path = Path(__file__).resolve().parent / filename

	if data_path.exists():
		print(f"File già presente: {data_path}")
		return data_path
	if legacy_path.exists():
		print(f"Uso il file già presente: {legacy_path}")
		return legacy_path

	url = f"{BASE_URL}/{filename}"
	temp_path = data_path.with_suffix(data_path.suffix + ".part")
	print(f"Scarico {filename}...")
	with requests.get(url, stream=True, timeout=(10, 120)) as response:
		response.raise_for_status()
		with temp_path.open("wb") as output:
			for chunk in response.iter_content(chunk_size=1024 * 1024):
				if chunk:
					output.write(chunk)
	temp_path.replace(data_path)
	return data_path


path_us = get_dataset_path("natality2024us.dta")
path_ps = get_dataset_path("natality2024ps.dta")

nat2024us = pd.read_stata(path_us)
nat2024ps = pd.read_stata(path_ps)

if set(nat2024us.columns) != set(nat2024ps.columns):
	raise ValueError("I due file non contengono le stesse variabili")

nat2024 = pd.concat([nat2024us, nat2024ps], ignore_index=True)

print(nat2024.head())
print(f"Righe unite: {len(nat2024)}")

