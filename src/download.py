"""Fetch the NESO historic demand CSVs (not committed to git)."""
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
BASE = "https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/resource"
RESOURCES = {
    2019: "dd9de980-d724-415a-b344-d8ae11321432",
    2020: "33ba6857-2a55-479f-9308-e5c4c53d4381",
    2021: "18c69c42-f20d-46f0-84e9-e279045befc6",
    2022: "bb44a1b5-75b1-4db2-8491-257f23385006",
    2023: "bf5ab335-9b40-4ea4-b93a-ab4af7bce003",
    2024: "f6d02c0f-957b-48cb-82ee-09003f2ba759",
}

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for year, rid in RESOURCES.items():
        dest = RAW / f"demanddata_{year}.csv"
        url = f"{BASE}/{rid}/download/demanddata_{year}.csv"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r:
            body = r.read()
        # a redirect that lands on an error page returns a small HTML blob, not a
        # CSV; fail loudly here instead of letting the parser guess at it later
        if len(body) < 500_000 or not body.lstrip()[:1].isalpha():
            raise RuntimeError(f"{year}: expected a CSV, got {len(body):,} bytes")
        dest.write_bytes(body)
        print(f"{year}: {dest.stat().st_size:,} bytes")
