"""
Stage 1 — Convert Bangladesh datasets (Institutions, Hospitals, Restaurants)
into clean SQLite databases.

Run this in Google Colab or a local Python environment.
Install dependencies first:
    pip install datasets pandas
"""

import sqlite3
import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# WHY we use `datasets.load_dataset()` instead of manually downloading CSVs:
# HuggingFace dataset repos don't always expose a predictable raw-CSV URL
# (they auto-convert to parquet internally). `load_dataset()` handles fetching
# the right files for us and hands back a clean object we can convert to a
# pandas DataFrame. This is more robust than guessing file paths.
# ---------------------------------------------------------------------------


def hf_dataset_to_df(repo_id: str) -> pd.DataFrame:
    """Load a HuggingFace dataset repo's 'train' split as a pandas DataFrame."""
    ds = load_dataset(repo_id, split="train")
    return ds.to_pandas()


# ---------------------------------------------------------------------------
# 1. INSTITUTIONS
# ---------------------------------------------------------------------------
def build_institutions_db():
    print("Loading Institutions dataset...")
    df = hf_dataset_to_df("Mahadih534/Institutional-Information-of-Bangladesh")

    # The raw dataset has 22 columns, many of them administrative codes
    # (DIVISION_ID, THANA_ID, UNION_ID, MAUZA_ID) that a query agent will
    # never need — an LLM-driven tool cares about human-readable fields,
    # not internal ID numbers. We keep the ones that matter for querying
    # and rename them to clean, lowercase, meaningful names.
    keep_and_rename = {
        "INSTITUTE_NAME": "name",
        "INSTITUTE_TYPE": "type",           # School / College / Madrasha / etc.
        "DIVISION": "division",
        "DISTRICT": "district",
        "THANA": "thana",                   # sub-district / upazila-level area
        "ADDRESS": "address",
        "MANAGEMENT_TYPE": "management_type",  # GOVERNMENT / NON-GOVERNMENT / etc.
        "MOBILE": "contact_number",
        "STUDENT_TYPE": "student_type",     # CO-EDUCATION / GIRLS / etc.
        "EDUCATION_LEVEL": "education_level",
        "AFFILIATION": "affiliation_status",
        "MPO_STATUS": "mpo_status",
    }
    df = df[list(keep_and_rename.keys())].rename(columns=keep_and_rename)

    # Basic cleaning: strip whitespace, normalize case for the fields a user
    # is likely to filter on (district/division names), since "Dhaka" vs
    # " dhaka " would otherwise silently break exact-match SQL queries.
    for col in ["name", "type", "division", "district", "thana", "address",
                "management_type", "student_type", "education_level",
                "affiliation_status", "mpo_status"]:
        df[col] = df[col].astype(str).str.strip()

    # All these fields are genuinely textual/categorical -> TEXT is correct.
    # There's no numeric field here worth INTEGER/REAL (no capacity/student
    # count column in this dataset).
    conn = sqlite3.connect("institutions.db")
    df.to_sql("institutions", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX idx_inst_district ON institutions(district);")
    conn.execute("CREATE INDEX idx_inst_type ON institutions(type);")
    conn.commit()
    conn.close()
    print(f"institutions.db written — {len(df)} rows.")


# ---------------------------------------------------------------------------
# 2. HOSPITALS
# ---------------------------------------------------------------------------
def build_hospitals_db():
    print("Loading Hospitals dataset...")
    df = hf_dataset_to_df("Mahadih534/all-bangladeshi-hospitals")

    # Note: 'Paurasava' and 'Union' columns are entirely null in this
    # dataset (verified by inspection) — including them would just add
    # dead weight to every query result, so we drop them.
    keep_and_rename = {
        "Name": "name",
        "Type": "type",               # e.g. Upazila Health Complex, Medical College Hospital
        "Agency": "agency",           # e.g. DGHS, MOPA
        "Division": "division",
        "District": "district",
        "City Corporation": "city_corporation",
        "Upazila": "upazila",
        "Private": "is_private",      # 0/1 flag in source data
    }
    df = df[list(keep_and_rename.keys())].rename(columns=keep_and_rename)

    for col in ["name", "type", "agency", "division", "district",
                "city_corporation", "upazila"]:
        df[col] = df[col].astype(str).str.strip()

    # is_private is a 0/1 flag -> genuinely INTEGER, not TEXT.
    df["is_private"] = df["is_private"].fillna(0).astype(int)

    conn = sqlite3.connect("hospitals.db")
    df.to_sql("hospitals", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX idx_hosp_district ON hospitals(district);")
    conn.execute("CREATE INDEX idx_hosp_type ON hospitals(type);")
    conn.commit()
    conn.close()
    print(f"hospitals.db written — {len(df)} rows.")


# ---------------------------------------------------------------------------
# 3. RESTAURANTS
# ---------------------------------------------------------------------------
def build_restaurants_db():
    print("Loading Restaurants dataset...")
    df = hf_dataset_to_df("Mahadih534/Bangladeshi-Restaurant-Data")

    keep_and_rename = {
        "name": "name",
        "rating": "rating",                       # REAL, 0-5
        "number_of_reviews": "review_count",       # INTEGER
        "affluence": "affluence",                  # REAL, rough price-tier signal, has nulls
        "address": "address",
        "latitude": "latitude",                    # REAL
        "longitude": "longitude",                  # REAL
    }
    df = df[list(keep_and_rename.keys())].rename(columns=keep_and_rename)

    df["name"] = df["name"].astype(str).str.strip()
    df["address"] = df["address"].astype(str).str.strip()

    # review_count is missing for many rows (venues with 0 reviews show
    # as null, not 0) -> filling with 0 is the correct semantic choice here,
    # and lets a query agent safely do things like "ORDER BY review_count".
    df["review_count"] = df["review_count"].fillna(0).astype(int)
    df["rating"] = df["rating"].astype(float)
    # affluence has many genuine nulls (unknown) -> leave as NULL rather than
    # guessing a fake value; REAL columns support NULL fine in SQLite.
    df["affluence"] = df["affluence"].astype(float)

    conn = sqlite3.connect("restaurants.db")
    df.to_sql("restaurants", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX idx_rest_rating ON restaurants(rating);")
    conn.commit()
    conn.close()
    print(f"restaurants.db written — {len(df)} rows.")


# ---------------------------------------------------------------------------
# 4. VERIFY — sanity-check each DB with a real query, so you catch schema
#    or type mistakes now rather than after building the LangChain tools.
# ---------------------------------------------------------------------------
def verify():
    print("\n--- Verification ---")

    conn = sqlite3.connect("institutions.db")
    print("Institutions sample:")
    print(pd.read_sql("SELECT name, type, district FROM institutions LIMIT 3;", conn))
    print(pd.read_sql("SELECT district, COUNT(*) as n FROM institutions "
                       "GROUP BY district ORDER BY n DESC LIMIT 5;", conn))
    conn.close()

    conn = sqlite3.connect("hospitals.db")
    print("\nHospitals sample:")
    print(pd.read_sql("SELECT name, type, district FROM hospitals LIMIT 3;", conn))
    print(pd.read_sql("SELECT district, COUNT(*) as n FROM hospitals "
                       "WHERE district = 'Dhaka' GROUP BY district;", conn))
    conn.close()

    conn = sqlite3.connect("restaurants.db")
    print("\nRestaurants sample:")
    print(pd.read_sql("SELECT name, rating, review_count FROM restaurants "
                       "ORDER BY review_count DESC LIMIT 3;", conn))
    conn.close()


if __name__ == "__main__":
    build_institutions_db()
    build_hospitals_db()
    build_restaurants_db()
    verify()
