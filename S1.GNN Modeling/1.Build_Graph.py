#!/usr/bin/env python3
"""
Build a patient-encounter directed graph from an encoded encounter table.

ONE input:
    - input_csv: CSV path with columns: PATID, ENCID, ADMIT_DATE, and feature columns.

TWO outputs (saved in the same directory as input_csv):
    - adj_matrix.pickle            # SciPy sparse adjacency (weights = year gaps)
    - output_node_features.pickle  # DataFrame of node features (index = ENCID, aligned to graph nodes)

Details:
    - KNN edges: Jaccard over feature columns (IDs/dates/outcomes/status excluded).
    - Temporal edges: connect to the *next* encounter within each patient.
    - Only positive (forward-in-time) edges are added; when multiple proposals exist,
      the smallest gap is kept.
"""

import os
import argparse
import time
import pickle

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


EXCLUDE_COLS = {
    "PATID",
    "ENCID",
    "ADMIT_DATE",
    "outcome_final_AD",
    "current_status",
    "next_status",
}


def years_between(t1, t2) -> float:
    """Return (t2 - t1) in years for numpy/pandas datetimes."""
    return (np.datetime64(t2) - np.datetime64(t1)) / np.timedelta64(1, "D") / 365.25


def build_graph(input_csv: str, top_k: int = 200) -> None:
    # ---- I/O setup ----
    input_csv = os.path.abspath(input_csv)
    out_dir = os.path.dirname(input_csv) or "."
    adj_out = os.path.join(out_dir, "adj_matrix.pickle")
    feats_out = os.path.join(out_dir, "output_node_features.pickle")

    # ---- Load & basic checks ----
    df = pd.read_csv(input_csv)
    required = {"PATID", "ENCID", "ADMIT_DATE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Normalize date dtype
    df["ADMIT_DATE"] = pd.to_datetime(df["ADMIT_DATE"], errors="coerce")
    if df["ADMIT_DATE"].isna().any():
        bad = df[df["ADMIT_DATE"].isna()]
        raise ValueError(
            f"ADMIT_DATE contains unparseable values (rows: {bad.index.tolist()[:5]} ...)."
        )

    # ---- Feature columns (exclude IDs, dates, and specified outcomes/status) ----
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    if len(feature_cols) == 0:
        raise ValueError(
            "No feature columns found after excluding "
            f"{sorted(EXCLUDE_COLS)}. Please provide feature columns."
        )

    # ---- Fit KNN on feature matrix using Jaccard ----
    X = df[feature_cols].to_numpy()
    # Use k up to the dataset size; KNN will include self, we'll skip it later.
    k_eff = min(top_k, len(df))
    start = time.time()
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric="jaccard")
    nbrs.fit(X)
    _, indices = nbrs.kneighbors(X)
    elapsed = time.time() - start
    print(f"[KNN] Computed neighbors in {elapsed:.2f}s (metric=jaccard, k={k_eff}).")

    # ---- Prep lookup helpers ----
    meta = df[["PATID", "ENCID", "ADMIT_DATE"]].copy()
    encids = meta["ENCID"].to_numpy()
    admit_dates = meta["ADMIT_DATE"].to_numpy()
    enc_to_row = pd.Series(meta.index.values, index=meta["ENCID"]).to_dict()

    # ---- Build graph ----
    G = nx.DiGraph()
    # Add nodes
    for _, row in meta.iterrows():
        G.add_node(
            row["ENCID"],
            admit_date=row["ADMIT_DATE"].isoformat(),
            patid=row["PATID"],
        )

    # ---- Add KNN edges (positive time gap only) ----
    print("[Graph] Inserting KNN edges...")
    for idx in tqdm(range(len(df)), unit="node"):
        src_enc = encids[idx]
        src_dt = admit_dates[idx]

        nbr_idx_list = indices[idx]
        nbr_enc_list = encids[nbr_idx_list]

        for dst_enc in nbr_enc_list:
            if dst_enc == src_enc:
                continue  # skip self
            dst_dt = admit_dates[enc_to_row[dst_enc]]
            gap_years = years_between(src_dt, dst_dt)
            if gap_years > 0:
                if G.has_edge(src_enc, dst_enc):
                    if gap_years < G[src_enc][dst_enc]["weight"]:
                        G[src_enc][dst_enc]["weight"] = gap_years
                else:
                    G.add_edge(src_enc, dst_enc, weight=gap_years)

    # ---- Add next-encounter edges within patient ----
    print("[Graph] Inserting per-patient next-encounter edges...")
    for _, sub in tqdm(df.sort_values("ADMIT_DATE").groupby("PATID"), unit="patient"):
        rows = sub[["ENCID", "ADMIT_DATE"]].sort_values("ADMIT_DATE").to_numpy()
        for i in range(len(rows) - 1):
            src_enc, src_dt = rows[i]
            dst_enc, dst_dt = rows[i + 1]
            gap_years = years_between(src_dt, dst_dt)
            if gap_years > 0:
                if G.has_edge(src_enc, dst_enc):
                    if gap_years < G[src_enc][dst_enc]["weight"]:
                        G[src_enc][dst_enc]["weight"] = gap_years
                else:
                    G.add_edge(src_enc, dst_enc, weight=gap_years)

    print(f"[Graph] Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

    # ---- Outputs ----
    node_order = list(G.nodes())
    adj = nx.adjacency_matrix(G, nodelist=node_order, weight="weight")
    node_features = df.set_index("ENCID").reindex(node_order)

    with open(adj_out, "wb") as f:
        pickle.dump(adj, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(feats_out, "wb") as f:
        pickle.dump(node_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[Done] Saved adjacency to: {adj_out}")
    print(f"[Done] Saved node features to: {feats_out}")


def main():
    ap = argparse.ArgumentParser(description="Build encounter graph with KNN + next-encounter edges.")
    ap.add_argument("input_csv", help="Path to CSV with PATID, ENCID, ADMIT_DATE, feature columns.")
    ap.add_argument("--k", type=int, default=200, help="Top-K neighbors for KNN (default: 200).")
    args = ap.parse_args()
    build_graph(args.input_csv, top_k=args.k)


if __name__ == "__main__":
    main()

