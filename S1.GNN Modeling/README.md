## S1.GNN Modeling

This project provides a simple pipeline to:

- Build a graph from a CSV file
- Train a GNN model using the generated graph
- Export node embeddings from a trained model

### Project Structure

- `Example_data/example_data.csv`: Example input data
- `1.Build_Graph.py`: Build the graph from a CSV
- `2.Model_tuning.py`: Train/tune a model using a settings JSON
- `3.Get_embedding.py`: Export embeddings from a trained model
- `Models/`: Model architectures (GAT, GCN, GraphSAGE, Magnet)
- `Settings/Tuning/tuning_Magnet.json`: Example settings for the Magnet model

### 1) Build the Graph

Given a CSV (e.g., `Example_data/example_data.csv`), run:

```bash
python 1.Build_Graph.py /path/to/example_data.csv --k 25
```

- **k**: number of nearest neighbors used to construct the graph
- Outputs (written to the working directory unless otherwise configured):
  - `adj_matrix.pickle`
  - `output_node_features.pickle`

### 2) Train a Model

Train using the generated graph and a settings JSON. For example, to train the Magnet model:

```bash
python 2.Model_tuning.py -s Settings/Tuning/tuning_Magnet.json
```

- Training artifacts (checkpoints, logs, etc.) are written to the output directory (../temp) configured in the settings JSON

### 3) Export Embeddings

After training completes, export node embeddings with the same settings JSON:

```bash
python 3.Get_embedding.py -s Settings/Tuning/tuning_Magnet.json
```

- Embeddings will be written to the configured output directory (../temp)

### Notes

- Ensure the paths in your settings JSON (input data, graph files, output directories) are correct for your environment
- The example commands assume the repository root as the working directory
- Replace `/path/to/example_data.csv` with your actual CSV path (e.g., `Example_data/example_data.csv`)

### Troubleshooting

- If you encounter missing dependency errors, install the required Python packages indicated by the error messages
- Confirm that your `Settings/Tuning/*.json` references the files produced by the graph-building step: `adj_matrix.pickle` and `output_node_features.pickle`