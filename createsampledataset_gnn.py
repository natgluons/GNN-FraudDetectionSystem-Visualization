import pandas as pd

# Creating sample nodes data
nodes_data = {
    'user_id': [1, 2, 3, 4, 5, 6],
    'user_type': ['customer', 'merchant', 'customer', 'merchant', 'customer', 'merchant'],
    'province': ['Jakarta', 'Bandung', 'Surabaya', 'Jakarta', 'Bandung', 'Surabaya'],
    'account_age_months': [12, 24, 6, 36, 18, 48],
    'outbound_trx': [5, 15, 2, 20, 7, 25],
    'inbound_trx': [3, 10, 5, 8, 6, 18],
    'hit_count': [1, 2, 0, 4, 1, 3],
    'reported_risk': [0.1, 0.3, 0.0, 0.5, 0.2, 0.6]
}

# Creating sample edges data
edges_data = {
    'source_id': [1, 2, 3, 4, 5, 6],
    'target_id': [2, 3, 4, 5, 6, 1],
    'trans_amount': [100, 200, 150, 250, 300, 350],
    'trans_type': ['online', 'offline', 'online', 'offline', 'online', 'offline'],
    'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06']
}

# Convert dictionaries to DataFrames
nodes_df = pd.DataFrame(nodes_data)
edges_df = pd.DataFrame(edges_data)

# Save to CSV
nodes_df.to_csv('./gnn_sampledata/nodes.csv', index=False)
edges_df.to_csv('./gnn_sampledata/edges.csv', index=False)

nodes_df, edges_df