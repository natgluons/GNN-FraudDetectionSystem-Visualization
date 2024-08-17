import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch_geometric.nn import MessagePassing
import networkx as nx
import json

# Custom HGT Layer (Handling edge features as well)
class CustomHGTLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomHGTLayer, self).__init__(aggr='mean')  # "Mean" aggregation.
        self.linear = torch.nn.Linear(in_channels + 1, out_channels)  # Add 1 for edge attributes.

    def forward(self, x, edge_index, edge_attr):
        # Propagate the message passing
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # Combine node features with edge features
        return F.relu(self.linear(torch.cat([x_j, edge_attr.unsqueeze(-1)], dim=-1)))

# Custom HGT Model (as defined in the pre-trained model)
class CustomHGTModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_node_types):
        super(CustomHGTModel, self).__init__()
        self.node_type_embedding = torch.nn.Embedding(num_node_types, hidden_channels)

        self.layer1 = CustomHGTLayer(in_channels, hidden_channels)
        self.layer2 = CustomHGTLayer(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.layer1(x, edge_index, edge_attr))
        x = F.relu(self.layer2(x, edge_index, edge_attr))
        return x

# Function to load and preprocess data
def load_data(node_path, edge_path):
    nodes = pd.read_csv(node_path)
    edges = pd.read_csv(edge_path)

    # Ensure IDs are consistent and strip any whitespace
    nodes['user_id'] = nodes['user_id'].astype(str).str.strip()
    edges['source_id'] = edges['source_id'].astype(str).str.strip()
    edges['target_id'] = edges['target_id'].astype(str).str.strip()

    return nodes, edges

def preprocess(nodes, edges):
    # Convert categorical columns to strings for encoding
    nodes = nodes.astype(str)
    edges = edges.astype(str)
    
    return nodes, edges

def encode_data(nodes, edges):
    # Create mapping from user_id to a numeric index
    user_id_encoder = LabelEncoder()
    
    # Explicitly fit the encoder on the node IDs
    user_id_encoder.fit(nodes['user_id'])

    # Encode node IDs
    nodes['user_idx'] = user_id_encoder.transform(nodes['user_id'])

    # Map edges to node indices using the same encoder
    edges['source_idx'] = user_id_encoder.transform(edges['source_id'])
    edges['target_idx'] = user_id_encoder.transform(edges['target_id'])

    # Encode other categorical variables in nodes and edges
    encoders = {}
    for col in nodes.columns:
        if col != 'user_id' and nodes[col].dtype == 'object':
            encoder = LabelEncoder()
            nodes[col] = encoder.fit_transform(nodes[col])
            encoders[col] = encoder

    for col in edges.columns:
        if col not in ['source_id', 'target_id', 'source_idx', 'target_idx'] and edges[col].dtype == 'object':
            encoder = LabelEncoder()
            edges[col] = encoder.fit_transform(edges[col])
            encoders[col] = encoder

    return nodes, edges, encoders

def count_and_merge_transactions(nodes, edges):
    # Drop existing outbound_trx and inbound_trx columns if they exist
    nodes = nodes.drop(columns=['outbound_trx', 'inbound_trx'], errors='ignore')
    
    # Count outbound transactions for each user (source)
    outbound_counts = edges['source_id'].value_counts().rename('outbound_trx')
    # Count inbound transactions for each user (target)
    inbound_counts = edges['target_id'].value_counts().rename('inbound_trx')

    # Merge these counts back into the nodes DataFrame
    nodes = nodes.set_index('user_id')
    nodes = nodes.join(outbound_counts, on='user_id')
    nodes = nodes.join(inbound_counts, on='user_id')

    # Fill NaN values with 0 and convert to integer
    nodes['outbound_trx'] = nodes['outbound_trx'].fillna(0).astype(int)
    nodes['inbound_trx'] = nodes['inbound_trx'].fillna(0).astype(int)
    
    return nodes.reset_index()

def calculate_centrality_scores(edges):
    # Create a directed graph from edges
    G = nx.DiGraph()
    G.add_edges_from(zip(edges['source_id'], edges['target_id']))

    # Compute betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G)
    betweenness_centrality = {node: score*1e9 for node, score in betweenness_centrality.items()}

    # Min-Max Scaling betweenness centrality
    min_betweenness = min(betweenness_centrality.values(), default=0)
    max_betweenness = max(betweenness_centrality.values(), default=1)
    range_betweenness = max_betweenness - min_betweenness
    betweenness_centrality = {node: (score - min_betweenness) / range_betweenness
                            for node, score in betweenness_centrality.items()}

    # Compute in-degree and out-degree centralities
    in_degree_centrality = dict(G.in_degree(weight='weight'))
    out_degree_centrality = dict(G.out_degree(weight='weight'))

    # Calculate syndicate score
    syndicate_score = {}
    for node in G.nodes():
        bc = betweenness_centrality.get(node, 0)
        idc = in_degree_centrality.get(node, 0)
        odc = out_degree_centrality.get(node, 0)
        syndicate_score[node] = 0.2 * bc + 0.4 * idc + 0.4 * odc

    # Min-Max Scaling
    # min_syndicate_score = min(syndicate_score.values(), default=0)
    # max_syndicate_score = max(syndicate_score.values(), default=1)
    # range_syndicate_score = max_syndicate_score - min_syndicate_score

    # syndicate_score = {node: (score - min_syndicate_score) / range_syndicate_score
    #                 for node, score in syndicate_score.items()}
    
    # Z-Score Normalization 
    mean_syndicate_score = np.mean(list(syndicate_score.values()))
    std_syndicate_score = np.std(list(syndicate_score.values()))

    standardized_syndicate_score = {node: (score - mean_syndicate_score) / std_syndicate_score
                                    for node, score in syndicate_score.items()}

    # Sigmoid Transformation
    syndicate_score = {node: 1 / (1 + np.exp(-score))
                    for node, score in standardized_syndicate_score.items()}

    centrality_df = pd.DataFrame({
        'user_id': list(G.nodes()),
        'betweenness_centrality': [betweenness_centrality.get(node, 0) for node in G.nodes()],
        'syndicate_score': [syndicate_score.get(node, 0) for node in G.nodes()]
    })

    return G, centrality_df

def cluster_embeddings(embeddings, n_clusters):
    # If the number of embeddings is less than the number of clusters, reduce n_clusters
    if len(embeddings) < n_clusters:
        print(f"Warning: Reducing number of clusters to {len(embeddings)} due to insufficient samples.")
        n_clusters = len(embeddings)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(embeddings)
    
    # Return the cluster centers and labels
    return kmeans.cluster_centers_, kmeans.labels_

def classify_new_data(embeddings, all_clusters):
    # Calculate cosine similarity of each embedding to each cluster
    similarities = cosine_similarity(embeddings, all_clusters)

    # Determine the most similar cluster for each embedding
    closest_clusters = np.argmax(similarities, axis=1)

    # Map the indices to cluster types (e.g., 0-3 for fraud, 4-7 for non-fraud)
    cluster_labels = ['fraud_type1', 'fraud_type2', 'fraud_type3', 'fraud_type4',
                      'nonfraud_type1', 'nonfraud_type2', 'nonfraud_type3', 'nonfraud_type4']
    
    classified_types = [cluster_labels[idx] for idx in closest_clusters]

    return classified_types

def calculate_risk_scores(embeddings, fraud_clusters, nonfraud_clusters):
    risk_scores = []
    for embedding in embeddings:
        # Calculate similarity to fraud clusters
        fraud_similarity = np.mean(cosine_similarity([embedding], fraud_clusters))
        
        # Calculate similarity to non-fraud clusters
        nonfraud_similarity = np.mean(cosine_similarity([embedding], nonfraud_clusters))
        
        # Risk score could be the ratio or difference between these similarities
        risk_score = fraud_similarity / (fraud_similarity + nonfraud_similarity + 1e-5)
        risk_scores.append(risk_score)
    
    return risk_scores

def print_grouped_cluster_assignments(labels, user_ids, cluster_type):
    cluster_assignments = pd.DataFrame({
        f'{cluster_type}_cluster': labels,
        'user_id': user_ids
    })
    
    # Group by cluster and aggregate user_ids
    grouped_assignments = cluster_assignments.groupby(f'{cluster_type}_cluster')['user_id'].apply(lambda x: ','.join(x)).reset_index()
    
    # Add cluster names
    grouped_assignments['cluster_name'] = grouped_assignments[f'{cluster_type}_cluster'].apply(lambda x: f'{cluster_type}_type{x+1}')
    
    print(f"\n{cluster_type.capitalize()} cluster assignments:")
    print(grouped_assignments[['cluster_name', 'user_id']])

def generate_json_for_d3(G, embeddings, graph_result_path, nodes_data):
    # for i in G.nodes():
    #     print(i)

    # for source, target in G.edges():
    #     print(source, target)

    
    def get_val(i, col):
        if i in nodes_data['user_id'].values:
            return nodes_data.loc[nodes_data['user_id'] == i, col].values[0]
        else:
            return None

    # Create nodes list with ID and group (optional: group could be from clustering)
    nodes = [
        {
            "id": str(i),
            "group": 1,
            "inbound_trx_count": int(nodes_data.loc[nodes_data['user_id'] == i, 'inbound_trx'].values[0] if i in nodes_data['user_id'].values else 0),
            "outbound_trx_count": int(nodes_data.loc[nodes_data['user_id'] == i, 'outbound_trx'].values[0] if i in nodes_data['user_id'].values else 0),
            "risk_score": float(get_val(i, 'risk_score')),
            "classified_type": get_val(i, 'classified_type'),
            "betweenness_centrality": float(get_val(i, 'betweenness_centrality')),
            "syndicate_score": float(get_val(i, 'syndicate_score')),
            # "lda_embeddings_x": float(get_val(i, 'lda_embeddings_x')),
            # "lda_embeddings_y": float(get_val(i, 'lda_embeddings_y')),
        }
        for i in G.nodes()]
    # Modify 'group' as needed
    
    # Create links list
    links = [
        {
            "source": str(source),
            "target": str(target),
            "value": 1,
        }
        for source, target in G.edges()]
    
    # Create JSON structure
    graph_json = {
        "nodes": nodes,
        "links": links
    }
    
    # Save JSON to file
    with open(graph_result_path, 'w') as f:
        json.dump(graph_json, f) 

def main():
    # Paths to the data

    ## training data and model
    # node_path = './dummydata/training_nodes_dummydata.csv'
    # edge_path = './dummydata/training_edges_dummydata.csv'
    # model_path = './model/trained_hgt_model_dummydata.pth'
    node_path = './realdata/training_nodes_realdata.csv'
    edge_path = './realdata/training_edges_realdata.csv'
    model_path = './model/trained_hgt_model_realdata.pth'

    ## input data
    # new_data_path_nodes = './dummydata/input_nodes_dummydata.csv'
    # new_data_path_edges = './dummydata/input_edges_dummydata.csv'
    new_data_path_nodes = './realdata/input_nodes_realdata4.csv'
    new_data_path_edges = './realdata/input_edges_realdata4.csv'

    ## output data
    # result_file_path = './result/gnn_result_dummydata.csv'
    # graph_result_path = './result/json_for_d3_graph_dummydata.json'
    result_file_path = './result/gnn_result_realdata.csv'
    graph_result_path = './result/json_for_d3_graph_realdata.json'

    # Load and preprocess the data
    nodes, edges = load_data(node_path, edge_path)
    nodes, edges = preprocess(nodes, edges)
    
    try:
        nodes, edges, encoders = encode_data(nodes, edges)
    except ValueError as e:
        print(f"Error during encoding: {e}")
        return

    # Prepare data for PyTorch Geometric
    node_features = torch.tensor(nodes.drop(columns=['user_id', 'user_idx']).values, dtype=torch.float)
    edge_index = torch.tensor([edges['source_idx'].values, edges['target_idx'].values], dtype=torch.long)
    edge_attr = torch.tensor(edges['trans_amount'].values, dtype=torch.float)  # Assuming 'trans_amount' as edge attribute

    # Assuming 'reported_risk' as the target
    y = torch.tensor(nodes['reported_risk'].values, dtype=torch.float).view(-1, 1)

    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )

    # Initialize the custom HGT model
    num_node_types = 1  # Assuming all nodes are of a single type
    model = CustomHGTModel(in_channels=node_features.shape[1], hidden_channels=32, out_channels=64, num_node_types=num_node_types)

    # Load the pre-trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Generate embeddings for all users
    embeddings = model(data.x, data.edge_index, data.edge_attr).detach().numpy()

    # Cluster fraud cases into 4 types
    fraud_embeddings = embeddings[nodes['reported_risk'].values == 1]
    fraud_clusters, fraud_labels = cluster_embeddings(fraud_embeddings, n_clusters=4)

    # Cluster non-fraud cases into 4 types
    nonfraud_embeddings = embeddings[nodes['reported_risk'].values == 0]
    nonfraud_clusters, nonfraud_labels = cluster_embeddings(nonfraud_embeddings, n_clusters=4)

    # Check for similarity between fraud and non-fraud clusters
    for i, fraud_cluster in enumerate(fraud_clusters):
        for j, nonfraud_cluster in enumerate(nonfraud_clusters):
            similarity = cosine_similarity([fraud_cluster], [nonfraud_cluster])[0][0]
            if similarity > 0.9:  # Threshold can be adjusted
                print(f"\nWarning: High similarity ({similarity:.4f}) between fraud_type{i+1} and nonfraud_type{j+1}")

    # Print grouped cluster assignments for original nodes 
    print_grouped_cluster_assignments(fraud_labels, nodes[nodes['reported_risk'].values == 1]['user_id'], 'fraud')
    print_grouped_cluster_assignments(nonfraud_labels, nodes[nodes['reported_risk'].values == 0]['user_id'], 'nonfraud')

    # Combine fraud and non-fraud clusters
    all_clusters = np.vstack((fraud_clusters, nonfraud_clusters))

    # Calculate risk scores and classify types for the original nodes
    risk_scores = calculate_risk_scores(embeddings, fraud_clusters, nonfraud_clusters)
    classified_types = classify_new_data(embeddings, all_clusters)
    nodes['risk_score'] = risk_scores
    nodes['classified_type'] = classified_types

    # Process new users
    new_nodes, new_edges = load_data(new_data_path_nodes, new_data_path_edges)
    new_nodes, new_edges = preprocess(new_nodes, new_edges)

    new_nodes_encoded, _, _ = encode_data(new_nodes, new_edges)

    new_node_features = torch.tensor(new_nodes_encoded.drop(columns=['user_id', 'user_idx']).values, dtype=torch.float)
    new_edge_index = torch.tensor([new_edges['source_idx'].values, new_edges['target_idx'].values], dtype=torch.long)
    new_edge_attr = torch.tensor(new_edges['trans_amount'].values, dtype=torch.float)

    new_data = Data(
        x=new_node_features,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr
    )

    new_embeddings = model(new_data.x, new_data.edge_index, new_data.edge_attr).detach().numpy()

    # Calculate risk scores for the new users
    new_risk_scores = calculate_risk_scores(new_embeddings, fraud_clusters, nonfraud_clusters)

    # Classify the new data based on the pre-trained model
    new_classified_types = classify_new_data(new_embeddings, np.vstack((fraud_clusters, nonfraud_clusters)))

    # Add the risk scores and classifications to the new data DataFrame
    new_nodes['risk_score'] = new_risk_scores
    new_nodes['classified_type'] = new_classified_types

    # Count transactions and merge them into the new_nodes DataFrame
    new_nodes = count_and_merge_transactions(new_nodes, new_edges)

    # Calculate centrality scores and syndicate score, return G
    G, centrality_scores = calculate_centrality_scores(new_edges)

    # Merge centrality scores into new_nodes
    new_nodes = new_nodes.merge(centrality_scores, on='user_id', how='left')

    # Print the results for the new data
    print("\nNew nodes risk scores, classifications, and centrality scores:")
    print(new_nodes[['user_id', 'risk_score', 'classified_type', 'inbound_trx', 'outbound_trx', 'betweenness_centrality', 'syndicate_score']])

    # Save the results to a CSV file
    new_nodes[['user_id', 'risk_score', 'classified_type', 'inbound_trx', 'outbound_trx', 'betweenness_centrality', 'syndicate_score']].to_csv(result_file_path, index=False)
    print(f"\nUser Classification & Scoring saved to {result_file_path}")

    # Generate JSON for D3 visualization
    generate_json_for_d3(G, new_embeddings, graph_result_path, new_nodes)
    print(f"Graph JSON saved to {graph_result_path}")

if __name__ == "__main__":
    main()
