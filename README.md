# Syndicate Detection using Graph Neural Networks (GNN)

This project implements a sophisticated Fraud Detection System (FDS) enhanced with Graph Neural Network (GNN) techniques, tailored to identify fraudulent activities within digital ecosystems, specifically within online gambling and digital financial transactions in Indonesia.

## Overview

With the rise of complex and organized fraud syndicates in the digital ecosystem, traditional rule-based fraud detection methods fall short. This project introduces a GNN-based approach to model relationships and behaviors within the transaction network, offering a powerful tool to detect fraud syndicates by analyzing patterns, connections, and centrality measures.

### Features
- **Graph Scaling**: Handles large-scale datasets, with the capability to process bulk inputs of up to 10,000 users.
- **Node Embeddings**: Uses GNN to generate embeddings for nodes (users, transactions), which can then be clustered and analyzed.
- **Fraud Detection**: Utilizes a combination of graph analysis and machine learning to distinguish between fraudulent and non-fraudulent users.
- **Interactive Visualization**: Provides a network graph visualization with search functionality to zoom in on specific user IDs, displaying key metrics such as risk score, syndicate score, and transaction details.
- **Real-time Analysis**: Supports real-time detection with the ability to search and analyze user-specific data on demand.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/natgluons/Syndicate-Indication-using-Network-Graph-Analytics.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Syndicate-Indication-using-Network-Graph-Analytics
   ```
3. Install the necessary Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Train the Model

Before using the system for detection, you need to train the model using the provided `train_hgt_model.py` script:

```bash
python train_hgt_model.py --data_dir ./realdata --model_dir ./model
```

### 2. Perform Fraud Detection

Once the model is trained, use the `hgt_kmeans_riskscore_graph.py` script to perform fraud detection on a new set of data:

```bash
python hgt_kmeans_riskscore_graph.py --data_dir ./realdata --model_path ./model/trained_hgt_model_realdata.pth --output_dir ./result
```

### 3. Visualize the Network Graph

Use the `index.html` and `network_graph.js` files to visualize the network graph. This provides an interactive interface where you can search for specific user IDs and zoom in to view their relationships and metrics:

1. Start a local server (for example, using Python's `http.server`):
   ```bash
   python -m http.server
   ```
2. Open `index.html` in your web browser.

### 4. Search and Analyze Specific Users

- Enter the user ID in the search bar and click "Search" to zoom in on the node representing that user.
- The node will be highlighted, and a tooltip will display detailed information about the user's inbound/outbound transactions, risk score, and syndicate score.

## Data Description

- **Nodes**: Represent users or entities (e.g., merchants, customers).
- **Edges**: Represent transactions between entities.
- **Features**: Various features are extracted from nodes and edges, such as transaction amounts, account types, and other demographic attributes.

## Methodology

- **Heterogeneous Graph Transformer (HGT)**: Used to handle the complex relationships within the heterogeneous graph of transactions.
- **K-means Clustering**: Applied to cluster similar behaviors together, separating fraudulent and non-fraudulent entities.
- **Syndicate Score**: A composite score calculated using in-degree, out-degree, and betweenness centrality measures to identify potential syndicate activities.

![image](https://github.com/user-attachments/assets/0e62843c-c19f-4d5f-87cb-594ddbb45a17)
