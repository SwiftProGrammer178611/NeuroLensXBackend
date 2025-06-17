from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
import warnings
import tempfile
import os
import zipfile
import certifi
from bson.binary import Binary
import pickle
from pymongo import MongoClient
import gudhi
from ripser import ripser
from sklearn.preprocessing import normalize
import io
import base64
from functools import lru_cache
import json
import math
from fastapi import FastAPI, Request
from pydantic import BaseModel

import os
from dotenv import load_dotenv

load_dotenv()


app = FastAPI(title="NeuroCartographer API")
class GitLabUpdateRequest(BaseModel):
    repo_url: str

@app.post("/trigger-gitlab-update")
def trigger_gitlab_update(req: GitLabUpdateRequest):
    # Optional: Parse repo name, trigger pipeline using GitLab API
    print(f"Triggering update for repo: {req.repo_url}")
    # This is where GitLab API integration or CI trigger would go
    return {"message": f"GitLab update simulated for {req.repo_url}"}
# Set environment variable for tokenizers parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set matplotlib backend to 'Agg' to avoid GUI-related errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')



# Add CORS middleware to allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# MongoDB connection
mongo_uri = "mongodb+srv://moinsh2008:Comcast24680@cluster0.ujwqs.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
db = client['GoogleAIInAction']
library_col = db['concept_library']
collection = db['GoogleAIInAction']

# Preloaded models
preloaded_models = {
    "bert-base-uncased": "bert-base-uncased",
    "roberta-base": "roberta-base",
    "distilbert-base-uncased": "distilbert-base-uncased",
}

# Determine device - use CPU for consistency
device = torch.device("cpu")
print(f"Using device: {device}")

# Helper function to convert numpy types to Python native types and handle invalid JSON values
def convert_to_native_python_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    Also handles non-JSON-compliant values like inf, -inf, and NaN.
    
    Args:
        obj: Any Python object that might contain numpy types or invalid JSON values
        
    Returns:
        Object with all numpy types converted to native Python types and invalid JSON values replaced
    """
    if isinstance(obj, dict):
        return {k: convert_to_native_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_python_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_python_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Handle inf, -inf, and NaN values
        value = float(obj)
        if math.isinf(value) or math.isnan(value):
            return 0.0  # Replace with 0.0 or another appropriate value
        return value
    elif isinstance(obj, np.ndarray):
        return convert_to_native_python_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, float):
        # Handle inf, -inf, and NaN values for regular Python floats too
        if math.isinf(obj) or math.isnan(obj):
            return 0.0  # Replace with 0.0 or another appropriate value
        return obj
    elif isinstance(obj, torch.Tensor):
        # Convert torch tensors to lists
        return convert_to_native_python_types(obj.cpu().detach().numpy())
    else:
        return obj

# Helper function to ensure tensors are on the same device
def ensure_tensor_on_device(tensor, target_device=device):
    """
    Ensure a tensor is on the specified device.
    
    Args:
        tensor: PyTorch tensor
        target_device: Target device (default: CPU)
        
    Returns:
        Tensor on the target device
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.to(target_device)
    return tensor

# Model loading with caching
@lru_cache(maxsize=3)  # Cache up to 3 models
def load_models(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path, output_hidden_states=True).to(device)
    model.eval()
    embed_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    kw_model = KeyBERT('all-MiniLM-L6-v2')
    rag_generator = pipeline("text-generation", model="gpt2", max_length=150, device=device if device.type != "mps" else -1)
    return tokenizer, model, embed_model, kw_model, rag_generator

# Helper functions
def clean_tokens(tokens):
    junk_tokens = {
        "[CLS]", "[SEP]", "sep", "cls",
        ",", ".", "!", "?", ";", ":", "`", "''", "'", '"', "-", "(", ")", "[PAD]", "[UNK]"
    }
    filtered = [t for t in tokens if t.lower() not in junk_tokens and not t.startswith("##")]
    seen = set()
    unique_tokens = []
    for t in filtered:
        if t not in seen:
            unique_tokens.append(t)
            seen.add(t)
    return unique_tokens

def extract_topological_features(persistence_diagrams, persistence_pairs, threshold=0.1):
    """
    Extract quantitative topological features from persistence diagrams.
    Implements the "shape signatures" mentioned in the project outline.
    """
    features = {}
    
    # Process persistence diagrams from Ripser
    all_births = []
    all_deaths = []
    all_lifetimes = []
    
    for dim, diagram in enumerate(persistence_diagrams):
        if len(diagram) == 0:
            continue
            
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        lifetimes = deaths - births
        
        # Filter by persistence threshold
        significant = lifetimes > threshold
        if np.any(significant):
            all_births.extend(births[significant])
            all_deaths.extend(deaths[significant])
            all_lifetimes.extend(lifetimes[significant])
    
    # Compute topological summary statistics
    if len(all_lifetimes) > 0:
        features['total_persistence'] = np.sum(all_lifetimes)
        features['max_persistence'] = np.max(all_lifetimes)
        features['mean_persistence'] = np.mean(all_lifetimes)
        features['persistence_entropy'] = compute_persistence_entropy(all_lifetimes)
        features['n_significant_features'] = len(all_lifetimes)
        features['betti_numbers'] = compute_betti_numbers(persistence_diagrams, threshold)
    else:
        # Default values if no significant features found
        features['total_persistence'] = 0.0
        features['max_persistence'] = 0.0
        features['mean_persistence'] = 0.0
        features['persistence_entropy'] = 0.0
        features['n_significant_features'] = 0
        features['betti_numbers'] = [0, 0, 0, 0]
    
    return features

def compute_persistence_entropy(lifetimes):
    """
    Compute persistence entropy from lifetimes.
    """
    if len(lifetimes) == 0:
        return 0.0
    
    total = np.sum(lifetimes)
    if total == 0:
        return 0.0
    
    probs = lifetimes / total
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return entropy

def compute_betti_numbers(persistence_diagrams, threshold=0.1):
    """
    Compute Betti numbers from persistence diagrams.
    """
    betti = []
    for dim, diagram in enumerate(persistence_diagrams):
        if len(diagram) == 0:
            betti.append(0)
            continue
            
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        lifetimes = deaths - births
        
        # Count significant features
        significant = lifetimes > threshold
        betti.append(np.sum(significant))
    
    # Ensure we have at least 4 dimensions
    while len(betti) < 4:
        betti.append(0)
    
    return betti[:4]

def compute_tda_feature_vectors(activations, persistence_diagrams, topological_features):
    """
    Compute TDA feature vectors for each activation.
    """
    n_samples = len(activations)
    
    # Create a basic feature vector with topological statistics
    tda_features = np.zeros((n_samples, 4))
    
    # Fill with basic topological features
    for i in range(n_samples):
        tda_features[i, 0] = topological_features.get('total_persistence', 0.0)
        tda_features[i, 1] = topological_features.get('max_persistence', 0.0)
        tda_features[i, 2] = topological_features.get('persistence_entropy', 0.0)
        tda_features[i, 3] = topological_features.get('n_significant_features', 0)
    
    return tda_features

def combine_features_with_tda(original_features, tda_features, tda_weight=0.3):
    """
    Combine original neural activations with TDA-derived features.
    This creates the "enhanced feature space" mentioned in the project outline.
    """
    from sklearn.preprocessing import StandardScaler

    # Sanitize TDA features
    tda_features = np.nan_to_num(tda_features, nan=0.0, posinf=1e6, neginf=-1e6)

    scaler_orig = StandardScaler()
    orig_normalized = scaler_orig.fit_transform(original_features)

    scaler_tda = StandardScaler()
    tda_normalized = scaler_tda.fit_transform(tda_features)

    # Ensure dimensions match by zero-padding the TDA part if needed
    padded_tda = np.hstack([
        tda_normalized,
        np.zeros((len(tda_normalized), orig_normalized.shape[1] - tda_normalized.shape[1]))
    ])[:, :orig_normalized.shape[1]]

    # Combine with weight
    combined = (1 - tda_weight) * orig_normalized + tda_weight * padded_tda

    return combined

def perform_topology_clustering(enhanced_features, n_clusters, distance_matrix):
    """
    Perform clustering that incorporates topological structure.
    Uses both geometric and topological information for robust clustering.
    """
    # Method 1: Enhanced KMeans on TDA features
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(enhanced_features)
    
    # Method 2: Hierarchical clustering with topological distances (for validation)
    try:
        linkage_matrix = linkage(distance_matrix, method='ward')
        hierarchical_clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
        
        # Combine both methods for robustness
        # If they largely agree, use KMeans; otherwise, prefer hierarchical
        agreement = np.mean(cluster_ids == hierarchical_clusters)
        if agreement < 0.7:
            print(f"Low agreement ({agreement:.2f}) between clustering methods, using hierarchical")
            cluster_ids = hierarchical_clusters
    except:
        print("Hierarchical clustering failed, using KMeans only")
    
    return cluster_ids

def enhance_cluster_analysis_with_tda(activations, tokens, positions, n_clusters=5, 
                                    max_dimension=1, max_edge_length=2.0, 
                                    tda_weight=0.3, persistence_threshold=0.1):
    """
    Enhanced clustering using Topological Data Analysis (TDA).
    
    Args:
        activations: Neural activation vectors (n_samples, n_features)
        tokens: List of token strings
        positions: List of (layer, token) position tuples
        n_clusters: Number of clusters for final grouping
        max_dimension: Maximum homology dimension to compute
        max_edge_length: Maximum edge length for Vietoris-Rips complex
        tda_weight: Weight for combining TDA features with original features
        persistence_threshold: Minimum persistence for significant features
    
    Returns:
        cluster_ids: Array of cluster assignments
        tda_results: Dictionary with persistence pairs and topological features
        viz_data: Dictionary with enhanced features and reduced dimensions
    """
    
    print(f"Starting TDA analysis on {len(activations)} neural activations...")
    
    # Step 1: Compute distance matrix for topological analysis
    print("Computing distance matrix...")
    distances = pdist(activations, metric='euclidean')
    distance_matrix = squareform(distances)
    
    # Step 2: Compute persistent homology using Ripser
    print("Computing persistent homology...")
    try:
        # Use Ripser for persistent homology computation
        rips_result = ripser(activations, maxdim=max_dimension, thresh=max_edge_length)
        persistence_diagrams = rips_result['dgms']
        
        # Also compute using GUDHI for additional topological features
        rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge_length)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
        persistence_pairs = simplex_tree.persistence()
        
    except Exception as e:
        print(f"Warning: TDA computation failed ({e}), using fallback method")
        # Fallback: create mock persistence data
        persistence_diagrams = [np.array([[0, 0.1], [0, 0.2]]), np.array([[0.1, 0.3]])]
        persistence_pairs = [(0, (0, 0.1)), (0, (0, 0.2)), (1, (0.1, 0.3))]
    
    # Step 3: Extract topological features
    print("Extracting topological shape signatures...")
    topological_features = extract_topological_features(persistence_diagrams, persistence_pairs, persistence_threshold)
    
    # Step 4: Create TDA-enhanced feature vectors
    print("Generating TDA-enhanced features...")
    tda_features = compute_tda_feature_vectors(activations, persistence_diagrams, topological_features)
    
    # Step 5: Combine original activations with TDA features
    enhanced_features = combine_features_with_tda(activations, tda_features, tda_weight)
    
    # Step 6: Perform topology-aware clustering
    print("Performing topology-aware clustering...")
    cluster_ids = perform_topology_clustering(enhanced_features, n_clusters, distance_matrix)
    
    # Step 7: Dimensionality reduction for visualization
    print("Computing visualization embeddings...")
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(enhanced_features)
    
    # Prepare results
    tda_results = {
        'persistence_diagrams': persistence_diagrams,
        'persistence_pairs': persistence_pairs,
        'topological_features': topological_features,
        'distance_matrix': distance_matrix
    }
    
    viz_data = {
        'enhanced_features': enhanced_features,
        'reduced_features': reduced_features,
        'tda_features': tda_features
    }
    
    print(f"TDA analysis complete! Found {len(set(cluster_ids))} topology-based clusters")
    return cluster_ids, tda_results, viz_data

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for frontend display"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

# Pydantic models for request/response validation
class ConceptLibraryRequest(BaseModel):
    concepts: List[str]

class ConceptLibraryResponse(BaseModel):
    concepts: List[str]
    message: str = "Concept library retrieved successfully"

class AnalyzeQueryRequest(BaseModel):
    query: str
    selected_cavs: List[str]
    max_nodes_graph: int = 200
    graph_strategy: str = "Top by Activation"
    tda_weight: float = 0.3
    num_clusters: int = 5

class RagGenerateRequest(BaseModel):
    query: str
    rag_input: str

class BiasAnalysisRequest(BaseModel):
    query: str
    selected_cavs: List[str]
    selected_biases: List[str]
    num_clusters: int = 5

class BiasScoringRequest(BaseModel):
    selected_cavs: List[str]
    num_samples: int = 100

# Endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to NeuroCartographer API"}

@app.get("/concept-library", response_model=ConceptLibraryResponse)
def get_concept_library():
    """Get the saved concept library"""
    lib_doc = library_col.find_one({"_id": "default"})
    saved_library = lib_doc["concepts"] if lib_doc and "concepts" in lib_doc else []
    return {"concepts": saved_library}

@app.post("/concept-library", response_model=ConceptLibraryResponse)
def save_concept_library(request: ConceptLibraryRequest):
    """Save the concept library"""
    # Clean and deduplicate concepts
    new_lib = list(set([c.strip() for c in request.concepts if c.strip()]))
    
    # Write back to MongoDB
    library_col.update_one(
        {"_id": "default"},
        {"$set": {"concepts": new_lib}},
        upsert=True
    )
    
    return {"concepts": new_lib, "message": "Concept library saved successfully"}

@app.post("/analyze-query")
def analyze_query(request: AnalyzeQueryRequest):
    """Analyze a query with the selected model and concepts"""
    query = request.query or "She is a doctor."
    selected_cavs = request.selected_cavs or ["gender", "profession", "bias", "doctor", "nurse"]
    max_nodes = request.max_nodes_graph or 50
    strategy = request.graph_strategy or "modularity"
    tda_weight = request.tda_weight if request.tda_weight is not None else 0.5
    k = request.num_clusters or 5

    
    # Validate inputs
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not selected_cavs:
        raise HTTPException(status_code=400, detail="At least one concept must be selected")
    
    # Load model (using default bert-base-uncased for now)
    model_path = preloaded_models["bert-base-uncased"]
    tokenizer, model, embed_model, kw_model, rag_generator = load_models(model_path)
    
    # Process query through model
    inputs = tokenizer(query, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)
    hidden_states = output.hidden_states
    all_layers = torch.stack(hidden_states).squeeze(1)
    num_layers, seq_len, dim = all_layers.shape
    combined = all_layers.reshape(-1, dim)
    positions = [(layer, token) for layer in range(num_layers) for token in range(seq_len)]
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # TDA-Enhanced Clustering
    cluster_ids, tda_results, viz_data = enhance_cluster_analysis_with_tda(
        combined.cpu().numpy(), tokens, positions, n_clusters=k, tda_weight=tda_weight
    )
    
    # Extract visualization components
    reduced = viz_data['reduced_features']
    enhanced_features = viz_data['enhanced_features']
    
    # Map tokens to clusters
    cluster_token_map = {i: [] for i in range(k)}
    for i, cid in enumerate(cluster_ids):
        token_idx = positions[i][1]
        cluster_token_map[cid].append(tokens[token_idx])
    
    # Generate cluster labels and sentences
    clean_cluster_labels_text = {}
    clean_cluster_sentences = {}
    for cid, toks in cluster_token_map.items():
        cleaned_tokens = clean_tokens(toks)
        if len(cleaned_tokens) == 0:
            label = "(no meaningful tokens)"
            text = ""
        else:
            text = " ".join(cleaned_tokens)
            keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=3)
            label = ", ".join([kw[0] for kw in keywords]) if keywords else "No keywords"
        clean_cluster_labels_text[cid] = label
        clean_cluster_sentences[cid] = text
    
    # Compute CAV embeddings
    cav_embeddings = embed_model.encode(selected_cavs, convert_to_tensor=True)
    
    # Store cluster embeddings in MongoDB
    cluster_collection = db['cluster_embeddings']
    cluster_collection.drop()  # Clear existing for fresh run
    
    for cid in range(k):
        cluster_text = clean_cluster_sentences[cid]
        if not cluster_text.strip():
            continue
        # GitLab Issue for Unlabeled Clusters
        if label == "(no meaningful tokens)" and cluster_text.strip() == "":
            import requests

            GITLAB_TOKEN = os.getenv("GITLAB_TOKEN")
            GITLAB_PROJECT_ID = "YOUR_PROJECT_ID"
            GITLAB_API_URL = f"https://gitlab.com/api/v4/projects/{GITLAB_PROJECT_ID}/issues"

            headers = {
                "PRIVATE-TOKEN": GITLAB_TOKEN,
                "Content-Type": "application/json"
            }

            issue_payload = {
                "title": f"Unlabeled Cluster Detected (ID: {cid})",
                "description": "This cluster lacks meaningful tokens or labels. Please define a concept prompt to associate with it.",
                "labels": "NeuroCartographer, unlabeled-cluster"
            }

            try:
                response = requests.post(GITLAB_API_URL, json=issue_payload, headers=headers)
                if response.status_code != 201:
                    print(f"Failed to create GitLab issue: {response.text}")
            except Exception as e:
                print(f"Error creating GitLab issue: {e}")

        
        cluster_embedding = embed_model.encode(cluster_text, convert_to_numpy=True).tolist()
        
        # Prepare topological features for storage
        topo_features = {}
        if tda_results and 'topological_features' in tda_results:
            topo_features = {
                "total_persistence": float(tda_results['topological_features']['total_persistence']),
                "max_persistence": float(tda_results['topological_features']['max_persistence']),
                "n_significant_features": int(tda_results['topological_features']['n_significant_features']),
                "persistence_entropy": float(tda_results['topological_features']['persistence_entropy'])
            }
        
        cluster_doc = {
            "cluster_id": int(cid),
            "label": clean_cluster_labels_text[cid],
            "text": cluster_text,
            "embedding": cluster_embedding,
            "tda_features": topo_features
        }
        
        cluster_collection.insert_one(cluster_doc)
    
    # Compute cluster-concept similarities
    cluster_cav_table = []
    for cid in range(k):
        cluster_text = clean_cluster_sentences[cid]
        if not cluster_text.strip():
            cluster_cav_table.append({
                "cluster_id": int(cid),
                "label": clean_cluster_labels_text[cid],
                "related_concept": "-",
                "similarity_score": "-"
            })
            continue
        
        # Ensure tensors are on the same device
        cluster_emb = embed_model.encode(cluster_text, convert_to_tensor=True).to(device)
        cav_embeddings_device = ensure_tensor_on_device(cav_embeddings, device)
        
        similarities = util.cos_sim(cluster_emb, cav_embeddings_device)[0].cpu().numpy()
        top_cav_idx = np.argmax(similarities)
        top_cav = selected_cavs[top_cav_idx]
        sim_score = float(similarities[top_cav_idx])
        
        cluster_cav_table.append({
            "cluster_id": int(cid),
            "label": clean_cluster_labels_text[cid],
            "related_concept": top_cav,
            "similarity_score": f"{sim_score:.2f}"
        })
    
    # Generate PCA scatter plot
    fig, ax = plt.subplots(figsize=(8,6))
    colors = plt.cm.get_cmap("tab10", k)
    for cid in range(k):
        pts = reduced[cluster_ids == cid]
        if len(pts) > 0:  # Check if there are points in this cluster
            ax.scatter(pts[:,0], pts[:,1], color=colors(cid), label=f"Cluster {cid}: {clean_cluster_labels_text[cid]}")
    ax.set_title(f"PCA + KMeans Clusters for: \"{query}\"")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    pca_plot = fig_to_base64(fig)
    
    # Generate persistence diagram
    persistence_fig = None
    if tda_results and 'persistence_pairs' in tda_results:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Separate by dimension
        dim_0_points = []
        dim_1_points = []
        
        for dim, (birth, death) in tda_results['persistence_pairs']:
            if birth != death:  # Ignore trivial features
                if dim == 0:
                    dim_0_points.append((birth, death))
                elif dim == 1:
                    dim_1_points.append((birth, death))
        
        # Plot persistence points
        if dim_0_points:
            births_0, deaths_0 = zip(*dim_0_points)
            ax.scatter(births_0, deaths_0, c='red', alpha=0.7, s=50, label='H₀ (Connected Components)')
        
        if dim_1_points:
            births_1, deaths_1 = zip(*dim_1_points)
            ax.scatter(births_1, deaths_1, c='blue', alpha=0.7, s=50, label='H₁ (Loops)')
        
        # Add diagonal line (y = x)
        max_val = max([p[1] for p in dim_0_points + dim_1_points] + [p[0] for p in dim_0_points + dim_1_points]) if dim_0_points or dim_1_points else 1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='y = x')
        
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        ax.set_title("Neural Activation Topology")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        persistence_fig = fig_to_base64(fig)
    
    # Generate layer activation chart
    activations_flat = combined.cpu().numpy()
    neuron_indices = np.arange(len(activations_flat))
    max_activation_scores = activations_flat.max(axis=1)
    
    df_neurons = pd.DataFrame({
        "neuron_index": neuron_indices,
        "layer": [pos[0] for pos in positions],
        "token_idx": [pos[1] for pos in positions],
        "token": [tokens[pos[1]] for pos in positions],
        "activation_score": max_activation_scores,
        "cluster_id": cluster_ids
    })


    
    layer_avg_activation = df_neurons.groupby("layer")["activation_score"].mean()
    highest_bias_layer = layer_avg_activation.idxmax()
    highest_bias_score = layer_avg_activation.max()
    
    # Create layer activation chart
    fig, ax = plt.subplots(figsize=(10, 6))
    layer_indices = list(layer_avg_activation.index)
    ax.bar(layer_indices, layer_avg_activation.values)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Average Activation Score")
    ax.set_title("Average Activation Score per Layer")
    layer_chart = fig_to_base64(fig)
    
    # Compute concept activation strengths across layers
    token_texts = [tokens[pos[1]] for pos in positions]
    token_embs = embed_model.encode(token_texts, convert_to_tensor=False)
    token_embs_norm = normalize(token_embs)
    cav_emb_norm = normalize(cav_embeddings.cpu().numpy())
    neuron_cav_sim_matrix = np.dot(token_embs_norm, cav_emb_norm.T)
    
    layer_similarities = {concept: [] for concept in selected_cavs}
    for layer_idx in range(num_layers):
        layer_neuron_indices = [i for i, pos in enumerate(positions) if pos[0] == layer_idx]
        if not layer_neuron_indices:
            for concept in selected_cavs:
                layer_similarities[concept].append(0.0)
            continue
        
        layer_sims = neuron_cav_sim_matrix[layer_neuron_indices, :]
        avg_sim_per_concept = np.mean(layer_sims, axis=0)
        
        for i_cav, concept in enumerate(selected_cavs):
            layer_similarities[concept].append(float(avg_sim_per_concept[i_cav]))
    
    # Create concept activation chart
    fig, ax = plt.subplots(figsize=(10,6))
    x = list(range(num_layers))
    for concept in selected_cavs:
        ax.plot(x, layer_similarities[concept], marker='o', label=concept)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Average Activation Similarity")
    ax.set_title("Concept Activation Strengths Across Layers")
    ax.legend()
    concept_chart = fig_to_base64(fig)
    
    # Create heatmap
    heatmap_data = np.zeros((seq_len, num_layers))
    for i, (layer, token) in enumerate(positions):
        heatmap_data[token, layer] = max_activation_scores[i]
    
    fig, ax = plt.subplots(figsize=(12,6))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Token")
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(tokens)
    fig.colorbar(im, ax=ax)
    heatmap = fig_to_base64(fig)
    
    # Prepare top neurons data
    top_n = 10
    top_indices = np.argsort(max_activation_scores)[-top_n:][::-1]
    top_neurons = []
    
    for rank, idx in enumerate(top_indices, 1):
        top_neurons.append({
            "rank": int(rank),
            "neuron_index": int(idx),
            "layer": int(df_neurons.loc[idx, 'layer']),
            "token": df_neurons.loc[idx, 'token'],
            "activation": float(df_neurons.loc[idx, 'activation_score']),
            "cluster_id": int(df_neurons.loc[idx, 'cluster_id'])

        })
    # === Build Interactive Graph from Real Data ===
    interactive_graph_data = {
        "nodes": [],
        "links": []
    }

    for idx, row in df_neurons.iterrows():
        interactive_graph_data["nodes"].append({
            "id": f"{row['layer']}_{row['token']}_{row['token_idx']}",
            "token": row["token"],
            "token_idx": int(row["token_idx"]),
            "layer": int(row["layer"]),
            "cluster": int(row["cluster_id"]),
            "activation": float(row["activation_score"]),
            "val": 3
        })


    # Optional: naive linking between adjacent tokens
    for i in range(len(interactive_graph_data["nodes"]) - 1):
        interactive_graph_data["links"].append({
            "source": interactive_graph_data["nodes"][i]["id"],
            "target": interactive_graph_data["nodes"][i + 1]["id"],
            "weight": 1.0
        })

    # Prepare response
    response = {
        "query": query,
        "num_layers": int(num_layers),
        "num_tokens": int(seq_len),
        "tokens": tokens,
        "cluster_cav_table": cluster_cav_table,
        "cluster_labels": {str(k): v for k, v in clean_cluster_labels_text.items()},
        "cluster_sentences": {str(k): v for k, v in clean_cluster_sentences.items()},
        "highest_bias_layer": int(highest_bias_layer),
        "highest_bias_score": float(highest_bias_score),
        "layer_similarities": {k: [float(x) for x in v] for k, v in layer_similarities.items()},
        "top_neurons": top_neurons,
        "charts": {
            "pca_plot": pca_plot,
            "persistence_diagram": persistence_fig,
            "layer_chart": layer_chart,
            "concept_chart": concept_chart,
            "heatmap": heatmap
        },
        "interactive_graph": convert_to_native_python_types(interactive_graph_data),
        "activation_table": convert_to_native_python_types(df_neurons.to_dict(orient="records")),
        "tda_features": convert_to_native_python_types(tda_results["topological_features"]) if tda_results and "topological_features" in tda_results else {}
    }
    
    # Convert all numpy types to native Python types for JSON serialization
    # and handle non-JSON-compliant values like inf, -inf, and NaN
    return convert_to_native_python_types(response)

from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel, TextGenerationModel

import os

# Ensure credentials set in your environment or service account JSON
from vertexai.preview.language_models import TextEmbeddingModel
import vertexai

vertexai.init(project="my-project-85084-for-hackathon", location="us-central1")

from fastapi import HTTPException
from vertexai.language_models import TextEmbeddingModel, TextGenerationModel

def mongodb_vector_search(embedding: list, top_k: int = 5):
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index_name",  # ← replace with your actual index name
                "queryVector": embedding,
                "path": "embedding",  # ← replace with the correct field storing cluster vectors
                "numCandidates": 100,
                "limit": top_k
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1  # ← this field must exist in your cluster documents
            }
        }
    ]
    return list(collection.aggregate(pipeline))

from vertexai.generative_models import GenerativeModel
from fastapi import HTTPException
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel

class RagRequest(BaseModel):
    query: str
    rag_input: str
from vertexai.preview.language_models import TextGenerationModel
import vertexai
from vertexai.preview.generative_models import GenerativeModel
def call_vertex_ai(prompt: str) -> str:
    # Initialize Vertex AI (run once)
    vertexai.init(
        project="my-project-85084-for-hackathon",          # ← replace with your GCP project ID
        location="us-central1"                  # or other region like "europe-west4"
    )
    

    model = GenerativeModel("gemini-2.5-pro")  # or "gemini-pro"
    response = model.generate_content(prompt)
    return response.text
def generate_recommendations_from_ai(summary_data: dict) -> list:
    vertexai.init(project="my-project-85084-for-hackathon", location="us-central1")
    model = GenerativeModel("gemini-2.5-pro")

    prompt = f"""
You are an AI debugging assistant. Given this bias and interpretability analysis:

{json.dumps(summary_data, indent=2)}

Suggest 3 actionable steps a developer could take to improve model fairness and interpretability.
Respond as a simple numbered list like:
1. ...
2. ...
3. ...
"""

    response = model.generate_content(prompt)

    # Attempt to parse numbered list instead of JSON
    text = response.text.strip()
    lines = [line.lstrip("1234567890. ").strip() for line in text.split("\n") if line.strip()]
    return [{"title": line, "description": ""} for line in lines]


@app.get("/recommendations/{query}")
async def get_recommendations(query: str):
    # Placeholder for now — later replace with dynamic `rag_input` or `analysisResults` logic
    context = """
Top activated neurons relate to the word 'doctor' and structural tokens.
Cluster 1 labeled 'gender bias', cluster 2 labeled 'profession'.
The token 'she' activates gender-sensitive neurons in deeper layers.
"""

    suggestions = generate_recommendations_from_ai(context)
    
    return {"recommendations": suggestions if suggestions else []}

@app.post("/rag-generate")
async def generate_rag(request: RagRequest):
    if not request.rag_input or not request.query:
        raise HTTPException(status_code=400, detail="Missing input for RAG.")

    prompt = f"""You are an AI interpretability assistant. Based on this analysis:

{request.rag_input}

Answer the following question:
{request.query}
"""

    # Replace with your actual Vertex AI logic
    response_text = call_vertex_ai(prompt)

    return {"generated_text": response_text}

@app.post("/bias-analysis")
def bias_analysis(request: BiasAnalysisRequest):
    """Perform bias analysis comparing two selected biases"""
    query = request.query
    selected_cavs = request.selected_cavs
    selected_biases = request.selected_biases
    num_clusters = request.num_clusters
    
    # Validate inputs
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if len(selected_biases) != 2:
        raise HTTPException(status_code=400, detail="Exactly two biases must be selected for comparison")
    
    # Load model
    model_path = preloaded_models["bert-base-uncased"]
    tokenizer, model, embed_model, kw_model, rag_generator = load_models(model_path)
    
    # Process query through model
    inputs = tokenizer(query, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs)
    hidden_states = output.hidden_states
    all_layers = torch.stack(hidden_states).squeeze(1)
    num_layers, seq_len, dim = all_layers.shape
    
    # Compute CAV embeddings - ensure all on same device
    cav_embeddings = embed_model.encode(selected_cavs, convert_to_tensor=True).to(device)
    bias_embeddings = embed_model.encode(selected_biases, convert_to_tensor=True).to(device)
    
    # Compute layer-wise bias activations
    layer_bias_activations = []
    
    for layer_idx in range(num_layers):
        layer_activations = all_layers[layer_idx].reshape(seq_len, dim)
        layer_tokens = [tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[i] for i in range(seq_len)]
        
        # Encode tokens - ensure on same device
        token_embs = embed_model.encode(layer_tokens, convert_to_tensor=True).to(device)
        
        # Compute similarities to bias concepts
        bias_similarities = []
        for bias_idx, bias in enumerate(selected_biases):
            bias_emb = bias_embeddings[bias_idx].unsqueeze(0).to(device)
            bias_sim = util.cos_sim(token_embs, bias_emb)
            bias_similarities.append(bias_sim.mean().item())
        
        # Safely compute ratio to avoid division by zero
        bias_ratio = 0.0
        if bias_similarities[1] != 0:
            bias_ratio = bias_similarities[0] / bias_similarities[1]
        
        layer_bias_activations.append({
            "layer": int(layer_idx),
            "bias1_name": selected_biases[0],
            "bias1_activation": float(bias_similarities[0]),
            "bias2_name": selected_biases[1],
            "bias2_activation": float(bias_similarities[1]),
            "ratio": float(bias_ratio)
        })
    
    # Create bias comparison chart
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = [item["layer"] for item in layer_bias_activations]
    bias1_values = [item["bias1_activation"] for item in layer_bias_activations]
    bias2_values = [item["bias2_activation"] for item in layer_bias_activations]
    
    ax.plot(layers, bias1_values, 'o-', label=selected_biases[0])
    ax.plot(layers, bias2_values, 's-', label=selected_biases[1])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Bias Activation")
    ax.set_title(f"Bias Comparison: {selected_biases[0]} vs {selected_biases[1]}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    bias_chart = fig_to_base64(fig)
    
    # Create ratio chart - cap ratios at 5 for visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ratios = [min(item["ratio"], 5) for item in layer_bias_activations]
    
    ax.bar(layers, ratios)
    ax.axhline(y=1, color='r', linestyle='--')
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Ratio ({selected_biases[0]} / {selected_biases[1]})")
    ax.set_title(f"Bias Ratio by Layer: {selected_biases[0]} / {selected_biases[1]}")
    
    ratio_chart = fig_to_base64(fig)
    
    # Safely compute average ratio
    ratio_avg = 0.0
    if ratios:
        ratio_avg = sum(ratios) / len(ratios)
    
    # Find max bias layer safely
    max_bias_layer = 0
    if ratios:
        max_bias_layer = layer_bias_activations[np.argmax(ratios)]["layer"]
    
     # Compute group means (example logic, adjust as needed)
    group_0_values = bias1_values[::2]
    group_1_values = bias1_values[1::2]

    group_0_mean = float(sum(group_0_values) / len(group_0_values)) if group_0_values else 0.0
    group_1_mean = float(sum(group_1_values) / len(group_1_values)) if group_1_values else 0.0

    # ✅ Build the response dictionary
    response = {
        "query": query,
        "selected_biases": selected_biases,
        "layer_bias_activations": layer_bias_activations,
        "charts": {
            "bias_comparison": bias_chart,
            "bias_ratio": ratio_chart
        },
        "summary": {
            "bias1_avg": float(sum(bias1_values) / len(bias1_values)) if bias1_values else 0.0,
            "bias2_avg": float(sum(bias2_values) / len(bias2_values)) if bias2_values else 0.0,
            "ratio_avg": float(ratio_avg),
            "max_bias_layer": int(max_bias_layer),
            "group0_mean": group_0_mean,
            "group1_mean": group_1_mean
        }
    }

    # ✅ Get recommendations dynamically from AI
    recommendations = generate_recommendations_from_ai(response["summary"])
    response["recommendations"] = recommendations

    return convert_to_native_python_types(response)

@app.post("/bias-scoring")
def bias_scoring(request: BiasScoringRequest):
    """Compute bias scores for selected concepts"""
    selected_cavs = request.selected_cavs
    num_samples = request.num_samples
    
    # Validate inputs
    if not selected_cavs:
        raise HTTPException(status_code=400, detail="At least one concept must be selected")
    
    try:
        # Load model
        model_path = preloaded_models["bert-base-uncased"]
        tokenizer, model, embed_model, kw_model, rag_generator = load_models(model_path)
        
        # Compute CAV embeddings - ensure on CPU
        cav_embeddings = embed_model.encode(selected_cavs, convert_to_tensor=True).cpu()
        
        # Retrieve stored cluster embeddings from MongoDB
        cluster_collection = db['cluster_embeddings']
        clusters = list(cluster_collection.find({}, {"_id": 0, "cluster_id": 1, "embedding": 1, "label": 1}))
        
        # If no clusters found, return empty scores
        if not clusters:
            return {
                "bias_scores": {cav: 0.0 for cav in selected_cavs},
                "normalized_scores": {cav: 0.0 for cav in selected_cavs},
                "charts": {
                    "bias_scores": None
                }
            }
        
        # Compute bias scores
        bias_scores = {}
        for cav in selected_cavs:
            cav_idx = selected_cavs.index(cav)
            cav_embedding = cav_embeddings[cav_idx].cpu()  # Ensure on CPU
            
            # Compute similarity to each cluster
            similarities = []
            for cluster in clusters:
                # Convert list to tensor and ensure on CPU
                cluster_embedding = torch.tensor(cluster["embedding"]).cpu()
                # Both tensors must be on the same device (CPU)
                sim = util.cos_sim(cav_embedding.unsqueeze(0), cluster_embedding.unsqueeze(0)).item()
                similarities.append(sim)
            
            # Compute average similarity as bias score
            bias_scores[cav] = float(sum(similarities) / len(similarities)) if similarities else 0.0
        
        # Normalize scores
        max_score = max(bias_scores.values()) if bias_scores else 1.0
        if max_score == 0:
            max_score = 1.0  # Avoid division by zero
        normalized_scores = {cav: float(score / max_score) for cav, score in bias_scores.items()}
        
        # Create bias scores chart
        fig, ax = plt.subplots(figsize=(10, 6))
        cavs = list(normalized_scores.keys())
        scores = list(normalized_scores.values())
        
        ax.bar(cavs, scores)
        ax.set_xlabel("Concept")
        ax.set_ylabel("Normalized Bias Score")
        ax.set_title("Normalized Bias Scores by Concept")
        ax.set_ylim(0, 1.1)
        
        for i, score in enumerate(scores):
            ax.text(i, score + 0.05, f"{score:.2f}", ha='center')
        
        bias_scores_chart = fig_to_base64(fig)
        
        # Prepare response
        response = {
            "bias_scores": bias_scores,
            "normalized_scores": normalized_scores,
            "charts": {
                "bias_scores": bias_scores_chart
            }
        }
        
        # Convert all numpy types to native Python types for JSON serialization
        # and handle non-JSON-compliant values like inf, -inf, and NaN
        return convert_to_native_python_types(response)
    
    except Exception as e:
        # Log the error for debugging
        print(f"Error in bias_scoring: {str(e)}")
        # Return a safe fallback response
        return {
            "bias_scores": {cav: 0.0 for cav in selected_cavs},
            "normalized_scores": {cav: 0.0 for cav in selected_cavs},
            "charts": {
                "bias_scores": None
            },
            "error": str(e)
        }
from fastapi import Request
import gitlab

import gitlab

@app.post("/trigger-gitlab-update")
def trigger_gitlab_update(payload: dict):
    repo_url = payload.get("repo_url")
    if not repo_url:
        raise HTTPException(status_code=400, detail="Repo URL required")

    # Extract project path like 'username/project'
    if "gitlab.com" not in repo_url:
        raise HTTPException(status_code=400, detail="Invalid GitLab URL")
    
    project_path = repo_url.replace("https://gitlab.com/", "").rstrip("/")

    # Connect to GitLab
    gl = gitlab.Gitlab('https://gitlab.com', private_token=os.getenv("GITLAB_TOKEN"))
    project = gl.projects.get(project_path)

    # Get latest pipeline or commit
    commits = project.commits.list(per_page=1)
    latest_commit = commits[0] if commits else None

    return {
        "status": "GitLab integration successful",
        "project": project_path,
        "latest_commit": latest_commit.id if latest_commit else "N/A"
    }


@app.post("/label-reminder")
def label_reminder():

    import os
    gl = gitlab.Gitlab('https://gitlab.com', private_token=os.getenv('GITLAB_TOKEN'))

    project = gl.projects.get('CodingWGitlab-group/CodingWGitlab-project')

    missing_labels = []
    for cluster_id in cluster_map.keys():
        label = get_cluster_label(cluster_id)
        if label == "unknown":
            issue_title = f"Missing label for cluster {cluster_id}"
            issue_body = f"Cluster {cluster_id} has no semantic label. Please annotate it."

            # Prevent duplicate issues
            existing = project.issues.list(search=issue_title)
            if not existing:
                project.issues.create({
                    'title': issue_title,
                    'description': issue_body
                })
                missing_labels.append(cluster_id)

    return {"status": "GitLab issues created", "clusters": missing_labels}
def get_cluster_label(cluster_id: int) -> str:
    doc = collection.find_one({"cluster_id": cluster_id})
    if doc and "label" in doc:
        return doc["label"]
    return "unknown"
@app.get("/recommendations/{query}")
async def get_recommendations(query: str):
    # Reuse summary from latest analysis or construct a mock one
    context = """
Cluster 0: profession
Cluster 1: gender bias
Top activated tokens: 'doctor', 'she'
High ratio found in Layer 11 (1.9x bias difference)
"""
    raw = generate_recommendations_from_ai(context, query)
    lines = [line.strip("-• ").strip() for line in raw.split("\n") if line.strip()]
    return {"recommendations": lines}

# Add this at the very end of testAPI.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# For Vercel deployment
app = app  # Make sure 'app' is available at module level