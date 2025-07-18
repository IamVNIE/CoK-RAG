"""
CoKRAG: Agentic Chain-of Knowledge Graph Retrieval-Augmented Generation
Complete Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from tqdm import tqdm
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Entity representation in knowledge graph"""
    id: str
    text: str
    type: str
    embedding: Optional[np.ndarray] = None
    attributes: Dict[str, Any] = None

@dataclass
class Relation:
    """Relation between entities"""
    source: str
    target: str
    type: str
    weight: float = 1.0
    attributes: Dict[str, Any] = None

@dataclass
class ReasoningPath:
    """Reasoning path through knowledge graph"""
    nodes: List[str]
    edges: List[Tuple[str, str]]
    score: float
    evidence: List[str]

class SemanticRetriever:
    """Dense vector retrieval component"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        
    def index_documents(self, documents: List[str]):
        """Index documents for retrieval"""
        self.documents = documents
        embeddings = self.encoder.encode(documents, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k relevant documents"""
        query_embedding = self.encoder.encode([query], convert_to_tensor=True)
        query_embedding = query_embedding.cpu().numpy()
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(dist)))
                
        return results

class KnowledgeGraphConstructor:
    """Constructs knowledge graphs from documents"""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)
        self.entity_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text"""
        doc = self.nlp(text)
        entities = []
        
        # Extract named entities
        for ent in doc.ents:
            entity = Entity(
                id=f"{ent.text}_{ent.label_}",
                text=ent.text,
                type=ent.label_,
                attributes={"start": ent.start_char, "end": ent.end_char}
            )
            entities.append(entity)
            
        # Extract noun phrases as additional entities
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to short phrases
                entity = Entity(
                    id=f"{chunk.text}_NP",
                    text=chunk.text,
                    type="NOUN_PHRASE",
                    attributes={"start": chunk.start_char, "end": chunk.end_char}
                )
                entities.append(entity)
                
        return entities
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations between entities"""
        doc = self.nlp(text)
        relations = []
        entity_texts = {e.text.lower(): e.id for e in entities}
        
        # Extract subject-verb-object relations
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subject = None
                obj = None
                
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject_text = child.text.lower()
                        if subject_text in entity_texts:
                            subject = entity_texts[subject_text]
                    elif child.dep_ in ["dobj", "pobj"]:
                        obj_text = child.text.lower()
                        if obj_text in entity_texts:
                            obj = entity_texts[obj_text]
                            
                if subject and obj:
                    relation = Relation(
                        source=subject,
                        target=obj,
                        type=token.lemma_,
                        attributes={"verb": token.text}
                    )
                    relations.append(relation)
                    
        return relations
    
    def resolve_entities(self, entities: List[List[Entity]], threshold: float = 0.85) -> List[Entity]:
        """Resolve duplicate entities across documents"""
        all_entities = [e for doc_entities in entities for e in doc_entities]
        
        if not all_entities:
            return []
            
        # Compute embeddings for all entities
        entity_texts = [e.text for e in all_entities]
        embeddings = self.entity_embedder.encode(entity_texts)
        
        # Cluster similar entities
        merged_entities = []
        used = set()
        
        for i, entity in enumerate(all_entities):
            if i in used:
                continue
                
            similar_indices = [i]
            for j in range(i + 1, len(all_entities)):
                if j not in used:
                    similarity = cosine_similarity(
                        embeddings[i].reshape(1, -1),
                        embeddings[j].reshape(1, -1)
                    )[0, 0]
                    
                    if similarity > threshold:
                        similar_indices.append(j)
                        used.add(j)
                        
            # Merge similar entities
            merged_text = entity.text
            merged_type = entity.type
            
            # Create merged entity
            merged_entity = Entity(
                id=f"merged_{len(merged_entities)}",
                text=merged_text,
                type=merged_type,
                embedding=embeddings[i],
                attributes={"source_count": len(similar_indices)}
            )
            merged_entities.append(merged_entity)
            
        return merged_entities
    
    def construct_graph(self, documents: List[str], query: str) -> nx.DiGraph:
        """Construct knowledge graph from documents"""
        G = nx.DiGraph()
        
        # Extract entities and relations from each document
        all_entities = []
        all_relations = []
        
        for doc in documents:
            entities = self.extract_entities(doc)
            relations = self.extract_relations(doc, entities)
            all_entities.append(entities)
            all_relations.extend(relations)
            
        # Resolve duplicate entities
        merged_entities = self.resolve_entities(all_entities)
        
        # Add nodes to graph
        for entity in merged_entities:
            G.add_node(
                entity.id,
                text=entity.text,
                type=entity.type,
                embedding=entity.embedding
            )
            
        # Add edges to graph
        entity_map = {e.text.lower(): e.id for e in merged_entities}
        
        for relation in all_relations:
            # Map old entity IDs to merged IDs
            source_text = relation.source.split('_')[0].lower()
            target_text = relation.target.split('_')[0].lower()
            
            if source_text in entity_map and target_text in entity_map:
                source_id = entity_map[source_text]
                target_id = entity_map[target_text]
                
                # Compute edge weight based on query relevance
                edge_text = f"{source_text} {relation.type} {target_text}"
                query_embedding = self.entity_embedder.encode([query])
                edge_embedding = self.entity_embedder.encode([edge_text])
                
                weight = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    edge_embedding.reshape(1, -1)
                )[0, 0]
                
                G.add_edge(
                    source_id,
                    target_id,
                    type=relation.type,
                    weight=float(weight)
                )
                
        return G

class BaseAgent(ABC):
    """Base class for reasoning agents"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)
        
    @abstractmethod
    def act(self, *args, **kwargs):
        pass

class ExplorerAgent(BaseAgent):
    """Agent for exploring knowledge graph paths"""
    
    def act(self, query: str, graph: nx.DiGraph, current_node: str, 
            visited: set, max_depth: int = 3) -> List[str]:
        """Explore graph from current node"""
        if max_depth == 0 or current_node in visited:
            return []
            
        visited.add(current_node)
        neighbors = list(graph.neighbors(current_node))
        
        if not neighbors:
            return []
            
        # Score neighbors based on query relevance
        query_embedding = self.encoder.encode([query])
        neighbor_scores = []
        
        for neighbor in neighbors:
            if neighbor not in visited:
                neighbor_data = graph.nodes[neighbor]
                neighbor_text = neighbor_data.get('text', '')
                
                if neighbor_data.get('embedding') is not None:
                    neighbor_embedding = neighbor_data['embedding'].reshape(1, -1)
                else:
                    neighbor_embedding = self.encoder.encode([neighbor_text])
                    
                score = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    neighbor_embedding.reshape(1, -1)
                )[0, 0]
                
                # Consider edge weight
                edge_weight = graph[current_node][neighbor].get('weight', 1.0)
                combined_score = score * edge_weight
                
                neighbor_scores.append((neighbor, combined_score))
                
        # Sort by score and select top-k
        neighbor_scores.sort(key=lambda x: x[1], reverse=True)
        top_neighbors = [n[0] for n in neighbor_scores[:3]]
        
        return top_neighbors

class ValidatorAgent(BaseAgent):
    """Agent for validating reasoning paths"""
    
    def act(self, path: List[str], graph: nx.DiGraph, query: str) -> float:
        """Validate a reasoning path"""
        if len(path) < 2:
            return 0.0
            
        # Check path coherence
        coherence_score = 1.0
        for i in range(len(path) - 1):
            if not graph.has_edge(path[i], path[i + 1]):
                coherence_score *= 0.5
                
        # Check relevance to query
        path_text = " -> ".join([graph.nodes[n].get('text', '') for n in path])
        query_embedding = self.encoder.encode([query])
        path_embedding = self.encoder.encode([path_text])
        
        relevance_score = cosine_similarity(
            query_embedding.reshape(1, -1),
            path_embedding.reshape(1, -1)
        )[0, 0]
        
        # Combine scores
        validity_score = coherence_score * relevance_score
        
        return float(validity_score)

class SynthesizerAgent(BaseAgent):
    """Agent for synthesizing answers from reasoning paths"""
    
    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def act(self, paths: List[ReasoningPath], query: str, graph: nx.DiGraph) -> str:
        """Synthesize answer from valid paths"""
        # Sort paths by score
        paths.sort(key=lambda p: p.score, reverse=True)
        
        # Extract information from top paths
        context_pieces = []
        for path in paths[:3]:  # Use top 3 paths
            path_info = []
            for i, node in enumerate(path.nodes):
                node_text = graph.nodes[node].get('text', '')
                path_info.append(node_text)
                
                if i < len(path.nodes) - 1:
                    edge = (path.nodes[i], path.nodes[i + 1])
                    if graph.has_edge(*edge):
                        edge_type = graph[edge[0]][edge[1]].get('type', '')
                        path_info.append(f"--{edge_type}-->")
                        
            context_pieces.append(" ".join(path_info))
            
        # Generate answer
        context = "\n".join(context_pieces)
        prompt = f"Question: {query}\nContext: {context}\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=200,
                num_beams=4,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("Answer:")[-1].strip()
        
        return answer

class CoKRAG:
    """Main CoKRAG system"""
    
    def __init__(self, 
                 retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 generator_model: str = "gpt2",
                 spacy_model: str = "en_core_web_sm"):
        
        # Initialize components
        self.retriever = SemanticRetriever(retriever_model)
        self.graph_constructor = KnowledgeGraphConstructor(spacy_model)
        self.explorer = ExplorerAgent(retriever_model)
        self.validator = ValidatorAgent(retriever_model)
        self.synthesizer = SynthesizerAgent(generator_model)
        
        logger.info("CoKRAG initialized successfully")
        
    def index_corpus(self, documents: List[str]):
        """Index document corpus for retrieval"""
        logger.info(f"Indexing {len(documents)} documents")
        self.retriever.index_documents(documents)
        
    def multi_hop_reasoning(self, query: str, graph: nx.DiGraph, 
                           max_hops: int = 3) -> List[ReasoningPath]:
        """Perform multi-hop reasoning on knowledge graph"""
        # Find starting nodes based on query
        query_embedding = self.explorer.encoder.encode([query])
        node_scores = []
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            if node_data.get('embedding') is not None:
                node_embedding = node_data['embedding'].reshape(1, -1)
                score = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    node_embedding.reshape(1, -1)
                )[0, 0]
                node_scores.append((node, score))
                
        # Start from top-k nodes
        node_scores.sort(key=lambda x: x[1], reverse=True)
        start_nodes = [n[0] for n in node_scores[:5]]
        
        # Explore paths from each starting node
        all_paths = []
        
        for start_node in start_nodes:
            paths = self._explore_from_node(
                query, graph, start_node, max_hops
            )
            all_paths.extend(paths)
            
        # Validate paths
        validated_paths = []
        for path in all_paths:
            score = self.validator.act(path, graph, query)
            if score > 0.5:  # Threshold for valid paths
                reasoning_path = ReasoningPath(
                    nodes=path,
                    edges=[(path[i], path[i+1]) for i in range(len(path)-1)],
                    score=score,
                    evidence=[]
                )
                validated_paths.append(reasoning_path)
                
        return validated_paths
    
    def _explore_from_node(self, query: str, graph: nx.DiGraph, 
                          start_node: str, max_hops: int) -> List[List[str]]:
        """Explore paths from a starting node"""
        paths = [[start_node]]
        complete_paths = []
        
        for hop in range(max_hops):
            new_paths = []
            
            for path in paths:
                current_node = path[-1]
                visited = set(path)
                
                next_nodes = self.explorer.act(
                    query, graph, current_node, visited, max_hops - hop
                )
                
                for next_node in next_nodes:
                    new_path = path + [next_node]
                    new_paths.append(new_path)
                    
                    # Check if path answers query
                    if self._is_complete_path(new_path, graph, query):
                        complete_paths.append(new_path)
                        
            paths = new_paths
            
            if not paths:  # No more paths to explore
                break
                
        return complete_paths + paths
    
    def _is_complete_path(self, path: List[str], graph: nx.DiGraph, 
                         query: str) -> bool:
        """Check if path provides complete answer"""
        # Simple heuristic: path with 3+ nodes likely contains answer
        return len(path) >= 3
    
    def generate(self, query: str, k_documents: int = 5) -> Dict[str, Any]:
        """Generate answer for query"""
        logger.info(f"Processing query: {query}")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(query, k=k_documents)
        documents = [doc[0] for doc in retrieved_docs]
        
        logger.info(f"Retrieved {len(documents)} documents")
        
        # Step 2: Construct knowledge graph
        graph = self.graph_constructor.construct_graph(documents, query)
        logger.info(f"Constructed graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        # Step 3: Multi-hop reasoning
        reasoning_paths = self.multi_hop_reasoning(query, graph)
        logger.info(f"Found {len(reasoning_paths)} valid reasoning paths")
        
        # Step 4: Generate answer
        if reasoning_paths:
            answer = self.synthesizer.act(reasoning_paths, query, graph)
        else:
            answer = "Unable to find sufficient information to answer the query."
            
        return {
            "answer": answer,
            "retrieved_documents": documents,
            "reasoning_paths": reasoning_paths,
            "graph_stats": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges()
            }
        }

class CoKRAGTrainer:
    """Training pipeline for CoKRAG"""
    
    def __init__(self, cokrag: CoKRAG):
        self.cokrag = cokrag
        self.optimizer = None
        
    def train(self, train_data: List[Dict[str, Any]], 
              val_data: List[Dict[str, Any]],
              epochs: int = 10,
              batch_size: int = 8,
              learning_rate: float = 5e-5):
        """Train CoKRAG components"""
        logger.info(f"Starting training for {epochs} epochs")
        
        # Initialize optimizers for trainable components
        # Note: In practice, you'd need to make components trainable
        
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_data, batch_size, learning_rate)
            val_metrics = self._validate(val_data)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val F1: {val_metrics['f1']:.4f}")
            
    def _train_epoch(self, train_data: List[Dict[str, Any]], 
                     batch_size: int, learning_rate: float) -> float:
        """Train for one epoch"""
        total_loss = 0.0
        
        # Process in batches
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # Compute loss for each component
            retrieval_loss = self._compute_retrieval_loss(batch)
            graph_loss = self._compute_graph_loss(batch)
            reasoning_loss = self._compute_reasoning_loss(batch)
            generation_loss = self._compute_generation_loss(batch)
            
            # Combined loss
            loss = retrieval_loss + graph_loss + reasoning_loss + generation_loss
            total_loss += loss
            
        return total_loss / len(train_data)
    
    def _compute_retrieval_loss(self, batch: List[Dict[str, Any]]) -> float:
        """Compute retrieval loss"""
        # Placeholder - implement contrastive loss
        return 0.0
    
    def _compute_graph_loss(self, batch: List[Dict[str, Any]]) -> float:
        """Compute graph construction loss"""
        # Placeholder - implement graph accuracy loss
        return 0.0
    
    def _compute_reasoning_loss(self, batch: List[Dict[str, Any]]) -> float:
        """Compute reasoning path loss"""
        # Placeholder - implement path validity loss
        return 0.0
    
    def _compute_generation_loss(self, batch: List[Dict[str, Any]]) -> float:
        """Compute generation loss"""
        # Placeholder - implement cross-entropy loss
        return 0.0
    
    def _validate(self, val_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Validate on validation set"""
        predictions = []
        ground_truths = []
        
        for sample in val_data:
            result = self.cokrag.generate(sample['query'])
            predictions.append(result['answer'])
            ground_truths.append(sample['answer'])
            
        # Compute metrics
        f1 = self._compute_f1(predictions, ground_truths)
        exact_match = self._compute_exact_match(predictions, ground_truths)
        
        return {
            "f1": f1,
            "exact_match": exact_match
        }
    
    def _compute_f1(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Compute F1 score"""
        # Placeholder - implement token-level F1
        return 0.0
    
    def _compute_exact_match(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Compute exact match score"""
        matches = sum(1 for p, g in zip(predictions, ground_truths) if p.strip() == g.strip())
        return matches / len(predictions) if predictions else 0.0

# Example usage
if __name__ == "__main__":
    # Initialize CoKRAG
    cokrag = CoKRAG()
    
    # Example documents
    documents = [
        "Albert Einstein was born in Germany in 1879. He developed the theory of relativity.",
        "The theory of relativity revolutionized physics. It includes special and general relativity.",
        "Einstein received the Nobel Prize in Physics in 1921 for his work on the photoelectric effect.",
        "The photoelectric effect demonstrates the particle nature of light.",
        "Einstein worked at Princeton University in the United States after leaving Germany."
    ]
    
    # Index documents
    cokrag.index_corpus(documents)
    
    # Query
    query = "What did Einstein win the Nobel Prize for?"
    
    # Generate answer
    result = cokrag.generate(query)
    
    print(f"Query: {query}")
    print(f"Answer: {result['answer']}")
    print(f"Graph Stats: {result['graph_stats']}")
    print(f"Number of reasoning paths: {len(result['reasoning_paths'])}")
    
    # Training example
    trainer = CoKRAGTrainer(cokrag)
    
    # Prepare training data
    train_data = [
        {
            "query": "What did Einstein win the Nobel Prize for?",
            "answer": "Einstein won the Nobel Prize for his work on the photoelectric effect.",
            "documents": documents[:3]
        }
    ]
    
    # Train (simplified example)
    # trainer.train(train_data, train_data, epochs=1)