"""
CoKRAG: Utilities, Visualization, and Complete Example
Additional components for the CoKRAG system
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import pandas as pd
import json
from pathlib import Path
import pickle
import time
from datetime import datetime
import numpy as np
import networkx as nx
from sklearn.metrics import precision_recall_fscore_support
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Import main CoKRAG components (assuming they're in cokrag.py)
# from cokrag import CoKRAG, Entity, Relation, ReasoningPath

class CoKRAGVisualizer:
    """Visualization utilities for CoKRAG"""
    
    @staticmethod
    def visualize_knowledge_graph(graph: nx.DiGraph, query: str = None, 
                                  reasoning_paths: List = None,
                                  save_path: str = None):
        """Visualize the knowledge graph with optional reasoning paths highlighted"""
        plt.figure(figsize=(15, 10))
        
        # Layout
        pos = nx.spring_layout(graph, k=2, iterations=50)
        
        # Draw all nodes and edges first
        nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='lightblue', alpha=0.6)
        nx.draw_networkx_edges(graph, pos, alpha=0.3, edge_color='gray')
        
        # Highlight reasoning paths if provided
        if reasoning_paths:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(reasoning_paths)))
            
            for idx, path in enumerate(reasoning_paths):
                if hasattr(path, 'nodes'):
                    path_nodes = path.nodes
                    path_edges = [(path_nodes[i], path_nodes[i+1]) 
                                  for i in range(len(path_nodes)-1)]
                    
                    # Highlight path nodes
                    nx.draw_networkx_nodes(graph, pos, nodelist=path_nodes,
                                         node_size=700, node_color=[colors[idx]], 
                                         alpha=0.8)
                    
                    # Highlight path edges
                    nx.draw_networkx_edges(graph, pos, edgelist=path_edges,
                                         edge_color=[colors[idx]], width=3, alpha=0.8)
        
        # Draw labels
        labels = {node: graph.nodes[node].get('text', node)[:20] + '...' 
                  if len(graph.nodes[node].get('text', node)) > 20 
                  else graph.nodes[node].get('text', node)
                  for node in graph.nodes()}
        nx.draw_networkx_labels(graph, pos, labels, font_size=8)
        
        # Title
        if query:
            plt.title(f"Knowledge Graph for Query: {query}", fontsize=16)
        else:
            plt.title("Knowledge Graph", fontsize=16)
            
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_reasoning_paths(reasoning_paths: List, save_path: str = None):
        """Visualize reasoning path scores and lengths"""
        if not reasoning_paths:
            print("No reasoning paths to visualize")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Path scores
        scores = [path.score for path in reasoning_paths]
        ax1.bar(range(len(scores)), scores)
        ax1.set_xlabel("Path Index")
        ax1.set_ylabel("Validity Score")
        ax1.set_title("Reasoning Path Scores")
        
        # Path lengths
        lengths = [len(path.nodes) for path in reasoning_paths]
        ax2.hist(lengths, bins=max(lengths)-min(lengths)+1, edgecolor='black')
        ax2.set_xlabel("Path Length (number of nodes)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Path Lengths")
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class CoKRAGEvaluator:
    """Evaluation utilities for CoKRAG"""
    
    def __init__(self):
        self.results = []
        
    def evaluate_single(self, prediction: str, ground_truth: str, 
                       reasoning_paths: List = None) -> Dict[str, float]:
        """Evaluate a single prediction"""
        # Exact match
        exact_match = 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0
        
        # Token-level F1
        pred_tokens = prediction.lower().split()
        truth_tokens = ground_truth.lower().split()
        
        common_tokens = set(pred_tokens) & set(truth_tokens)
        precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common_tokens) / len(truth_tokens) if truth_tokens else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Reasoning complexity
        avg_path_length = np.mean([len(p.nodes) for p in reasoning_paths]) if reasoning_paths else 0
        num_paths = len(reasoning_paths) if reasoning_paths else 0
        
        return {
            "exact_match": exact_match,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "avg_path_length": avg_path_length,
            "num_reasoning_paths": num_paths
        }
    
    def evaluate_dataset(self, predictions: List[str], ground_truths: List[str],
                        additional_info: List[Dict] = None) -> Dict[str, float]:
        """Evaluate on entire dataset"""
        assert len(predictions) == len(ground_truths)
        
        all_metrics = []
        for i, (pred, truth) in enumerate(zip(predictions, ground_truths)):
            info = additional_info[i] if additional_info else {}
            metrics = self.evaluate_single(pred, truth, info.get('reasoning_paths'))
            all_metrics.append(metrics)
            
        # Aggregate metrics
        aggregated = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregated[f"avg_{key}"] = np.mean(values)
            aggregated[f"std_{key}"] = np.std(values)
            
        return aggregated
    
    def plot_evaluation_results(self, results: Dict[str, float], save_path: str = None):
        """Plot evaluation results"""
        # Extract metrics
        metrics = {}
        for key, value in results.items():
            if key.startswith('avg_'):
                metric_name = key.replace('avg_', '')
                metrics[metric_name] = value
                
        # Create bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics.keys(), metrics.values())
        
        # Color bars based on value
        for bar, value in zip(bars, metrics.values()):
            if value > 0.7:
                bar.set_color('green')
            elif value > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
                
        plt.xlabel("Metrics")
        plt.ylabel("Score")
        plt.title("CoKRAG Evaluation Results")
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class CoKRAGDataProcessor:
    """Data processing utilities for CoKRAG"""
    
    @staticmethod
    def load_hotpotqa(file_path: str, max_samples: int = None) -> List[Dict]:
        """Load HotpotQA dataset"""
        samples = []
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        for i, item in enumerate(data):
            if max_samples and i >= max_samples:
                break
                
            sample = {
                'id': item['_id'],
                'query': item['question'],
                'answer': item['answer'],
                'type': item['type'],
                'supporting_facts': item['supporting_facts'],
                'context': [para[1] for para in item['context']]
            }
            samples.append(sample)
            
        return samples
    
    @staticmethod
    def prepare_training_data(samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Prepare training and validation data"""
        # Split 80-20
        split_idx = int(0.8 * len(samples))
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        # Format for CoKRAG
        train_data = []
        for sample in train_samples:
            train_data.append({
                'query': sample['query'],
                'answer': sample['answer'],
                'documents': sample['context'],
                'supporting_facts': sample.get('supporting_facts', [])
            })
            
        val_data = []
        for sample in val_samples:
            val_data.append({
                'query': sample['query'],
                'answer': sample['answer'],
                'documents': sample['context'],
                'supporting_facts': sample.get('supporting_facts', [])
            })
            
        return train_data, val_data

class CoKRAGAnalyzer:
    """Analysis utilities for CoKRAG"""
    
    @staticmethod
    def analyze_graph_statistics(graphs: List[nx.DiGraph]) -> pd.DataFrame:
        """Analyze statistics of constructed knowledge graphs"""
        stats = []
        
        for i, graph in enumerate(graphs):
            stat = {
                'graph_id': i,
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'density': nx.density(graph),
                'avg_degree': np.mean([d for n, d in graph.degree()]),
                'max_degree': max([d for n, d in graph.degree()]) if graph.degree() else 0,
                'num_components': nx.number_weakly_connected_components(graph),
                'largest_component_size': len(max(nx.weakly_connected_components(graph), key=len))
                                         if graph.number_of_nodes() > 0 else 0
            }
            stats.append(stat)
            
        return pd.DataFrame(stats)
    
    @staticmethod
    def analyze_retrieval_performance(retrieval_results: List[Dict]) -> Dict[str, float]:
        """Analyze retrieval performance"""
        # Calculate metrics
        avg_num_docs = np.mean([len(r['documents']) for r in retrieval_results])
        avg_doc_length = np.mean([np.mean([len(d.split()) for d in r['documents']]) 
                                 for r in retrieval_results])
        
        # Relevance scores if available
        if 'scores' in retrieval_results[0]:
            avg_top_score = np.mean([r['scores'][0] if r['scores'] else 0 
                                    for r in retrieval_results])
            score_drop = np.mean([r['scores'][0] - r['scores'][-1] 
                                 if len(r['scores']) > 1 else 0 
                                 for r in retrieval_results])
        else:
            avg_top_score = 0
            score_drop = 0
            
        return {
            'avg_num_documents': avg_num_docs,
            'avg_document_length': avg_doc_length,
            'avg_top_relevance_score': avg_top_score,
            'avg_score_drop': score_drop
        }

class CoKRAGExperiment:
    """Complete experimental setup for CoKRAG"""
    
    def __init__(self, experiment_name: str, output_dir: str = "./experiments"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.cokrag = None
        self.evaluator = CoKRAGEvaluator()
        self.visualizer = CoKRAGVisualizer()
        self.analyzer = CoKRAGAnalyzer()
        
        # Logging
        self.log_file = self.output_dir / "experiment_log.txt"
        self._log(f"Experiment {experiment_name} initialized")
        
    def _log(self, message: str):
        """Log message to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
            
    def run_complete_experiment(self, train_data: List[Dict], test_data: List[Dict],
                               model_config: Dict = None):
        """Run complete experiment pipeline"""
        self._log("Starting complete experiment")
        
        # Initialize CoKRAG with config
        self._log("Initializing CoKRAG")
        from cokrag import CoKRAG  # Import here to avoid circular dependency
        self.cokrag = CoKRAG(**model_config) if model_config else CoKRAG()
        
        # Prepare corpus
        all_documents = []
        for sample in train_data + test_data:
            all_documents.extend(sample.get('documents', []))
        all_documents = list(set(all_documents))  # Remove duplicates
        
        self._log(f"Indexing {len(all_documents)} unique documents")
        self.cokrag.index_corpus(all_documents)
        
        # Run inference on test set
        self._log(f"Running inference on {len(test_data)} test samples")
        results = self._run_inference(test_data)
        
        # Evaluate
        self._log("Evaluating results")
        evaluation_results = self._evaluate_results(results, test_data)
        
        # Analyze
        self._log("Analyzing results")
        analysis_results = self._analyze_results(results)
        
        # Visualize
        self._log("Creating visualizations")
        self._create_visualizations(results, evaluation_results, analysis_results)
        
        # Save results
        self._log("Saving results")
        self._save_results(results, evaluation_results, analysis_results)
        
        self._log("Experiment completed successfully")
        
    def _run_inference(self, test_data: List[Dict]) -> List[Dict]:
        """Run inference on test data"""
        results = []
        
        for i, sample in enumerate(test_data):
            self._log(f"Processing sample {i+1}/{len(test_data)}")
            
            start_time = time.time()
            result = self.cokrag.generate(sample['query'])
            inference_time = time.time() - start_time
            
            result['ground_truth'] = sample['answer']
            result['inference_time'] = inference_time
            result['sample_id'] = sample.get('id', i)
            
            results.append(result)
            
        return results
    
    def _evaluate_results(self, results: List[Dict], test_data: List[Dict]) -> Dict:
        """Evaluate results"""
        predictions = [r['answer'] for r in results]
        ground_truths = [r['ground_truth'] for r in results]
        additional_info = [{'reasoning_paths': r.get('reasoning_paths', [])} for r in results]
        
        evaluation_results = self.evaluator.evaluate_dataset(
            predictions, ground_truths, additional_info
        )
        
        # Add timing statistics
        inference_times = [r['inference_time'] for r in results]
        evaluation_results['avg_inference_time'] = np.mean(inference_times)
        evaluation_results['std_inference_time'] = np.std(inference_times)
        
        return evaluation_results
    
    def _analyze_results(self, results: List[Dict]) -> Dict:
        """Analyze results"""
        analysis = {}
        
        # Graph statistics
        graphs = []
        for r in results:
            if 'graph' in r:
                graphs.append(r['graph'])
                
        if graphs:
            graph_stats = self.analyzer.analyze_graph_statistics(graphs)
            analysis['graph_statistics'] = graph_stats.to_dict()
            
        # Reasoning path analysis
        all_paths = []
        for r in results:
            all_paths.extend(r.get('reasoning_paths', []))
            
        if all_paths:
            path_lengths = [len(p.nodes) for p in all_paths]
            path_scores = [p.score for p in all_paths]
            
            analysis['reasoning_paths'] = {
                'total_paths': len(all_paths),
                'avg_path_length': np.mean(path_lengths),
                'std_path_length': np.std(path_lengths),
                'avg_path_score': np.mean(path_scores),
                'std_path_score': np.std(path_scores)
            }
            
        return analysis
    
    def _create_visualizations(self, results: List[Dict], 
                              evaluation_results: Dict,
                              analysis_results: Dict):
        """Create visualizations"""
        # Evaluation results plot
        self.evaluator.plot_evaluation_results(
            evaluation_results,
            save_path=str(self.output_dir / "evaluation_results.png")
        )
        
        # Sample knowledge graph visualizations
        for i in range(min(3, len(results))):  # Visualize first 3 graphs
            if 'graph' in results[i]:
                self.visualizer.visualize_knowledge_graph(
                    results[i]['graph'],
                    query=results[i].get('query'),
                    reasoning_paths=results[i].get('reasoning_paths'),
                    save_path=str(self.output_dir / f"knowledge_graph_{i}.png")
                )
                
        # Reasoning paths visualization
        if any('reasoning_paths' in r for r in results):
            all_paths = []
            for r in results[:10]:  # First 10 samples
                all_paths.extend(r.get('reasoning_paths', []))
                
            if all_paths:
                self.visualizer.plot_reasoning_paths(
                    all_paths,
                    save_path=str(self.output_dir / "reasoning_paths.png")
                )
                
    def _save_results(self, results: List[Dict], 
                     evaluation_results: Dict,
                     analysis_results: Dict):
        """Save all results"""
        # Save raw results
        with open(self.output_dir / "raw_results.json", 'w') as f:
            # Convert non-serializable objects
            serializable_results = []
            for r in results:
                sr = r.copy()
                if 'graph' in sr:
                    del sr['graph']  # Graphs aren't JSON serializable
                if 'reasoning_paths' in sr:
                    sr['reasoning_paths'] = [
                        {
                            'nodes': p.nodes,
                            'score': p.score,
                            'evidence': p.evidence
                        } for p in sr['reasoning_paths']
                    ]
                serializable_results.append(sr)
                
            json.dump(serializable_results, f, indent=2)
            
        # Save evaluation results
        with open(self.output_dir / "evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2)
            
        # Save analysis results
        with open(self.output_dir / "analysis_results.json", 'w') as f:
            json.dump(analysis_results, f, indent=2)
            
        # Save summary report
        self._create_summary_report(evaluation_results, analysis_results)
        
    def _create_summary_report(self, evaluation_results: Dict, analysis_results: Dict):
        """Create a summary report"""
        report = f"""
CoKRAG Experiment Summary Report
================================
Experiment: {self.experiment_name}
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Evaluation Results:
------------------
- Exact Match: {evaluation_results.get('avg_exact_match', 0):.3f} ± {evaluation_results.get('std_exact_match', 0):.3f}
- F1 Score: {evaluation_results.get('avg_f1', 0):.3f} ± {evaluation_results.get('std_f1', 0):.3f}
- Precision: {evaluation_results.get('avg_precision', 0):.3f} ± {evaluation_results.get('std_precision', 0):.3f}
- Recall: {evaluation_results.get('avg_recall', 0):.3f} ± {evaluation_results.get('std_recall', 0):.3f}
- Average Inference Time: {evaluation_results.get('avg_inference_time', 0):.3f}s

Reasoning Analysis:
------------------
"""
        if 'reasoning_paths' in analysis_results:
            rp = analysis_results['reasoning_paths']
            report += f"""- Total Reasoning Paths: {rp['total_paths']}
- Average Path Length: {rp['avg_path_length']:.2f} ± {rp['std_path_length']:.2f}
- Average Path Score: {rp['avg_path_score']:.3f} ± {rp['std_path_score']:.3f}
"""
            
        if 'graph_statistics' in analysis_results:
            gs = analysis_results['graph_statistics']
            report += f"""
Graph Statistics:
----------------
- Average Nodes: {np.mean(list(gs['num_nodes'].values())):.2f}
- Average Edges: {np.mean(list(gs['num_edges'].values())):.2f}
- Average Density: {np.mean(list(gs['density'].values())):.3f}
"""
            
        with open(self.output_dir / "summary_report.txt", 'w') as f:
            f.write(report)

# Example usage
if __name__ == "__main__":
    # Create experiment
    experiment = CoKRAGExperiment("hotpotqa_baseline")
    
    # Create sample data
    sample_data = [
        {
            'id': '1',
            'query': 'What award did the developer of the theory of relativity win?',
            'answer': 'Nobel Prize in Physics',
            'documents': [
                "Albert Einstein developed the theory of relativity.",
                "Einstein won the Nobel Prize in Physics in 1921.",
                "The Nobel Prize was awarded for his work on the photoelectric effect."
            ]
        },
        {
            'id': '2',
            'query': 'Where did Einstein work after leaving Germany?',
            'answer': 'Princeton University',
            'documents': [
                "Einstein left Germany in 1933.",
                "He joined Princeton University in the United States.",
                "Princeton became his home until his death in 1955."
            ]
        }
    ]
    
    # Run experiment
    experiment.run_complete_experiment(
        train_data=sample_data,
        test_data=sample_data,
        model_config={
            'retriever_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'generator_model': 'gpt2',
            'spacy_model': 'en_core_web_sm'
        }
    )
    
    print("Experiment completed! Check the ./experiments/hotpotqa_baseline directory for results.")