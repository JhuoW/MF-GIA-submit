import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple
import rootutils
root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from icl import InContextLearner, load_pretrained_model
from utils.logging import logger, timer
from utils.utils import seed_setting
import matplotlib.pyplot as plt
import seaborn as sns


class GAlignEvaluator:
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.learner = load_pretrained_model(model_path, device)
        self.results = {}
        
    @timer()
    def evaluate_few_shot_scaling(self, 
                                  dataset_name: str,
                                  k_shots: List[int] = [1, 3, 5, 10],
                                  n_episodes: int = 100) -> Dict:
        logger.info(f"Evaluating few-shot scaling on {dataset_name}")
        
        results = {}
        for k in k_shots:
            logger.info(f"Testing {k}-shot learning...")
            res = self.learner.evaluate_on_dataset(
                dataset_name=dataset_name,
                k_shot=k,
                n_episodes=n_episodes,
                eval_mode='few-shot'
            )
            results[f"{k}-shot"] = res
            
        return results
    
    @timer()
    def evaluate_cross_domain(self,
                            source_datasets: List[str],
                            target_datasets: List[str],
                            k_shot: int = 5,
                            n_episodes: int = 50) -> pd.DataFrame:
        logger.info("Evaluating cross-domain transfer...")
        
        results_matrix = []
        
        for target in target_datasets:
            row = {'target': target}
            
            fs_res = self.learner.evaluate_on_dataset(
                dataset_name=target,
                k_shot=k_shot,
                n_episodes=n_episodes,
                eval_mode='few-shot'
            )
            row['few_shot_acc'] = fs_res['accuracy_mean']
            row['few_shot_std'] = fs_res['accuracy_std']
            
            # Zero-shot evaluation
            zs_res = self.learner.evaluate_on_dataset(
                dataset_name=target,
                n_episodes=n_episodes,
                eval_mode='zero-shot'
            )
            row['zero_shot_acc'] = zs_res['accuracy_mean']
            row['zero_shot_std'] = zs_res['accuracy_std']
            
            row['in_pretrain'] = target in source_datasets
            
            results_matrix.append(row)
        
        return pd.DataFrame(results_matrix)
    
    @timer()
    def evaluate_m_way_scaling(self,
                              dataset_name: str,
                              m_ways: List[int] = [2, 3, 5, 7],
                              k_shot: int = 5,
                              n_episodes: int = 50) -> Dict:
        logger.info(f"Evaluating m-way scaling on {dataset_name}")
        
        results = {}
        for m in m_ways:
            logger.info(f"Testing {m}-way classification...")
            res = self.learner.evaluate_on_dataset(
                dataset_name=dataset_name,
                k_shot=k_shot,
                n_episodes=n_episodes,
                m_way=m,
                eval_mode='few-shot'
            )
            results[f"{m}-way"] = res
            
        return results
    
    def plot_results(self, save_dir: str = "evaluation_results"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        if 'few_shot_scaling' in self.results:
            self._plot_few_shot_scaling(self.results['few_shot_scaling'], save_dir)
        
        if 'cross_domain' in self.results:
            self._plot_cross_domain(self.results['cross_domain'], save_dir)
        
        if 'm_way_scaling' in self.results:
            self._plot_m_way_scaling(self.results['m_way_scaling'], save_dir)
    
    def _plot_few_shot_scaling(self, results: Dict, save_dir: str):
        k_values = []
        accuracies = []
        stds = []
        
        for k_str, res in results.items():
            k = int(k_str.split('-')[0])
            k_values.append(k)
            accuracies.append(res['accuracy_mean'])
            stds.append(res['accuracy_std'])
        
        plt.figure(figsize=(8, 6))
        plt.errorbar(k_values, accuracies, yerr=stds, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Number of Shots (k)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Few-Shot Learning Performance Scaling', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/few_shot_scaling.png", dpi=150)
        plt.close()
        
        logger.info(f"Few-shot scaling plot saved to {save_dir}/few_shot_scaling.png")
    
    def _plot_cross_domain(self, df: pd.DataFrame, save_dir: str):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        x = np.arange(len(df))
        width = 0.35
        
        ax1.bar(x, df['few_shot_acc'], width, yerr=df['few_shot_std'], 
                color=['green' if ip else 'blue' for ip in df['in_pretrain']],
                alpha=0.7)
        ax1.set_xlabel('Target Dataset', fontsize=12)
        ax1.set_ylabel('Few-Shot Accuracy', fontsize=12)
        ax1.set_title('Few-Shot Cross-Domain Transfer', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['target'], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Zero-shot results
        ax2.bar(x, df['zero_shot_acc'], width, yerr=df['zero_shot_std'],
                color=['green' if ip else 'blue' for ip in df['in_pretrain']],
                alpha=0.7)
        ax2.set_xlabel('Target Dataset', fontsize=12)
        ax2.set_ylabel('Zero-Shot Accuracy', fontsize=12)
        ax2.set_title('Zero-Shot Cross-Domain Transfer', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['target'], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='In Pretraining'),
                          Patch(facecolor='blue', alpha=0.7, label='Not in Pretraining')]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/cross_domain_transfer.png", dpi=150)
        plt.close()
        
        logger.info(f"Cross-domain plot saved to {save_dir}/cross_domain_transfer.png")
    
    def _plot_m_way_scaling(self, results: Dict, save_dir: str):
        m_values = []
        accuracies = []
        stds = []
        
        for m_str, res in results.items():
            m = int(m_str.split('-')[0])
            m_values.append(m)
            accuracies.append(res['accuracy_mean'])
            stds.append(res['accuracy_std'])
        
        plt.figure(figsize=(8, 6))
        plt.errorbar(m_values, accuracies, yerr=stds, marker='s', linewidth=2, markersize=8, color='red')
        plt.xlabel('Number of Classes (m-way)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('M-Way Classification Performance', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/m_way_scaling.png", dpi=150)
        plt.close()
        
        logger.info(f"M-way scaling plot saved to {save_dir}/m_way_scaling.png")
    
    def save_results(self, save_path: str = "evaluation_results.json"):
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, pd.DataFrame):
                json_results[key] = value.to_dict('records')
            else:
                json_results[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {save_path}")
    
    def run_comprehensive_evaluation(self,
                                    test_datasets: List[str] = ["cora", "citeseer"],
                                    pretrain_datasets: List[str] = ["pubmed", "arxiv", "wikics"]):
        logger.info("Starting comprehensive evaluation...")
        
        if test_datasets:
            self.results['few_shot_scaling'] = self.evaluate_few_shot_scaling(
                dataset_name=test_datasets[0],
                k_shots=[1, 3, 5, 10],
                n_episodes=50
            )
        
        all_datasets = test_datasets + [d for d in pretrain_datasets if d not in test_datasets]
        self.results['cross_domain'] = self.evaluate_cross_domain(
            source_datasets=pretrain_datasets,
            target_datasets=all_datasets[:5],  # Limit to 5 datasets
            k_shot=5,
            n_episodes=30
        )
        
        if test_datasets:
            self.results['m_way_scaling'] = self.evaluate_m_way_scaling(
                dataset_name=test_datasets[0],
                m_ways=[2, 3, 5],
                k_shot=5,
                n_episodes=30
            )
        
        self.save_results()
        self.plot_results()
        
        self.print_summary()
    
    def print_summary(self):
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        if 'few_shot_scaling' in self.results:
            print("\nFew-Shot Scaling Results:")
            print("-"*40)
            for k_str, res in self.results['few_shot_scaling'].items():
                print(f"  {k_str:10s}: {res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f}")
        
        if 'cross_domain' in self.results:
            print("\nCross-Domain Transfer Results:")
            print("-"*40)
            df = self.results['cross_domain']
            print(df.to_string(index=False))
        
        if 'm_way_scaling' in self.results:
            print("\nM-Way Scaling Results:")
            print("-"*40)
            for m_str, res in self.results['m_way_scaling'].items():
                print(f"  {m_str:10s}: {res['accuracy_mean']:.4f} ± {res['accuracy_std']:.4f}")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="G-Align Comprehensive Evaluation")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to pretrained model')
    parser.add_argument('--test_datasets', nargs='+', default=["cora"],
                       help='Datasets to test on')
    parser.add_argument('--pretrain_datasets', nargs='+', 
                       default=["pubmed", "arxiv", "wikics"],
                       help='Datasets used in pretraining')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    seed_setting(args.seed)
    
    evaluator = GAlignEvaluator(args.model_path, args.device)
    
    evaluator.run_comprehensive_evaluation(
        test_datasets=args.test_datasets,
        pretrain_datasets=args.pretrain_datasets
    )


if __name__ == "__main__":
    main()