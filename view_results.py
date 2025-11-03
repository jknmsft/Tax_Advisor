"""
Results Viewer and Analyzer
Quick script to view and analyze evaluation results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse
from pathlib import Path

def find_results_directories():
    """Find all directories containing results"""
    current_dir = Path(".")
    result_dirs = []
    
    # Common result directory names
    common_names = ["results", "test_results", "evaluation_results", "output"]
    
    for name in common_names:
        if (current_dir / name).exists():
            result_dirs.append(current_dir / name)
    
    # Also check for numbered result directories
    for item in current_dir.iterdir():
        if item.is_dir() and "result" in item.name.lower():
            result_dirs.append(item)
    
    return result_dirs

def load_results(results_dir):
    """Load all result files from a directory"""
    results_dir = Path(results_dir)
    results = {}
    
    # Load CSV files
    for csv_file in ["answer_scores.csv", "df_with_answers.csv"]:
        csv_path = results_dir / csv_file
        if csv_path.exists():
            try:
                results[csv_file] = pd.read_csv(csv_path)
                print(f"✓ Loaded {csv_file}: {len(results[csv_file])} rows")
            except Exception as e:
                print(f"✗ Error loading {csv_file}: {e}")
        else:
            print(f"- {csv_file} not found")
    
    # Load JSON files
    bad_responses_path = results_dir / "bad_responses.json"
    if bad_responses_path.exists():
        try:
            with open(bad_responses_path, 'r') as f:
                results["bad_responses.json"] = json.load(f)
            print(f"✓ Loaded bad_responses.json: {len(results['bad_responses.json'])} entries")
        except Exception as e:
            print(f"✗ Error loading bad_responses.json: {e}")
    
    # Check for visualization
    viz_path = results_dir / "evaluation_visualizations.png"
    if viz_path.exists():
        results["visualization_path"] = viz_path
        print(f"✓ Found visualization: {viz_path}")
    else:
        print("- evaluation_visualizations.png not found")
    
    return results

def analyze_scores(scores_df):
    """Analyze evaluation scores"""
    print("\n" + "="*60)
    print("SCORE ANALYSIS")
    print("="*60)
    
    score_columns = ['reference_answer_score', 'base_answer_score', 'fine_tuned_answer_score']
    available_columns = [col for col in score_columns if col in scores_df.columns]
    
    if not available_columns:
        print("No score columns found!")
        return
    
    # Basic statistics
    print("\n📊 Score Statistics:")
    print(scores_df[available_columns].describe().round(3))
    
    # Best answer distribution
    if 'best_answer' in scores_df.columns:
        print("\n🏆 Best Answer Distribution:")
        counts = scores_df['best_answer'].value_counts()
        total = len(scores_df)
        for answer, count in counts.items():
            percentage = count / total * 100
            print(f"  {answer}: {count} ({percentage:.1f}%)")
    
    # Score comparisons
    if len(available_columns) >= 2:
        print("\n📈 Score Comparisons:")
        for i, col1 in enumerate(available_columns):
            for col2 in available_columns[i+1:]:
                diff = scores_df[col1] - scores_df[col2]
                mean_diff = diff.mean()
                print(f"  {col1} vs {col2}: {mean_diff:+.3f} average difference")
    
    # Quality insights
    print("\n💡 Quality Insights:")
    if 'reference_answer_score' in available_columns:
        high_quality = (scores_df['reference_answer_score'] >= 0.8).sum()
        print(f"  High-quality references (≥0.8): {high_quality}/{total} ({high_quality/total*100:.1f}%)")
    
    if 'fine_tuned_answer_score' in available_columns and 'base_answer_score' in available_columns:
        ft_wins = (scores_df['fine_tuned_answer_score'] > scores_df['base_answer_score']).sum()
        print(f"  Fine-tuned outperforms base: {ft_wins}/{total} ({ft_wins/total*100:.1f}%)")

def create_summary_visualization(scores_df, output_path=None):
    """Create a summary visualization"""
    score_columns = ['reference_answer_score', 'base_answer_score', 'fine_tuned_answer_score']
    available_columns = [col for col in score_columns if col in scores_df.columns]
    
    if len(available_columns) < 2:
        print("Not enough score columns for visualization")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Evaluation Results Summary', fontsize=16, fontweight='bold')
    
    # Score distributions
    ax1 = axes[0, 0]
    for col in available_columns:
        sns.histplot(scores_df[col], bins=30, alpha=0.6, label=col.replace('_', ' ').title(), ax=ax1)
    ax1.set_title('Score Distributions')
    ax1.set_xlabel('Score')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Best answer pie chart
    ax2 = axes[0, 1]
    if 'best_answer' in scores_df.columns:
        counts = scores_df['best_answer'].value_counts()
        ax2.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Best Answer Distribution')
    else:
        ax2.text(0.5, 0.5, 'Best Answer\nData Not Available', ha='center', va='center', fontsize=12)
        ax2.set_title('Best Answer Distribution')
    
    # Score comparison (if we have multiple scores)
    ax3 = axes[1, 0]
    if len(available_columns) >= 2:
        col1, col2 = available_columns[0], available_columns[1]
        ax3.scatter(scores_df[col1], scores_df[col2], alpha=0.6)
        ax3.plot([0, 1], [0, 1], 'r--', alpha=0.8)  # Perfect correlation line
        ax3.set_xlabel(col1.replace('_', ' ').title())
        ax3.set_ylabel(col2.replace('_', ' ').title())
        ax3.set_title(f'{col1.replace("_", " ").title()} vs {col2.replace("_", " ").title()}')
        ax3.grid(True, alpha=0.3)
    
    # Score differences (if we have fine-tuned and base)
    ax4 = axes[1, 1]
    if 'fine_tuned_answer_score' in available_columns and 'base_answer_score' in available_columns:
        diff = scores_df['fine_tuned_answer_score'] - scores_df['base_answer_score']
        sns.histplot(diff, bins=30, ax=ax4)
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        ax4.set_title('Fine-tuned vs Base Score Difference')
        ax4.set_xlabel('Score Difference (FT - Base)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Score Comparison\nData Not Available', ha='center', va='center', fontsize=12)
        ax4.set_title('Score Comparison')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved summary visualization to {output_path}")
    
    plt.show()

def view_results(results_dir):
    """Main function to view and analyze results"""
    results_dir = Path(results_dir)
    print(f"\n📁 Analyzing results in: {results_dir}")
    print("="*60)
    
    # Load results
    results = load_results(results_dir)
    
    if not results:
        print("No result files found!")
        return
    
    # Analyze scores if available
    if "answer_scores.csv" in results:
        analyze_scores(results["answer_scores.csv"])
        
        # Create summary visualization
        print(f"\n🎨 Creating summary visualization...")
        summary_viz_path = results_dir / "summary_analysis.png"
        create_summary_visualization(results["answer_scores.csv"], summary_viz_path)
    
    # Show bad responses if any
    if "bad_responses.json" in results:
        bad_responses = results["bad_responses.json"]
        if bad_responses:
            print(f"\n⚠️  BAD RESPONSES ANALYSIS")
            print(f"Total failed responses: {len(bad_responses)}")
            
            # Group by error type
            error_types = {}
            for response in bad_responses:
                error_key = response.get('error', 'Unknown error')[:50]  # First 50 chars
                error_types[error_key] = error_types.get(error_key, 0) + 1
            
            print("Error types:")
            for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error}: {count}")
    
    # Show existing visualization if available
    if "visualization_path" in results:
        print(f"\n🖼️  Original visualization available at: {results['visualization_path']}")
    
    print(f"\n✅ Analysis complete! Check {results_dir} for all files.")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description='View and analyze evaluation results')
    parser.add_argument('--dir', type=str, help='Results directory path')
    parser.add_argument('--list', action='store_true', help='List available result directories')
    
    args = parser.parse_args()
    
    if args.list:
        print("🔍 Searching for result directories...")
        result_dirs = find_results_directories()
        if result_dirs:
            print("Found result directories:")
            for i, dir_path in enumerate(result_dirs, 1):
                print(f"  {i}. {dir_path}")
        else:
            print("No result directories found in current location")
        return
    
    if args.dir:
        results_dir = args.dir
    else:
        # Interactive selection
        result_dirs = find_results_directories()
        if not result_dirs:
            print("No result directories found. Please specify --dir or run the evaluation first.")
            return
        elif len(result_dirs) == 1:
            results_dir = result_dirs[0]
            print(f"Using results directory: {results_dir}")
        else:
            print("Multiple result directories found:")
            for i, dir_path in enumerate(result_dirs, 1):
                print(f"  {i}. {dir_path}")
            
            try:
                choice = int(input("Select directory (number): ")) - 1
                results_dir = result_dirs[choice]
            except (ValueError, IndexError):
                print("Invalid selection. Using first directory.")
                results_dir = result_dirs[0]
    
    view_results(results_dir)

if __name__ == "__main__":
    main()