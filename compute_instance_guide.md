"""
Azure ML Compute Instance Setup and Testing Guide
Step-by-step instructions for running the tax advisor evaluation on a compute instance
"""

# Azure ML Compute Instance Testing Guide

## 🖥️ Setting Up Your Compute Instance

### 1. Create/Start Compute Instance
```bash
# In Azure ML Studio:
# 1. Go to Compute → Compute Instances
# 2. Create new or start existing instance
# 3. Recommended: Standard_DS3_v2 (4 cores, 14 GB RAM) for testing
# 4. Wait for instance to be "Running"
```

### 2. Access Your Compute Instance
```bash
# Option A: Jupyter Terminal
# 1. Click "Jupyter" link in Azure ML Studio
# 2. Open Terminal from Jupyter interface

# Option B: VS Code (if enabled)
# 1. Click "VS Code" link in Azure ML Studio
# 2. Use integrated terminal
```

### 3. Upload Your Files
```bash
# Option A: Using Jupyter interface
# 1. Upload files through Jupyter file browser
# 2. Create new folder: tax_advisor_evaluation
# 3. Upload train_data_v4.jsonl to the same folder

# Option B: Using git (recommended)
cd /home/azureuser
git clone <your-repo-url>
cd tax_advisor_evaluation

# Option C: Upload data file separately
# If you have the JSONL file locally, upload it via Jupyter:
# 1. Go to Jupyter file browser
# 2. Click "Upload" button
# 3. Select your train_data_v4.jsonl file
# 4. Upload to your project folder
```

## 🧪 Testing on Compute Instance

### Step 1: Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify Azure authentication (should work automatically)
python -c "from azure.identity import DefaultAzureCredential; print('Auth OK')"
```

### Step 2: Configuration
```bash
# Update config.json with your settings
nano config.json

# Or test without config file using command line args
```

### Step 3: Run Tests
```bash
# Test 1: Data processing without API calls
python test_local.py

# Test 2: Small dataset with local data file (recommended for testing)
python tax_advisor_evaluation_script.py --max_records 5 --max_workers 2 --use_local --output_dir ./test_results

# Test 3: Small dataset with Azure Data Lake (if configured)
python tax_advisor_evaluation_script.py --max_records 5 --max_workers 2 --output_dir ./test_results

# Test 4: Medium dataset for performance testing
python tax_advisor_evaluation_script.py --max_records 50 --max_workers 4 --use_local --output_dir ./test_results_50

# Test 5: Custom data file path
python tax_advisor_evaluation_script.py --local_data ./data/train_data_v4.jsonl --max_records 10 --max_workers 2

# Test 6: Monitor resource usage
htop  # or top to monitor CPU/memory usage during execution
```

## 📊 Monitoring and Analysis

### Real-time Monitoring
```bash
# Monitor logs in real-time
tail -f evaluation.log

# Check resource usage
htop

# Monitor disk space
df -h
```

### Results Analysis
```bash
# Check results
ls -la ./test_results/
head -10 ./test_results/answer_scores.csv
head -5 ./test_results/df_with_answers.csv

# Quick stats
python -c "
import pandas as pd
df = pd.read_csv('./test_results/answer_scores.csv')
print('Score Statistics:')
print(df[['reference_answer_score', 'base_answer_score', 'fine_tuned_answer_score']].describe())
print('\nBest Answer Counts:')
print(df['best_answer'].value_counts())
"

# View visualizations in Jupyter
# Open Jupyter → Navigate to test_results → Click evaluation_visualizations.png
```

### Download Results to Local Machine
```bash
# Method 1: Through Jupyter Interface
# 1. Go to Jupyter file browser
# 2. Navigate to your results folder
# 3. Select files → Right-click → Download

# Method 2: Using Azure ML SDK (if needed)
python -c "
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import shutil

# Create a zip of results for easy download
shutil.make_archive('evaluation_results', 'zip', './test_results')
print('Created evaluation_results.zip for download')
"
```

## 🔧 Optimization on Compute Instance

### Instance Size Recommendations for Testing
```bash
# Small testing (5-50 records):
# Standard_DS2_v2: 2 cores, 7 GB RAM, ~$0.14/hour
# max_workers: 2-4

# Medium testing (50-200 records):
# Standard_DS3_v2: 4 cores, 14 GB RAM, ~$0.27/hour  
# max_workers: 4-6

# Large testing (200-500 records):
# Standard_DS4_v2: 8 cores, 28 GB RAM, ~$0.54/hour
# max_workers: 8-12
```

### Performance Tuning on Compute Instance
```python
# Create a quick performance test script
cat > performance_test.py << 'EOF'
import time
import argparse
from tax_advisor_evaluation_script import TaxAdvisorEvaluator

def quick_perf_test():
    """Quick performance test with different worker configurations"""
    
    worker_configs = [2, 4, 6, 8]
    test_records = 20
    
    print("Performance Test Results:")
    print("Workers | Time (sec) | Records/sec")
    print("-" * 35)
    
    for workers in worker_configs:
        start_time = time.time()
        
        # Run test (you'd need to implement a mock version)
        # This is pseudocode - adapt based on your needs
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = test_records / duration if duration > 0 else 0
        
        print(f"{workers:>7} | {duration:>10.1f} | {throughput:>11.2f}")

if __name__ == "__main__":
    quick_perf_test()
EOF
```

## 🚀 Transitioning to Compute Cluster

Once testing is complete on compute instance:

```bash
# Test the submission script from compute instance
python submit_to_cluster.py --list-clusters

# Submit a small job to cluster for validation
python submit_to_cluster.py --use-curated --compute-name your-cluster-name

# Monitor the submitted job
# Go to Azure ML Studio → Jobs to track progress
```

## 💡 Best Practices for Compute Instance Testing

### 1. Resource Management
```bash
# Check available resources before starting
free -h          # Memory usage
df -h            # Disk usage
nproc            # Number of cores

# Set resource limits if needed
ulimit -v 4000000  # Limit virtual memory to ~4GB
```

### 2. Cost Optimization
```bash
# Auto-shutdown configuration
# Set in Azure ML Studio: Compute → Settings → Auto-shutdown
# Recommended: 30-60 minutes of inactivity

# Manual shutdown when done
# Always stop compute instance when not in use
```

### 3. Data Management
```bash
# Use Azure ML datasets for large data files
# Store results in Azure ML datastores
# Clean up temporary files regularly

# Example: Upload results to datastore
az ml data create --file ./test_results --name test-results-$(date +%Y%m%d)
```

### 4. Debugging Tips
```bash
# Enable verbose logging
python tax_advisor_evaluation_script.py --max_records 5 --max_workers 1 > debug.log 2>&1

# Use Python debugger for issues
python -m pdb tax_advisor_evaluation_script.py --max_records 2

# Check Azure service connectivity
ping cogseraccrciru5aiyeus2.cognitiveservices.azure.com
nslookup destadls2gzxfcx6xwhxssc.blob.core.windows.net
```

## ❗ Common Issues and Solutions

### Authentication Issues
```bash
# Re-authenticate if needed (usually not required on compute instance)
az login --identity  # For managed identity
az account show      # Verify current account
```

### Network Issues
```bash
# Test connectivity to Azure services
curl -I https://cogseraccrciru5aiyeus2.cognitiveservices.azure.com/
```

### Memory Issues
```bash
# Monitor memory usage during execution
watch -n 5 'free -h && ps aux --sort=-%mem | head -10'

# Reduce batch size if memory issues occur
python tax_advisor_evaluation_script.py --batch_size 5 --max_workers 2
```

### Permission Issues
```bash
# Verify Azure ML workspace access
python -c "from azure.ai.ml import MLClient; from azure.identity import DefaultAzureCredential; MLClient.from_config(credential=DefaultAzureCredential())"
```

## 📋 Testing Checklist

- [ ] Compute instance is running and accessible
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Configuration updated in `config.json`
- [ ] Authentication working (automatic on compute instance)
- [ ] Test data processing works (`python test_local.py`)
- [ ] Small API test successful (5-10 records)
- [ ] Performance test completed with different worker counts
- [ ] Results saved and analyzed
- [ ] Resource usage monitored and optimized
- [ ] Ready to transition to compute cluster

## 🎯 Expected Testing Timeline

- **Setup**: 10-15 minutes
- **Small test** (5 records): 2-5 minutes
- **Medium test** (50 records): 15-30 minutes  
- **Performance optimization**: 30-60 minutes
- **Total testing time**: 1-2 hours

This comprehensive testing on a compute instance will help you optimize settings before running the full dataset on a compute cluster!