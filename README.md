# Tax Advisor Model Evaluation - Optimized for Compute Cluster

This repository contains an optimized version of your Azure ML notebook converted to run efficiently on compute clusters with parallel processing.

## 🚀 Key Improvements

### Performance Optimizations
- **Parallel Processing**: Uses ThreadPoolExecutor with configurable workers (default: 8)
- **Batch Processing**: Processes records in batches to reduce overhead
- **Error Handling**: Robust error handling with detailed logging
- **Rate Limiting**: Built-in delays to avoid API rate limits
- **Resume Capability**: Failed batches don't stop the entire job

### Expected Performance Gains
- **5-10x faster execution** compared to sequential notebook
- **Better resource utilization** on compute clusters
- **Reduced memory footprint** through batch processing
- **Automatic error recovery** for transient API failures

## 📁 Files Overview

- `tax_advisor_evaluation_script.py` - Main optimized evaluation script
- `submit_to_cluster.py` - Azure ML job submission script
- `test_local.py` - Local testing and validation script
- `compute_helper.py` - Interactive configuration helper
- `view_results.py` - Interactive results viewer and analyzer
- `compute_instance_guide.md` - Detailed guide for testing on Azure ML compute instances
- `requirements.txt` - Python dependencies
- `config.json` - Configuration settings
- `README.md` - This documentation

## � Quick Start Guide

### Option A: Azure ML Compute Instance Testing (Recommended)
Perfect for iterative development and testing:
```bash
# 1. Start compute instance in Azure ML Studio
# 2. Upload files or clone repository
# 3. Install dependencies
pip install -r requirements.txt

# 4. Test with small dataset
python tax_advisor_evaluation_script.py --max_records 10 --max_workers 2

# 5. Scale up for larger testing
python tax_advisor_evaluation_script.py --max_records 100 --max_workers 4

# 6. Submit to cluster when ready
python submit_to_cluster.py --use-curated --compute-name your-cluster-name
```

### Option B: Direct Cluster Submission
Skip local testing and go directly to cluster:
```bash
# 1. Configure settings
python compute_helper.py  # Get optimal configuration

# 2. Update config.json with your values

# 3. Submit job
python submit_to_cluster.py --use-curated --compute-name your-cluster-name
```

## �🛠️ Setup Instructions

### 1. Prerequisites
- Azure ML workspace configured
- Compute cluster created in Azure ML (see compute requirements below)
- Appropriate permissions for Azure OpenAI and Data Lake Storage

### 1.1 Compute Cluster Requirements

#### **Minimum Requirements**
- **Instance Type**: Standard_D4s_v3 (4 vCPUs, 16 GB RAM)
- **Max Workers**: 4-6 workers
- **Suitable for**: Testing and small datasets (<500 records)

#### **Recommended Configuration**
- **Instance Type**: Standard_D8s_v3 (8 vCPUs, 32 GB RAM)
- **Max Workers**: 8-12 workers
- **Suitable for**: Production workloads (500-5000 records)

#### **High-Performance Configuration**
- **Instance Type**: Standard_D16s_v3 (16 vCPUs, 64 GB RAM)
- **Max Workers**: 16-24 workers
- **Suitable for**: Large datasets (>5000 records)

#### **Important Notes**
- **I/O Bound Workload**: Since the script makes API calls, you can use more workers than CPU cores
- **API Rate Limits**: Azure OpenAI rate limits are usually the bottleneck, not CPU
- **Memory Usage**: ~2-4 GB base + ~100 MB per worker
- **Cost Optimization**: Start with Standard_D4s_v3 and scale up if needed

#### **Compute Instance vs Compute Cluster for Testing**

**Azure ML Compute Instance (Recommended for Testing)**
- ✅ Always-on, persistent environment
- ✅ Pre-configured with Azure ML SDK and authentication
- ✅ Better network connectivity to Azure services
- ✅ Integrated Jupyter notebooks and terminal
- ✅ No job submission overhead
- ✅ Interactive debugging and monitoring
- 💰 Cost: ~$0.10-0.50/hour depending on size

**Azure ML Compute Cluster (For Production)**
- ✅ Auto-scaling and job queuing
- ✅ Better for large batch processing
- ✅ Automatic shutdown after job completion
- ✅ Multiple concurrent jobs
- ❌ Job submission overhead
- ❌ Less interactive debugging
- 💰 Cost: Only when jobs are running

### **Testing Workflow Recommendation**
1. **Develop & Test**: Use compute instance for interactive development
2. **Validate**: Test with 10-100 records on compute instance  
3. **Optimize**: Tune worker count and batch size based on results
4. **Production**: Submit full dataset to compute cluster
5. **Monitor**: Track job progress in Azure ML Studio

### 2. Configuration
Update `config.json` with your specific values:
```json
{
  "azure_openai": {
    "endpoint": "your-endpoint-here",
    "ft_deployment": "your-fine-tuned-deployment",
    "pv_deployment": "your-base-deployment"
  },
  "compute_cluster": {
    "name": "your-cluster-name",
    "instance_type": "Standard_D4s_v3"  // Minimum 4 cores
  }
}
```

### 3. Determine Optimal Compute Configuration
Use the helper to get personalized recommendations:
```bash
python compute_helper.py
```

### 4. Testing Options

#### Option A: Azure ML Compute Instance (Recommended)
Use an Azure ML compute instance for testing - it has better network connectivity to Azure services:
```bash
# In Azure ML Studio, start your compute instance
# Open Terminal or Jupyter terminal
# Clone/upload your files to the compute instance

# Test data processing without API calls
python test_local.py

# Test with small dataset (has real Azure credentials)
python tax_advisor_evaluation_script.py --max_records 10 --max_workers 2

# Test full pipeline with larger subset
python tax_advisor_evaluation_script.py --max_records 50 --max_workers 4
```

#### Option B: Local Machine Testing
Test locally on your development machine:
```bash
# Test data processing without API calls (works offline)
python test_local.py

# Test with credentials (requires Azure CLI login or service principal)
az login
python tax_advisor_evaluation_script.py --max_records 5 --max_workers 2
```

## 🚀 Running on Compute Cluster

### Option 1: Quick Start (Recommended)
Uses curated Azure ML environment:
```bash
python submit_to_cluster.py --use-curated --compute-name your-cluster-name
```

### Option 2: Custom Environment
Creates custom Docker environment:
```bash
python submit_to_cluster.py --compute-name your-cluster-name
```

### Option 3: Direct Script Execution
If you prefer to submit via Azure ML Studio or CLI:
```bash
# Copy all files to your Azure ML workspace
# Submit as a command job with:
# python tax_advisor_evaluation_script.py --max_workers 8 --batch_size 20
```

## 📊 Accessing Results and Visualizations

### 📁 Output Files Generated
The script creates these files in your output directory:
- `answer_scores.csv` - Evaluation results DataFrame with scores
- `df_with_answers.csv` - Original data with model answers added
- `bad_responses.json` - Log of failed API calls (if any)
- `evaluation_visualizations.png` - Score distribution plots and comparisons
- `evaluation.log` - Detailed execution log

### 🖥️ **Option 1: Compute Instance Results**
When running on Azure ML compute instance:

#### Location
```bash
# Results saved to the directory you specified
ls -la ./results/                    # Default location
ls -la ./test_results/              # If using --output_dir ./test_results
```

#### Access Methods
```bash
# Method A: Jupyter File Browser
# 1. Open Jupyter from compute instance
# 2. Navigate to your results folder
# 3. Download files or view directly

# Method B: Terminal Commands
cd ./results
ls -la                              # List all result files
head -10 answer_scores.csv          # Preview scores
cat evaluation.log | tail -20       # View recent logs

# Method C: Download to Local Machine
# In Jupyter interface: Right-click files → Download
```

#### View Visualizations
```bash
# Method A: If using Jupyter notebooks
from IPython.display import Image, display
display(Image('./results/evaluation_visualizations.png'))

# Method B: Using matplotlib in notebook
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('./results/evaluation_visualizations.png')
plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.axis('off')
plt.show()
```

### ☁️ **Option 2: Compute Cluster Results**
When running jobs on compute cluster:

#### Location in Azure ML Studio
```
Azure ML Studio → Jobs → [Your Job Name] → Outputs + logs → results
```

#### Access Methods
```bash
# Method A: Azure ML Studio UI
# 1. Go to Azure ML Studio (ml.azure.com)
# 2. Navigate to Jobs
# 3. Click on your job name (e.g., "tax-advisor-evaluation")
# 4. Go to "Outputs + logs" tab
# 5. Click on "results" folder
# 6. Download individual files or entire folder

# Method B: Azure ML SDK/CLI
az ml job download --name [job-name] --output-name results --download-path ./local_results

# Method C: Direct Datastore Access
# Results are stored at: azureml://datastores/workspaceblobstore/paths/tax-advisor-results/
```

#### Programmatic Access
```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Initialize client
ml_client = MLClient.from_config(credential=DefaultAzureCredential())

# Get job details
job = ml_client.jobs.get("[your-job-name]")

# Download results
ml_client.jobs.download(name="[your-job-name]", 
                       output_name="results", 
                       download_path="./downloaded_results")
```

### 📊 **Option 3: Azure Storage Explorer**
For direct access to stored results:

#### Setup
```bash
# Install Azure Storage Explorer (GUI application)
# Or use Azure CLI
az storage blob list --account-name [workspace-storage] --container-name azureml-blobstore-[guid]
```

#### Access Path
```
Storage Account → azureml-blobstore-[guid] → tax-advisor-results/
```

### 📈 **Analyzing Results**

#### Interactive Results Viewer (Recommended)
```bash
# Use the built-in results viewer for comprehensive analysis
python view_results.py                    # Auto-detect results directories
python view_results.py --dir ./results    # Specify directory
python view_results.py --list             # List all available result directories
```

#### Quick Analysis Script
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results (adjust path as needed)
scores_df = pd.read_csv('./results/answer_scores.csv')
answers_df = pd.read_csv('./results/df_with_answers.csv')

# Basic statistics
print("Score Statistics:")
print(scores_df[['reference_answer_score', 'base_answer_score', 'fine_tuned_answer_score']].describe())

# Best answer distribution
print("\nBest Answer Distribution:")
print(scores_df['best_answer'].value_counts())

# Create custom visualization
plt.figure(figsize=(12, 8))
for column in ['reference_answer_score', 'base_answer_score', 'fine_tuned_answer_score']:
    sns.histplot(scores_df[column], bins=30, alpha=0.3, label=column)
plt.legend()
plt.title('Score Distribution Comparison')
plt.show()
```

### 💾 **Backup and Sharing Results**

#### Best Practices
```bash
# Create timestamped backup
timestamp=$(date +%Y%m%d_%H%M%S)
cp -r ./results ./results_backup_$timestamp

# Upload to Azure ML Datastore for sharing
az ml data create --file ./results --name evaluation-results-$timestamp --description "Tax advisor evaluation results"

# Share via git (for small result files)
git add ./results/*.csv ./results/*.png
git commit -m "Add evaluation results - $timestamp"
git push
```

### 🔍 **Monitoring Job Progress**
For cluster jobs, monitor real-time:

#### Azure ML Studio
- **Logs tab**: View real-time execution logs
- **Metrics tab**: See custom metrics logged by the script
- **Outputs tab**: Access results as they're generated

#### Programmatic Monitoring
```python
import time
from azure.ai.ml import MLClient

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
job_name = "your-job-name"

# Monitor job status
while True:
    job = ml_client.jobs.get(job_name)
    print(f"Job Status: {job.status}")
    
    if job.status in ["Completed", "Failed", "Canceled"]:
        break
    
    time.sleep(30)  # Check every 30 seconds

# Download results when complete
if job.status == "Completed":
    ml_client.jobs.download(name=job_name, output_name="results", download_path="./final_results")
    print("Results downloaded successfully!")
```

## ⚙️ Configuration Options

### Command Line Arguments
```bash
python tax_advisor_evaluation_script.py \
  --max_workers 8 \           # Number of parallel workers
  --batch_size 20 \           # Records per batch
  --output_dir ./results \    # Output directory
  --max_records 100           # Limit records (for testing)
```

### Performance Tuning
- **max_workers**: Start with 1-2x your CPU cores (I/O bound workload)
  - 4 cores: 4-8 workers
  - 8 cores: 8-16 workers
  - Monitor API rate limits and adjust accordingly
- **batch_size**: 10-50 records per batch (smaller = more frequent checkpoints)
- **instance_type**: Minimum Standard_D4s_v3 (4 cores), recommended Standard_D8s_v3 (8 cores)

## 🔧 Troubleshooting

### Common Issues

1. **API Rate Limiting**
   - Reduce `max_workers` (try 4 instead of 8)
   - Increase `request_delay` in config
   - Use smaller `batch_size`

2. **Memory Issues**
   - Reduce `batch_size`
   - Use compute instance with more RAM
   - Process data in chunks using `max_records`

3. **Authentication Errors**
   - Ensure Azure ML workspace config is correct
   - Check Azure OpenAI permissions
   - Verify storage account access

4. **Timeout Issues**
   - Increase timeout values in OpenAI client
   - Use more robust compute instance
   - Consider processing smaller subsets

### Error Recovery
The script automatically:
- Logs all errors with context
- Continues processing other batches on failure
- Saves partial results regularly
- Provides detailed error reports

## 📈 Performance Monitoring

### Key Metrics
- **Success Rate**: Percentage of successful evaluations
- **Throughput**: Records processed per minute
- **Error Rate**: Failed API calls ratio
- **Resource Utilization**: CPU/Memory usage

### Optimization Tips
1. Monitor API quotas and adjust worker count accordingly
2. Use appropriate compute instance size for your dataset
3. Consider data locality for better I/O performance
4. Implement checkpointing for very large datasets

## 🔄 Migration from Notebook

Your original notebook has been optimized with these changes:

### Data Processing
- ✅ Parallel processing of evaluation loop
- ✅ Batch processing for efficiency
- ✅ Robust error handling
- ✅ Progress tracking and logging

### API Calls
- ✅ Concurrent API calls with rate limiting
- ✅ Automatic retry logic
- ✅ Better timeout handling
- ✅ Response validation

### Output Management
- ✅ Incremental result saving
- ✅ Comprehensive error logging
- ✅ Automatic visualization generation
- ✅ Azure ML metric logging

## 📞 Support

If you encounter issues:
1. Check the `evaluation.log` file for detailed error messages
2. Review the `bad_responses.json` for API call failures
3. Monitor Azure ML job logs in the studio
4. Verify configuration settings in `config.json`

## 🎯 Next Steps

After successful execution:
1. Analyze results in the generated CSV files
2. Review visualizations for insights
3. Consider further optimizations based on performance metrics
4. Scale up to full dataset with optimal configuration