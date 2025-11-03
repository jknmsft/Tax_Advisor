# Azure ML Workspace Configuration Setup

If you're getting an error about missing Azure ML workspace configuration, here are the solutions:

## Option 1: Quick Fix - Skip Azure ML Client (Recommended for testing)
The script now automatically handles missing Azure ML configuration and will continue working for basic evaluation.

## Option 2: Configure Azure ML Workspace (Required for cluster jobs)
If you plan to submit jobs to Azure ML compute clusters, you need to configure your workspace:

### Step 1: Find Your Workspace Details
1. Go to Azure Portal (portal.azure.com)
2. Navigate to your Azure ML workspace
3. Note down:
   - Subscription ID
   - Resource Group name
   - Workspace name

### Step 2: Update Configuration Files

#### Method A: Update .azureml/config.json
```json
{
    "subscription_id": "12345678-1234-1234-1234-123456789012",
    "resource_group": "my-resource-group",
    "workspace_name": "my-ml-workspace"
}
```

#### Method B: Set Environment Variables
```bash
# Windows (PowerShell)
$env:AZUREML_SUBSCRIPTION_ID="your-subscription-id"
$env:AZUREML_RESOURCE_GROUP="your-resource-group"
$env:AZUREML_WORKSPACE_NAME="your-workspace-name"

# Linux/Mac
export AZUREML_SUBSCRIPTION_ID="your-subscription-id"
export AZUREML_RESOURCE_GROUP="your-resource-group"
export AZUREML_WORKSPACE_NAME="your-workspace-name"
```

#### Method C: Use Azure CLI
```bash
az login
az account set --subscription "your-subscription-id"
az configure --defaults workspace="your-workspace-name" group="your-resource-group"
```

## Option 3: Run Without Azure ML Workspace
For basic evaluation without cluster submission:

```bash
# This will work even without Azure ML workspace configuration
python tax_advisor_evaluation_script.py --use_local --max_records 10
```

## Testing Your Setup

### Test 1: Basic Functionality (No Azure ML needed)
```bash
python tax_advisor_evaluation_script.py --use_local --max_records 5 --max_workers 1
```

### Test 2: With Azure ML Workspace
```bash
python tax_advisor_evaluation_script.py --max_records 5
```

If Test 2 works, you can proceed with cluster job submission.