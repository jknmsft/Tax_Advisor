"""
Azure ML Compute Cluster Job Submission Script
Submit the tax advisor evaluation to run on a compute cluster
"""

from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import DefaultAzureCredential
import os

def create_environment():
    """Create custom environment for the job"""
    env = Environment(
        name="tax-advisor-eval-env",
        description="Environment for tax advisor model evaluation",
        build=BuildContext(
            path=".",
            dockerfile_path="Dockerfile"
        )
    )
    return env

def create_dockerfile():
    """Create Dockerfile for the environment"""
    dockerfile_content = """
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script
COPY tax_advisor_evaluation_script.py .

WORKDIR /
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    print("Dockerfile created successfully")

def submit_job():
    """Submit the evaluation job to Azure ML compute cluster"""
    
    # Create Dockerfile if it doesn't exist
    if not os.path.exists("Dockerfile"):
        create_dockerfile()
    
    # Initialize ML Client
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)
    
    # Create environment
    env = create_environment()
    
    # Define the command job
    job = command(
        code=".",  # Local folder containing the script
        command="python tax_advisor_evaluation_script.py --max_workers 8 --batch_size 20 --output_dir ${{outputs.results}}",
        environment=env,
        compute="your-compute-cluster-name",  # Replace with your compute cluster name
        display_name="tax-advisor-evaluation",
        description="Parallel evaluation of tax advisor models",
        outputs={
            "results": Output(type="uri_folder", path="azureml://datastores/workspaceblobstore/paths/tax-advisor-results/")
        },
        # Resource configuration for better performance
        instance_count=1,
        # You can also specify instance_type if needed
        # instance_type="Standard_D4s_v3"
    )
    
    # Submit the job
    print("Submitting job to Azure ML...")
    returned_job = ml_client.jobs.create_or_update(job)
    
    print(f"Job submitted successfully!")
    print(f"Job name: {returned_job.name}")
    print(f"Job URL: {returned_job.studio_url}")
    print(f"Status: {returned_job.status}")
    
    return returned_job

def submit_with_existing_environment():
    """Submit job using an existing curated environment"""
    
    # Initialize ML Client
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)
    
    # Use a curated environment instead of creating custom one
    # This is faster as it uses pre-built environments
    job = command(
        code=".",
        command="pip install -r requirements.txt && python tax_advisor_evaluation_script.py --max_workers 8 --batch_size 20 --output_dir ${{outputs.results}}",
        environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",  # Use curated environment
        compute="your-compute-cluster-name",  # Replace with your compute cluster name
        display_name="tax-advisor-evaluation-curated",
        description="Parallel evaluation of tax advisor models using curated environment",
        outputs={
            "results": Output(type="uri_folder", path="azureml://datastores/workspaceblobstore/paths/tax-advisor-results/")
        },
        instance_count=1,
    )
    
    # Submit the job
    print("Submitting job with curated environment...")
    returned_job = ml_client.jobs.create_or_update(job)
    
    print(f"Job submitted successfully!")
    print(f"Job name: {returned_job.name}")
    print(f"Job URL: {returned_job.studio_url}")
    print(f"Status: {returned_job.status}")
    
    return returned_job

def list_compute_clusters():
    """List available compute clusters"""
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)
    
    print("Available compute clusters:")
    for compute in ml_client.compute.list():
        print(f"  - {compute.name} ({compute.type})")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Submit tax advisor evaluation job')
    parser.add_argument('--list-clusters', action='store_true', help='List available compute clusters')
    parser.add_argument('--use-curated', action='store_true', help='Use curated environment instead of custom')
    parser.add_argument('--compute-name', type=str, help='Name of compute cluster to use')
    
    args = parser.parse_args()
    
    if args.list_clusters:
        list_compute_clusters()
    else:
        # Update compute cluster name if provided
        if args.compute_name:
            # You would need to modify the functions above to use this parameter
            print(f"Using compute cluster: {args.compute_name}")
        
        if args.use_curated:
            submit_with_existing_environment()
        else:
            submit_job()