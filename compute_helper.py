"""
Compute Configuration Helper
Helps determine optimal compute settings based on your requirements
"""

def get_compute_recommendations():
    """Get compute configuration recommendations"""
    
    print("=== Azure ML Compute Configuration Helper ===\n")
    
    # Get user requirements
    try:
        dataset_size = int(input("How many records in your dataset? "))
        budget_conscious = input("Are you budget-conscious? (y/n): ").lower().startswith('y')
        rush_job = input("Do you need results urgently? (y/n): ").lower().startswith('y')
    except ValueError:
        print("Invalid input. Using default recommendations.")
        dataset_size = 1000
        budget_conscious = True
        rush_job = False
    
    print(f"\n=== Recommendations for {dataset_size} records ===\n")
    
    # Determine recommendations based on input
    if dataset_size <= 100:
        # Small dataset
        instance_type = "Standard_D2s_v3"
        workers = 2
        estimated_time = "10-30 minutes"
        cost_tier = "Low"
    elif dataset_size <= 500:
        # Medium dataset
        if budget_conscious:
            instance_type = "Standard_D4s_v3"
            workers = 6
        else:
            instance_type = "Standard_D8s_v3"
            workers = 8
        estimated_time = "30 minutes - 2 hours"
        cost_tier = "Low-Medium"
    elif dataset_size <= 2000:
        # Large dataset
        if budget_conscious and not rush_job:
            instance_type = "Standard_D4s_v3"
            workers = 6
            estimated_time = "2-6 hours"
        elif rush_job:
            instance_type = "Standard_D16s_v3"
            workers = 20
            estimated_time = "45 minutes - 2 hours"
        else:
            instance_type = "Standard_D8s_v3"
            workers = 12
            estimated_time = "1-3 hours"
        cost_tier = "Medium"
    else:
        # Very large dataset
        if budget_conscious:
            instance_type = "Standard_D8s_v3"
            workers = 12
            estimated_time = "3-8 hours"
        else:
            instance_type = "Standard_D16s_v3"
            workers = 24
            estimated_time = "1-4 hours"
        cost_tier = "Medium-High"
    
    # Display recommendations
    print(f"📊 **Recommended Configuration:**")
    print(f"   Instance Type: {instance_type}")
    print(f"   Max Workers: {workers}")
    print(f"   Estimated Time: {estimated_time}")
    print(f"   Cost Tier: {cost_tier}")
    
    # Get instance specs
    specs = get_instance_specs(instance_type)
    print(f"\n💻 **Instance Specifications:**")
    print(f"   vCPUs: {specs['cores']}")
    print(f"   RAM: {specs['ram']} GB")
    print(f"   Approximate Cost: ${specs['cost_per_hour']:.2f}/hour")
    
    # Configuration commands
    print(f"\n⚙️ **Configuration Commands:**")
    print(f"   # Update config.json:")
    print(f'   "max_workers": {workers},')
    print(f'   "instance_type": "{instance_type}",')
    print(f"\n   # Command line:")
    print(f"   python tax_advisor_evaluation_script.py --max_workers {workers}")
    print(f"   python submit_to_cluster.py --compute-name your-cluster --instance-type {instance_type}")
    
    # Rate limiting considerations
    print(f"\n⚠️ **Important Considerations:**")
    print(f"   • API Rate Limits: Monitor Azure OpenAI quotas")
    print(f"   • Start Conservative: Begin with fewer workers, scale up")
    print(f"   • I/O Bound: More workers than cores is OK for API calls")
    print(f"   • Cost Management: Use auto-shutdown and spot instances if available")
    
    return {
        'instance_type': instance_type,
        'workers': workers,
        'estimated_time': estimated_time,
        'cost_tier': cost_tier
    }

def get_instance_specs(instance_type):
    """Get specifications for Azure instance types"""
    specs = {
        'Standard_D2s_v3': {'cores': 2, 'ram': 8, 'cost_per_hour': 0.096},
        'Standard_D4s_v3': {'cores': 4, 'ram': 16, 'cost_per_hour': 0.192},
        'Standard_D8s_v3': {'cores': 8, 'ram': 32, 'cost_per_hour': 0.384},
        'Standard_D16s_v3': {'cores': 16, 'ram': 64, 'cost_per_hour': 0.768},
        'Standard_D32s_v3': {'cores': 32, 'ram': 128, 'cost_per_hour': 1.536},
    }
    return specs.get(instance_type, {'cores': 4, 'ram': 16, 'cost_per_hour': 0.192})

def analyze_current_config():
    """Analyze current configuration and suggest improvements"""
    try:
        import json
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        current_workers = config.get('processing', {}).get('max_workers', 4)
        current_instance = config.get('compute_cluster', {}).get('instance_type', 'Standard_D4s_v3')
        
        specs = get_instance_specs(current_instance)
        cores = specs['cores']
        
        print(f"\n=== Current Configuration Analysis ===")
        print(f"Instance Type: {current_instance}")
        print(f"vCPUs: {cores}")
        print(f"Max Workers: {current_workers}")
        print(f"Worker-to-Core Ratio: {current_workers/cores:.1f}")
        
        # Provide analysis
        if current_workers < cores:
            print("⚠️  UNDERUTILIZED: You could increase workers for better performance")
            print(f"   Suggested workers: {cores * 2}")
        elif current_workers > cores * 3:
            print("⚠️  OVERSUBSCRIBED: Too many workers may cause overhead")
            print(f"   Suggested workers: {cores * 2}")
        else:
            print("✅ OPTIMAL: Good worker-to-core ratio for I/O bound workload")
            
    except FileNotFoundError:
        print("config.json not found. Run from the project directory.")
    except Exception as e:
        print(f"Error analyzing config: {e}")

def cost_calculator():
    """Calculate estimated costs for different configurations"""
    print("\n=== Cost Calculator ===")
    
    try:
        dataset_size = int(input("Dataset size: "))
        hours_estimate = dataset_size / 500  # Rough estimate: 500 records per hour
        
        configurations = [
            ('Standard_D4s_v3', 6, 0.192),
            ('Standard_D8s_v3', 12, 0.384),
            ('Standard_D16s_v3', 24, 0.768),
        ]
        
        print(f"\nEstimated costs for {dataset_size} records:")
        print("Instance Type     | Workers | Hours | Cost")
        print("-" * 45)
        
        for instance, workers, cost_per_hour in configurations:
            # Adjust time estimate based on workers
            estimated_hours = hours_estimate * (6 / workers)  # Scale based on workers
            total_cost = estimated_hours * cost_per_hour
            print(f"{instance:<17} | {workers:>7} | {estimated_hours:>5.1f} | ${total_cost:>5.2f}")
            
    except ValueError:
        print("Invalid input.")

def main():
    """Main menu"""
    while True:
        print("\n" + "="*50)
        print("Azure ML Compute Configuration Helper")
        print("="*50)
        print("1. Get compute recommendations")
        print("2. Analyze current configuration")
        print("3. Cost calculator")
        print("4. Instance specifications")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            get_compute_recommendations()
        elif choice == '2':
            analyze_current_config()
        elif choice == '3':
            cost_calculator()
        elif choice == '4':
            show_instance_specs()
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1-5.")

def show_instance_specs():
    """Show all instance specifications"""
    print("\n=== Azure VM Instance Specifications ===")
    print("Instance Type     | vCPUs | RAM(GB) | Cost/Hour | Recommended Workers")
    print("-" * 75)
    
    instances = [
        ('Standard_D2s_v3', 2, 8, 0.096, '2-4'),
        ('Standard_D4s_v3', 4, 16, 0.192, '4-8'),
        ('Standard_D8s_v3', 8, 32, 0.384, '8-16'),
        ('Standard_D16s_v3', 16, 64, 0.768, '16-32'),
        ('Standard_D32s_v3', 32, 128, 1.536, '32-64'),
    ]
    
    for instance, cores, ram, cost, workers in instances:
        print(f"{instance:<17} | {cores:>5} | {ram:>7} | ${cost:>8.3f} | {workers:>15}")

if __name__ == "__main__":
    main()