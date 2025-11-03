"""
Local Testing Script for Tax Advisor Evaluation
Test the optimized script locally with a small subset of data
"""

import os
import sys
import json
import pandas as pd
from tax_advisor_evaluation_script import TaxAdvisorEvaluator, get_evaluation_prompt

def create_test_data():
    """Create sample test data for local testing"""
    # Create a small sample dataset that mimics your JSONL structure
    test_data = []
    
    for i in range(5):  # Create 5 test records
        record = {
            "messages": [{
                "content": f"""You are a tax advisor assistant.

Question:
What are the tax implications of selling my primary residence after living there for {i+1} years?

Answer from the finetuning:
The tax implications depend on several factors including how long you've lived there and your filing status.

Relevant Information Chunks::
- Primary residence exclusion applies if you've lived there 2 out of 5 years
- Single filers can exclude up to $250,000 in gains
- Married filing jointly can exclude up to $500,000 in gains"""
            }],
            "reference_answer": {
                "answer": f"If you've lived in your primary residence for {i+1} years and meet the ownership test, you may qualify for the capital gains exclusion. For single filers, you can exclude up to $250,000 in gains, and for married filing jointly, up to $500,000."
            }
        }
        test_data.append(record)
    
    return pd.DataFrame(test_data)

def test_data_extraction():
    """Test the data extraction functions"""
    print("Testing data extraction...")
    
    # Create test data
    df = create_test_data()
    
    # Mock configuration (you'll need to replace with real values for actual testing)
    config = {
        'endpoint': "https://your-endpoint.cognitiveservices.azure.com/",
        'ft_deployment': "your-ft-deployment",
        'pv_deployment': "your-pv-deployment", 
        'api_version': "2024-12-01-preview",
        'system_prompt': "Return a JSON object with an 'answer' field.",
        'evaluation_prompt': get_evaluation_prompt()
    }
    
    # Initialize evaluator (this will fail without real credentials, but we can test data processing)
    try:
        evaluator = TaxAdvisorEvaluator(config)
        
        # Test data extraction
        df_processed = evaluator.extract_data_components(df)
        
        print("Data extraction successful!")
        print(f"Extracted questions: {df_processed['extracted_question'].tolist()}")
        print(f"Extracted prompts: {df_processed['prompt'].tolist()[:1]}...")  # Show first prompt
        print(f"DataFrame shape: {df_processed.shape}")
        
        return df_processed
        
    except Exception as e:
        print(f"Error during data extraction test: {e}")
        # Still return the processed dataframe for testing
        return df

def create_mock_config():
    """Create a mock configuration for testing without real API calls"""
    return {
        'endpoint': "https://mock-endpoint.cognitiveservices.azure.com/",
        'ft_deployment': "mock-ft-deployment",
        'pv_deployment': "mock-pv-deployment",
        'api_version': "2024-12-01-preview",
        'system_prompt': "Return a JSON object with an 'answer' field.",
        'evaluation_prompt': get_evaluation_prompt(),
        'storage_account': "mock-storage",
        'storage_key': "mock-key",
        'jsonl_path': "mock-path.jsonl"
    }

def test_without_api():
    """Test the script structure without making actual API calls"""
    print("Testing script structure without API calls...")
    
    # Create test data
    df = create_test_data()
    
    # Mock the evaluator methods that don't require API calls
    config = create_mock_config()
    
    class MockEvaluator(TaxAdvisorEvaluator):
        def setup_clients(self):
            """Mock client setup"""
            print("Mock: Clients setup skipped")
            self.ft_client = None
            self.pv_client = None
            self.ml_client = None
        
        def generate_response(self, client, prompt, content, model, tokens=32678):
            """Mock response generation"""
            mock_response = {
                'success': True,
                'content': json.dumps({
                    'answer': f'Mock answer for model {model}: This is a test response.'
                })
            }
            return mock_response
    
    # Test with mock evaluator
    evaluator = MockEvaluator(config)
    
    # Test data extraction
    df_processed = evaluator.extract_data_components(df)
    print("✓ Data extraction works")
    
    # Test batch processing logic (without real API calls)
    batch_data = [(idx, row) for idx, row in df_processed.iterrows()]
    
    # Mock a small batch
    if len(batch_data) > 0:
        try:
            # This will use our mocked generate_response method
            results = evaluator.process_batch(batch_data[:2], 0)
            print(f"✓ Batch processing works - got {len(results)} results")
        except Exception as e:
            print(f"✗ Batch processing failed: {e}")
    
    # Test result saving
    try:
        test_output_dir = "./test_results"
        os.makedirs(test_output_dir, exist_ok=True)
        evaluator.save_results(df_processed, test_output_dir)
        print("✓ Result saving works")
    except Exception as e:
        print(f"✗ Result saving failed: {e}")
    
    print("Mock testing completed!")

def test_performance_estimation():
    """Estimate performance improvements"""
    print("\nPerformance Analysis:")
    print("="*50)
    
    # Original notebook approach (sequential)
    original_time_per_record = 15  # Estimate: 3 API calls × 5 seconds each
    total_records = 1000  # Adjust based on your dataset size
    
    sequential_time = original_time_per_record * total_records
    print(f"Original notebook (sequential):")
    print(f"  - Time per record: ~{original_time_per_record} seconds")
    print(f"  - Total time for {total_records} records: ~{sequential_time/3600:.1f} hours")
    
    # Optimized script approach (parallel)
    parallel_workers = 8
    batch_size = 20
    optimized_time_per_record = 8  # Reduced due to better error handling and batching
    
    parallel_time = (original_time_per_record * total_records) / parallel_workers * 0.7  # 70% efficiency
    print(f"\nOptimized script (parallel):")
    print(f"  - Workers: {parallel_workers}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Time per record: ~{optimized_time_per_record} seconds")
    print(f"  - Total time for {total_records} records: ~{parallel_time/3600:.1f} hours")
    
    speedup = sequential_time / parallel_time
    print(f"\nExpected speedup: {speedup:.1f}x faster")
    print(f"Time savings: {(sequential_time - parallel_time)/3600:.1f} hours")

def main():
    """Main testing function"""
    print("Tax Advisor Evaluation - Local Testing")
    print("="*50)
    
    # Test data extraction
    test_data_extraction()
    
    print("\n" + "="*50)
    
    # Test without API calls
    test_without_api()
    
    # Performance estimation
    test_performance_estimation()
    
    print("\n" + "="*50)
    print("Testing completed!")
    print("\nNext steps:")
    print("1. Update the configuration in tax_advisor_evaluation_script.py with your actual values")
    print("2. Test with a small subset using: python tax_advisor_evaluation_script.py --max_records 10")
    print("3. Submit to cluster using: python submit_to_cluster.py --use-curated")

if __name__ == "__main__":
    main()