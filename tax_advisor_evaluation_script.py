"""
Azure ML Tax Advisor Model Evaluation Script
Optimized for compute cluster execution with parallel processing
"""

import pandas as pd
import json
import os
import fsspec 
import re
import logging
import argparse
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
from azureml.core import Workspace, Run
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.ai.ml.entities import Data
from openai import AzureOpenAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TaxAdvisorEvaluator:
    """Main evaluation class with optimized processing"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the evaluator with configuration"""
        self.config = config
        self.setup_clients()
        self.evaluation_list = []
        self.bad_response_list = []
        
    def setup_clients(self):
        """Setup Azure ML and OpenAI clients"""
        logger.info("Setting up Azure clients...")
        
        # Authenticate and create MLClient
        credential = DefaultAzureCredential()
        self.ml_client = MLClient.from_config(credential=credential)
        
        # Setup token provider
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), 
            "https://cognitiveservices.azure.com/.default"
        )
        
        # Fine-tuned model client
        self.ft_client = AzureOpenAI(
            api_version=self.config['api_version'],
            azure_endpoint=self.config['endpoint'],
            azure_ad_token_provider=token_provider,
        )
        
        # Base model client
        self.pv_client = AzureOpenAI(
            api_version=self.config['api_version'],
            azure_endpoint=self.config['endpoint'],    
            azure_ad_token_provider=token_provider
        )
        
        logger.info("Azure clients configured successfully")
    
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess data from Azure Data Lake or local workspace"""
        
        # Determine data source based on configuration
        if self.config.get('use_local_data', False) or not self.config['jsonl_path'].startswith('abfs://'):
            # Load from local file or workspace folder
            local_path = self.config.get('local_jsonl_path', self.config['jsonl_path'])
            logger.info(f"Loading data from local/workspace file: {local_path}")
            
            try:
                df = pd.read_json(local_path, lines=True)
                logger.info(f"Loaded {len(df)} records from {local_path}")
            except FileNotFoundError:
                logger.error(f"File not found: {local_path}")
                logger.info("Make sure the file exists in your workspace or provide the correct path")
                raise
            except Exception as e:
                logger.error(f"Error reading local file {local_path}: {str(e)}")
                raise
                
        else:
            # Load from Azure Data Lake Storage
            logger.info("Loading data from Azure Data Lake...")
            
            # Setup filesystem connection
            fs = fsspec.filesystem(
                "abfs",
                account_name=self.config['storage_account'],
                credential=self.config['storage_key']
            )
            
            # Read JSONL file
            try:
                with fs.open(self.config['jsonl_path'], 'r') as f:
                    df = pd.read_json(f, lines=True)
                logger.info(f"Loaded {len(df)} records from {self.config['jsonl_path']}")
            except Exception as e:
                logger.error(f"Error reading from Azure Data Lake {self.config['jsonl_path']}: {str(e)}")
                raise
        
        # Extract components
        df = self.extract_data_components(df)
        
        # Initialize answer columns
        df['o4_ft_answer'] = ''
        df['o4_pv_answer'] = ''
        
        return df
    
    def extract_data_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract question, prompt, and other components from messages"""
        logger.info("Extracting data components...")
        
        # Extract question text
        def extract_question_text(cell):
            if isinstance(cell, list) and len(cell) == 1 and isinstance(cell[0], dict):        
                content = cell[0].get('content', '')
                match = re.search(r'Question:\n(.*?)\nAnswer', content, re.DOTALL)
                if match:            
                    return match.group(1).strip()        
            return None
        
        # Extract system prompt
        def extract_prompt(message_list):
            try:
                content = message_list[0].get('content', '')
                first_index = content.find('Question:')
                if first_index != -1:
                    return content[:first_index].strip()
                else:
                    return ''
            except (IndexError, AttributeError):
                return ''
        
        # Extract full message
        def extract_full_message(message_list):
            try:
                content = message_list[0].get('content', '')
                first_index = content.find('Question:')
                if first_index != -1:
                    return content[first_index:].strip()
                else:
                    return ''
            except (IndexError, AttributeError):
                return ''
        
        # Extract POKA chunks
        def extract_poka(message_list):
            try:
                content = message_list[0].get('content', '')
                first_index = content.find('Relevant Information Chunks::')
                if first_index != -1:
                    first_index = first_index + len('Relevant Information Chunks::')
                    return content[first_index:].strip()
                else:
                    return ''
            except (IndexError, AttributeError):
                return ''
        
        # Extract full FT answer
        def extract_full_ft_answer(message_list):
            try:
                content = message_list[0].get('content', '')
                first_index = content.find('Answer from the finetuning:')
                second_index = content.find('Relevant Information Chunks::')
                if first_index != -1:
                    first_index = first_index + len('Answer from the finetuning:')
                    return content[first_index:second_index].strip()
                else:
                    return ''
            except (IndexError, AttributeError):
                return ''
        
        # Apply extractions
        df['extracted_question'] = df['messages'].apply(extract_question_text)
        df['prompt'] = df['messages'].apply(extract_prompt)
        df['full_message'] = df['messages'].apply(extract_full_message)
        df['poka_chunks'] = df['messages'].apply(extract_poka)
        df['full_ft_answer'] = df['messages'].apply(extract_full_ft_answer)
        
        # Extract checkpoint answer
        df['ckp_answer'] = df['reference_answer'].apply(lambda x: x['answer'])
        
        logger.info("Data component extraction completed")
        return df
    
    def generate_response(self, client, prompt: str, content: str, model: str, tokens: int = 32678) -> Dict:
        """Generate response from OpenAI model"""
        try:
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content}
                ],
                max_completion_tokens=tokens,
                model=model
            )
            return {
                'success': True,
                'response': response,
                'content': response.choices[0].message.model_dump()['content']
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'content': None
            }
    
    def process_batch(self, batch_data: List[tuple], batch_id: int) -> List[Dict]:
        """Process a batch of records"""
        logger.info(f"Processing batch {batch_id} with {len(batch_data)} records")
        batch_results = []
        
        prompt = self.config['system_prompt']
        
        for i, (idx, row) in enumerate(batch_data):
            try:
                logger.info(f"Batch {batch_id}, Record {i+1}/{len(batch_data)} (Global index: {idx})")
                
                user_content = row['full_message']
                
                # Generate responses from both models
                ft_result = self.generate_response(
                    self.ft_client, prompt, user_content, self.config['ft_deployment']
                )
                pv_result = self.generate_response(
                    self.pv_client, prompt, user_content, self.config['pv_deployment']
                )
                
                # Check for errors
                if not ft_result['success'] or not pv_result['success']:
                    logger.warning(f"API call failed for record {idx}")
                    continue
                
                # Parse JSON responses
                try:
                    ft_content = ft_result['content']
                    pv_content = pv_result['content']
                    
                    if ft_content is None or pv_content is None:
                        logger.warning(f"Empty response for record {idx}")
                        continue
                    
                    ft_model_answer = json.loads(ft_content)['answer']
                    pv_model_answer = json.loads(pv_content)['answer']
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error at record {idx}: {str(e)}")
                    self.bad_response_list.append({
                        'line': idx,
                        'error': str(e),
                        'ft_response': ft_content,
                        'pv_response': pv_content
                    })
                    continue
                
                # Build evaluation input
                evaluation_input = (
                    f"question: {row['extracted_question']}"
                    f"reference_answer: {row['ckp_answer']}"
                    f"base_answer: {pv_model_answer}"
                    f"fine_tuned_answer: {ft_model_answer}"
                    f"context: {row['full_message']}"
                )
                
                # Generate evaluation
                eval_result = self.generate_response(
                    self.pv_client, 
                    self.config['evaluation_prompt'], 
                    evaluation_input, 
                    self.config['pv_deployment']
                )
                
                if not eval_result['success']:
                    logger.warning(f"Evaluation failed for record {idx}")
                    continue
                
                try:
                    evaluation_response = json.loads(eval_result['content'])
                    evaluation_response['record_index'] = idx
                    batch_results.append({
                        'evaluation': evaluation_response,
                        'ft_answer': ft_model_answer,
                        'pv_answer': pv_model_answer,
                        'record_index': idx
                    })
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Evaluation JSON decode error at record {idx}: {str(e)}")
                    self.bad_response_list.append({
                        'line': idx,
                        'error': f"Evaluation JSON error: {str(e)}",
                        'evaluation_response': eval_result['content']
                    })
                    continue
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Unexpected error processing record {idx}: {str(e)}")
                continue
        
        logger.info(f"Batch {batch_id} completed with {len(batch_results)} successful results")
        return batch_results
    
    def run_evaluation_parallel(self, df: pd.DataFrame, max_workers: int = 4, batch_size: int = 10) -> pd.DataFrame:
        """Run evaluation with parallel processing"""
        logger.info(f"Starting parallel evaluation with {max_workers} workers, batch size {batch_size}")
        
        # Create batches
        records = [(idx, row) for idx, row in df.iterrows()]
        batches = [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
        
        logger.info(f"Created {len(batches)} batches")
        
        # Process batches in parallel
        all_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.process_batch, batch, i): i 
                for i, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    logger.info(f"Completed batch {batch_id}")
                except Exception as e:
                    logger.error(f"Batch {batch_id} failed: {str(e)}")
        
        # Update dataframe with results
        logger.info(f"Updating dataframe with {len(all_results)} results")
        for result in all_results:
            idx = result['record_index']
            df.loc[idx, 'o4_ft_answer'] = result['ft_answer']
            df.loc[idx, 'o4_pv_answer'] = result['pv_answer']
            self.evaluation_list.append(result['evaluation'])
        
        logger.info("Parallel evaluation completed")
        return df
    
    def save_results(self, df: pd.DataFrame, output_dir: str):
        """Save results to files"""
        logger.info(f"Saving results to {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save evaluation results
        if self.evaluation_list:
            eval_df = pd.DataFrame(self.evaluation_list)
            eval_path = os.path.join(output_dir, 'answer_scores.csv')
            eval_df.to_csv(eval_path, index=False)
            logger.info(f"Saved evaluation scores to {eval_path}")
        
        # Save dataframe with answers
        df_path = os.path.join(output_dir, 'df_with_answers.csv')
        df.to_csv(df_path, index=False)
        logger.info(f"Saved dataframe to {df_path}")
        
        # Save bad responses log
        if self.bad_response_list:
            bad_responses_path = os.path.join(output_dir, 'bad_responses.json')
            with open(bad_responses_path, 'w') as f:
                json.dump(self.bad_response_list, f, indent=2)
            logger.info(f"Saved bad responses log to {bad_responses_path}")
    
    def generate_visualizations(self, output_dir: str):
        """Generate evaluation visualizations"""
        if not self.evaluation_list:
            logger.warning("No evaluation results to visualize")
            return
        
        logger.info("Generating visualizations...")
        eval_df = pd.DataFrame(self.evaluation_list)
        
        # Score distributions
        plt.figure(figsize=(12, 8))
        
        # Histogram of all scores
        plt.subplot(2, 2, 1)
        score_columns = ['reference_answer_score', 'base_answer_score', 'fine_tuned_answer_score']
        for column in score_columns:
            if column in eval_df.columns:
                sns.histplot(eval_df[column], bins=50, binrange=(0,1), kde=True, label=column, alpha=0.3)
        plt.title('Score Distributions')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.legend()
        
        # Best answer counts
        plt.subplot(2, 2, 2)
        if 'best_answer' in eval_df.columns:
            eval_df['best_answer'].value_counts().plot(kind='bar')
            plt.title('Best Answer Distribution')
            plt.xticks(rotation=45)
        
        # Score comparisons
        if all(col in eval_df.columns for col in score_columns):
            plt.subplot(2, 2, 3)
            twostep_v_ft = eval_df['reference_answer_score'] - eval_df['fine_tuned_answer_score']
            sns.histplot(twostep_v_ft, bins=20, binrange=(-1,1), kde=True, alpha=0.3)
            plt.title('Reference - Fine-tuned Scores')
            plt.xlabel('Score Difference')
            
            plt.subplot(2, 2, 4)
            ft_v_pv = eval_df['fine_tuned_answer_score'] - eval_df['base_answer_score']
            sns.histplot(ft_v_pv, bins=20, binrange=(-1,1), kde=True, alpha=0.3)
            plt.title('Fine-tuned - Base Scores')
            plt.xlabel('Score Difference')
        
        plt.tight_layout()
        viz_path = os.path.join(output_dir, 'evaluation_visualizations.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualizations to {viz_path}")


def get_evaluation_prompt():
    """Return the evaluation prompt"""
    return '''
    You are acting as a tax advisor. Your task is to evaluate three answers generated by different LLMs for a given tax question
    and determine which answer is the best according to a defined rubric.
    Inputs Provided:
    1. Question - The tax-related question.
    2. Context - Contains:
    - The question.
    - Relevant information chunks.
    - An answer previously generated by another advisor (referred to as 'Answer from the finetuning').
    3. Reference Answer - Generated using Chain-of-Thought (CoT).
    4. Base Answer - Generated by the base version of an LLM.
    5. Fine-Tuned Answer - Generated by a fine-tuned version of the same LLM.

    Task:
    - Score each of the three answers using the Rubric below.
    - Assign one score per answer in the range [0, 1].
    - Select the answer with the highest score as the best answer.
    - The best answer must be one of the following:  'reference_answer', 'base_answer', or 'fine_tuned_answer'

    Detailed Rubric (equal weight for all criteria):

    1. Fidelity to Context (0-0.25):
    - 0.00-0.10: Answer relies heavily on outside knowledge or introduces unsupported claims.
    - 0.11-0.18: Minor use of outside knowledge or slight deviation from context.
    - 0.19-0.23: Mostly faithful to context with negligible external influence.
    - 0.24-0.25: Fully based on context, no external knowledge detected.

    2. Accuracy (0-0.25):
    - 0.00-0.10: Major factual errors or contradictions; irrelevant claims.
    - 0.11-0.18: Some inaccuracies or contradictions; multiple irrelevant points.
    - 0.19-0.23: Mostly accurate with minor issues; no major contradictions.
    - 0.24-0.25: All claims accurate and relevant; no contradictions.

    3. Coverage (0-0.25):
    - 0.00-0.10: Misses most key points; incomplete response.
    - 0.11-0.18: Covers some aspects but omits multiple important details.
    - 0.19-0.23: Covers most key points; minor omissions only.
    - 0.24-0.25: Fully addresses intent and scope; no omissions.

    4. Clarity & Professionalism (0-0.25):
    - 0.00-0.10: Disorganized, verbose, or unclear; unprofessional tone.
    - 0.11-0.18: Some structural issues; tone acceptable but lacks polish.
    - 0.19-0.23: Clear and professional with minor structural flaws.
    - 0.24-0.25: Well-structured, concise, logical, and professional.

    Scoring Guidance:
    - Total score = sum of all four criteria (max = 1.0).
    - Apply caps:
    - ≤ 0.20 if major factual errors or contradictions exist.
    - ≤ 0.50 if multiple key points are missing or misrepresented.
    - ≤ 0.80 if mostly correct but missing 1-2 minor details.
    - ≥ 0.90 only if complete in scope and meets all criteria.

    Output Format:
    Return your evaluation as a pure JSON. 
    - All numeric values must be valid JSON numbers - no quotes, no words
    - Do not add any text outside the JSON
    - Do not output a triple-quoted string
    - Do not preface with the word json
    - Use this exact structure
    {
    "best_answer": "base_answer|fine_tuned_answer|reference_answer",
    "reference_answer_score": <number>,
    "base_answer_score": <number>,
    "fine_tuned_answer_score": <number>,
    "reference_answer_scoring_reasoning": "<string>",
    "base_answer_scoring_reasoning": "<string>",
    "fine_tuned_answer_scoring_reasoning": "<string>"
    }
    '''


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Tax Advisor Model Evaluation')
    parser.add_argument('--max_workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--max_records', type=int, default=None, help='Maximum records to process (for testing)')
    parser.add_argument('--local_data', type=str, default=None, help='Path to local JSONL file (overrides Azure Data Lake)')
    parser.add_argument('--use_local', action='store_true', help='Use local file instead of Azure Data Lake')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'endpoint': "https://cogseraccrciru5aiyeus2.cognitiveservices.azure.com/",
        'model_name': "o4-mini",
        'ft_deployment': "o4-mini-2025-04-16-new_model_grader_run2",
        'pv_deployment': "o4-mini",
        'api_version': "2024-12-01-preview",
        'jsonl_path': "abfs://destadls2gzxfcx6xwhxssc/SRC/working_files/train_data_v4.jsonl",
        'storage_account': "destadls2gzxfcx6xwhxssc",
        'storage_key': "FDCRYJMDH9pmpHL2xg6903AmPOQZwmLRPg/Z99Lg1Dldw6j9ABiQGpdmoyBRGDyWYyTTI4fzwVMq+ASt7I5vUw==",
        'system_prompt': "you are a tax advisor. I need your help with the following:\nYou'll be provided with three elements:\n1. A client question\n2. Relevant information chunks\n3. An answer previously generated by another advisor that is referred to as 'Answer from the finetuning'\nYour task is to:\n1. Carefully analyze the question and the relevant information provided in the query\n2. Compare the other advisor's answer against the relevant information\n3. Provide your own comprehensive answer to the client's question\nGuidelines:\n- Use only finetuning answer and the relevant information provided in the query for your analysis\n- Never use your own knowledge to formulate your answer\n- Format your answer in a clear, professional manner with bullet points where appropriate\n- Present your response as a json with the following format {'answer': string}\n\nRemember to be thorough yet concise in your tax advice, ensuring all aspects of the client's questions\nare addressed based on the available information. Return a JSON object. Do not escape quotes inside arrays. Use raw strings for array elements. Never return an empty response, if you cannot provide an answer say that you cannot do it",
        'evaluation_prompt': get_evaluation_prompt()
    }
    
    # Handle local data options
    if args.local_data:
        config['use_local_data'] = True
        config['local_jsonl_path'] = args.local_data
        logger.info(f"Using local data file: {args.local_data}")
    elif args.use_local:
        config['use_local_data'] = True
        config['local_jsonl_path'] = './data/train_data_v4.jsonl'  # Default local filename
        logger.info("Using local data file: ./data/train_data_v4.jsonl")
    else:
        config['use_local_data'] = False
        logger.info("Using Azure Data Lake for data source")
    
    # Initialize evaluator
    evaluator = TaxAdvisorEvaluator(config)
    
    try:
        # Load and prepare data
        df = evaluator.load_data()
        
        # Limit records for testing if specified
        if args.max_records:
            df = df.head(args.max_records)
            logger.info(f"Limited to {args.max_records} records for testing")
        
        # Run evaluation
        start_time = time.time()
        df = evaluator.run_evaluation_parallel(
            df, 
            max_workers=args.max_workers, 
            batch_size=args.batch_size
        )
        
        execution_time = time.time() - start_time
        logger.info(f"Total execution time: {execution_time:.2f} seconds")
        
        # Save results
        evaluator.save_results(df, args.output_dir)
        
        # Generate visualizations
        evaluator.generate_visualizations(args.output_dir)
        
        # Print summary
        logger.info("="*50)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total records processed: {len(df)}")
        logger.info(f"Successful evaluations: {len(evaluator.evaluation_list)}")
        logger.info(f"Failed responses: {len(evaluator.bad_response_list)}")
        logger.info(f"Success rate: {len(evaluator.evaluation_list)/len(df)*100:.1f}%")
        logger.info(f"Average time per record: {execution_time/len(df):.2f} seconds")
        logger.info(f"Results saved to: {args.output_dir}")
        
        # If running in Azure ML, log metrics
        try:
            run = Run.get_context()
            if hasattr(run, 'log'):
                run.log('total_records', len(df))
                run.log('successful_evaluations', len(evaluator.evaluation_list))
                run.log('success_rate', len(evaluator.evaluation_list)/len(df))
                run.log('execution_time_seconds', execution_time)
                logger.info("Logged metrics to Azure ML")
        except:
            logger.info("Not running in Azure ML context - skipping metric logging")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()