# %%
import pandas as pd
import os

def load_csv_safe(path):
    for enc in ('utf-8-sig', 'utf-8', 'cp949', 'euc-kr', 'latin-1'):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"❌ Failed to read {path} with common encodings")


def merge_model_outputs(base_path, llama_path, distill_path, output_path, domain_name, export_json=True):
    """
    Merge base dataset with llama and distill_llama outputs.
    
    Args:
        base_path: Path to original dataset (math.csv, med.csv, openQA.csv)
        llama_path: Path to llama inference output (llama_math.csv, etc.)
        distill_path: Path to distill_llama inference output
        output_path: Where to save merged CSV
        domain_name: 'math', 'med', or 'openQA' for logging
        export_json: If True, also export as JSON file
    """
    print(f"\n🔄 Merging {domain_name} datasets...")
    
    # Load base dataset
    if not os.path.exists(base_path):
        print(f"⚠️  Base dataset not found: {base_path}")
        return False
        
    base_df = load_csv_safe(base_path)
    print(f"  Base dataset: {base_df.shape}")
    
    # Load llama output
    if os.path.exists(llama_path):
        llama_df = load_csv_safe(llama_path)
        print(f"  Llama dataset: {llama_df.shape}")
        
        # Merge llama_output column
        if 'llama_output' in llama_df.columns:
            base_df['llama_output'] = llama_df['llama_output']
            print(f"  ✅ Added llama_output column")
        else:
            print(f"  ⚠️  No 'llama_output' column found in {llama_path}")
    else:
        print(f"  ⚠️  Llama dataset not found: {llama_path}")
        base_df['llama_output'] = None
    
    # Load distill_llama output
    if os.path.exists(distill_path):
        distill_df = load_csv_safe(distill_path)
        print(f"  Distill dataset: {distill_df.shape}")
        
        # Merge distill_llama_output column
        if 'distill_llama_output' in distill_df.columns:
            base_df['distill_llama_output'] = distill_df['distill_llama_output']
            print(f"  ✅ Added distill_llama_output column")
        else:
            print(f"  ⚠️  No 'distill_llama_output' column found in {distill_path}")
    else:
        print(f"  ⚠️  Distill dataset not found: {distill_path}")
        base_df['distill_llama_output'] = None
    
    # Save merged dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  💾 Saved merged CSV: {output_path}")
    
    # Export JSON if requested
    if export_json:
        json_path = output_path.replace('.csv', '.json')
        base_df.to_json(json_path, orient='records', indent=2, force_ascii=False)
        print(f"  💾 Saved merged JSON: {json_path}")
    
    print(f"  Final shape: {base_df.shape}")
    print(f"  Columns: {list(base_df.columns)}")
    
    return True


# Main execution
if __name__ == "__main__":
    print("=" * 60)
    print("📊 Dataset Merging Script")
    print("=" * 60)
    
    # Define paths
    data_dir = '../data'
    
    # Math dataset
    merge_model_outputs(
        base_path=f'{data_dir}/math.csv',
        llama_path=f'{data_dir}/llama_math.csv',
        distill_path=f'{data_dir}/distill_llama_math.csv',
        output_path=f'{data_dir}/math_cleaned.csv',
        domain_name='Math'
    )
    
    # Medical dataset
    merge_model_outputs(
        base_path=f'{data_dir}/med.csv',
        llama_path=f'{data_dir}/llama_med.csv',
        distill_path=f'{data_dir}/distill_llama_med.csv',
        output_path=f'{data_dir}/med_cleaned.csv',
        domain_name='Medical'
    )
    
    # OpenQA dataset
    merge_model_outputs(
        base_path=f'{data_dir}/openQA.csv',
        llama_path=f'{data_dir}/llama_openQA.csv',
        distill_path=f'{data_dir}/distill_llama_openQA.csv',
        output_path=f'{data_dir}/openQA_cleaned.csv',
        domain_name='OpenQA'
    )
    
    print("\n" + "=" * 60)
    print("✅ Dataset merging complete!")
    print("=" * 60)
    print("\nMerged datasets created:")
    print(f"  • {data_dir}/math_cleaned.csv + math_cleaned.json")
    print(f"  • {data_dir}/med_cleaned.csv + med_cleaned.json")
    print(f"  • {data_dir}/openQA_cleaned.csv + openQA_cleaned.json")

