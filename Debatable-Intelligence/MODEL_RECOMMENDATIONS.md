# Recommended HuggingFace Models for HPC

## Quick Reference by GPU Memory

### 🟢 24GB GPU (e.g., RTX 3090, RTX 4090, A10)
**Without Quantization:**
- `meta-llama/Llama-2-7b-chat-hf` ✅
- `mistralai/Mistral-7B-Instruct-v0.2` ✅
- `microsoft/phi-2` (2.7B) ✅
- `Qwen/Qwen2.5-7B-Instruct` ✅
- `google/gemma-7b-it` ✅

**With 4-bit Quantization:**
- `meta-llama/Llama-2-13b-chat-hf` ✅
- `mistralai/Mixtral-8x7B-Instruct-v0.1` ⚠️ (tight fit)

### 🟡 40GB GPU (e.g., A100-40GB, A6000)
**Without Quantization:**
- All 7B models ✅
- `meta-llama/Llama-2-13b-chat-hf` ✅
- `Qwen/Qwen2.5-14B-Instruct` ✅

**With 4-bit Quantization:**
- `meta-llama/Meta-Llama-3-70B-Instruct` ✅
- `mistralai/Mixtral-8x7B-Instruct-v0.1` ✅
- `Qwen/Qwen2.5-32B-Instruct` ✅

### 🔵 80GB GPU (e.g., A100-80GB, H100)
**Without Quantization:**
- All 7B-13B models ✅
- `meta-llama/Meta-Llama-3-70B-Instruct` ⚠️ (tight fit)
- `mistralai/Mixtral-8x7B-Instruct-v0.1` ✅

**With 4-bit Quantization:**
- `meta-llama/Meta-Llama-3-70B-Instruct` ✅ (comfortable)
- `Qwen/Qwen2.5-72B-Instruct` ✅
- `meta-llama/Llama-2-70b-chat-hf` ✅

## Model Recommendations by Task

### For Math Problems (`config_math.json`)
**Best Models:**
1. `deepseek-ai/deepseek-math-7b-instruct` - Specialized for math ⭐
2. `meta-llama/Meta-Llama-3-8B-Instruct` - Strong reasoning
3. `Qwen/Qwen2.5-Math-7B-Instruct` - Math-specific ⭐
4. `mistralai/Mistral-7B-Instruct-v0.2` - Good general reasoning

### For Open QA (`config_openqa.json`)
**Best Models:**
1. `meta-llama/Meta-Llama-3-8B-Instruct` - Comprehensive answers
2. `mistralai/Mistral-7B-Instruct-v0.2` - Balanced quality
3. `Qwen/Qwen2.5-7B-Instruct` - Strong instruction following
4. `microsoft/Phi-3-medium-4k-instruct` - Efficient

### For Medical Questions (`config_med.json`)
**Best Models:**
1. `meta-llama/Meta-Llama-3-8B-Instruct` - Detailed explanations
2. `mistralai/Mistral-7B-Instruct-v0.2` - Safe, balanced
3. `Qwen/Qwen2.5-7B-Instruct` - Good comprehension
4. `meta-llama/Llama-2-7b-chat-hf` - Reliable baseline

⚠️ **Note**: Avoid using models not specifically trained on medical data for real medical advice!

## Configuration Examples

### Conservative Setup (24GB GPU)
```json
{
  "models": {
    "huggingface": [
      "mistralai/Mistral-7B-Instruct-v0.2"
    ]
  }
}
```

### Balanced Setup (40GB GPU)
```json
{
  "models": {
    "huggingface": [
      "meta-llama/Llama-2-7b-chat-hf",
      "mistralai/Mistral-7B-Instruct-v0.2",
      "Qwen/Qwen2.5-7B-Instruct"
    ]
  }
}
```

### Comprehensive Setup (80GB GPU with quantization)
```json
{
  "models": {
    "huggingface": [
      "mistralai/Mistral-7B-Instruct-v0.2",
      "meta-llama/Meta-Llama-3-8B-Instruct",
      "Qwen/Qwen2.5-14B-Instruct",
      "Qwen/Qwen2.5-72B-Instruct"
    ]
  }
}
```

## Quantization Guide

The framework **automatically applies 4-bit quantization** for models > 10B parameters.

### Manual Quantization Settings (in model.py)
```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Use 4-bit precision
    bnb_4bit_quant_type="nf4",           # NormalFloat4 (best quality)
    bnb_4bit_compute_dtype=torch.float16, # Compute in FP16
    bnb_4bit_use_double_quant=True       # Double quantization for memory
)
```

### Memory Reduction Estimates
- **8-bit quantization**: ~50% memory reduction
- **4-bit quantization**: ~75% memory reduction
- **Example**: 70B model
  - FP16: ~140GB
  - 8-bit: ~70GB
  - 4-bit: ~35GB ✅ Fits on A100-40GB!

## License Requirements

### ⚠️ Gated Models (Require HuggingFace Token + Agreement)
- `meta-llama/Llama-2-*` - Accept Meta license
- `meta-llama/Meta-Llama-3-*` - Accept Meta license
- `mistralai/Mixtral-*` - Accept Mistral license

### ✅ Open Models (No Restrictions)
- `mistralai/Mistral-7B-Instruct-v0.2`
- `Qwen/Qwen2.5-*`
- `microsoft/phi-*`
- `google/gemma-*`

## Checking Model Availability

Before adding a model to your config, verify:

1. **Check HuggingFace Hub**: https://huggingface.co/[model-name]
2. **Check if gated**: Look for "Access repository" button
3. **Request access**: Click button and wait for approval (usually instant)
4. **Get token**: https://huggingface.co/settings/tokens
5. **Save token**: Add to `secret_keys/huggingface_key`

## Performance Tips

1. **Start small**: Test with 7B models first
2. **Monitor memory**: Use `nvidia-smi` to watch GPU memory
3. **Batch size = 1**: For inference, framework uses batch size 1
4. **Use quantization**: Enable for models >10B to save memory
5. **Clean cache**: Run `torch.cuda.empty_cache()` between runs if needed

## Common Issues

### "CUDA out of memory"
- ✅ Use smaller model
- ✅ Enable quantization
- ✅ Restart kernel/session

### "Model not found"
- ✅ Check model name spelling
- ✅ Ensure you have access (gated models)
- ✅ Verify HuggingFace token is valid

### "Quantization not working"
- ✅ Install: `pip install bitsandbytes accelerate`
- ✅ Check CUDA is available: `torch.cuda.is_available()`
- ✅ Verify CUDA version compatibility

## Model Comparison Table

| Model | Size | GPU Mem (FP16) | GPU Mem (4-bit) | Speed | Quality |
|-------|------|----------------|-----------------|-------|---------|
| Mistral-7B | 7B | ~14GB | ~4GB | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| Llama-2-7b | 7B | ~14GB | ~4GB | ⚡⚡⚡ | ⭐⭐⭐ |
| Llama-3-8B | 8B | ~16GB | ~5GB | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| Qwen2.5-7B | 7B | ~14GB | ~4GB | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| Llama-2-13b | 13B | ~26GB | ~7GB | ⚡⚡ | ⭐⭐⭐⭐ |
| Mixtral-8x7B | 47B | ~94GB | ~24GB | ⚡ | ⭐⭐⭐⭐⭐ |
| Llama-3-70B | 70B | ~140GB | ~35GB | ⚡ | ⭐⭐⭐⭐⭐ |

Speed: ⚡ = Slow, ⚡⚡ = Medium, ⚡⚡⚡ = Fast  
Quality: ⭐ = Poor, ⭐⭐⭐ = Good, ⭐⭐⭐⭐⭐ = Excellent
