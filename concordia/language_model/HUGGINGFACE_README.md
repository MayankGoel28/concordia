# Hugging Face Language Model Integration

This directory now includes a new Hugging Face language model integration that allows you to use any Hugging Face instruct model with Concordia.

## Files Added

- `huggingface_model.py` - Main implementation of the Hugging Face language model wrapper
- `examples/huggingface_model_example.py` - Example usage demonstrating text generation and choice selection
- `examples/test_log_probability_choice.py` - Test script for verifying log probability-based choice selection

## Key Features

### Log Probability-Based Choice Selection
The `sample_choice` method uses log probabilities to measure the model's preference for each choice, providing a more direct and principled approach than JSON parsing. This follows the same pattern as the `DefaultCompletion` wrapper in `together_ai.py`.

### Robust Error Handling
The implementation gracefully handles missing dependencies (torch, transformers) and provides clear guidance for installation.

### Flexible Configuration
- Support for quantization (4-bit, 8-bit)
- Device selection (CPU/GPU)
- Customizable system messages
- Temperature and max token controls

## Usage

```python
from concordia.language_model import huggingface_model

# Initialize the model
model = huggingface_model.HuggingFaceLanguageModel(
    model_name='microsoft/DialoGPT-medium',
    system_message='You are a helpful assistant.',
    device='auto',
    quantization='4bit'
)

# Generate text
response = model.sample_text("What is the capital of France?")

# Make a choice with log probabilities
choice_idx, chosen_option, log_probs = model.sample_choice(
    prompt="The weather today is",
    responses=["sunny", "rainy", "cloudy"]
)
```

## Installation

To use the Hugging Face model, install the required dependencies:

```bash
pip install torch transformers accelerate bitsandbytes
```

## Integration with Language Model Setup

The Hugging Face model is now integrated into the `utils.py` setup utility and can be selected as a language model option in Concordia configurations.
