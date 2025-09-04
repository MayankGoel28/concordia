#!/usr/bin/env python3
# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example script demonstrating how to use the HuggingFace language model through utils."""

from concordia.language_model import utils


def main():
  """Example usage of HuggingFaceLanguageModel through utils setup."""

  try:
    # Use the utils function to set up a Hugging Face model
    model = utils.language_model_setup(
        api_type='huggingface',
        model_name='microsoft/DialoGPT-small',  # A small model for testing
        device='cpu',  # Use 'cuda' if you have a GPU
    )

    print("Successfully created HuggingFace model through utils!")
    print(f"Model type: {type(model).__name__}")

    # Test text generation
    print("\nTesting text generation:")
    prompt = "The weather today is"
    response = model.sample_text(
        prompt=prompt,
        max_tokens=30,
        temperature=0.7
    )
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

    # Test choice selection
    print("\nTesting choice selection:")
    prompt = "What is the best programming language for beginners?"
    choices = ["Python", "JavaScript", "Java"]

    choice_idx, chosen_option, debug_info = model.sample_choice(
        prompt=prompt,
        responses=choices
    )

    print(f"Prompt: {prompt}")
    print(f"Choices: {choices}")
    print(f"Selected: {chosen_option} (index: {choice_idx})")

  except ImportError as e:
    print(f"Import Error: {e}")
    print("Please install the required dependencies:")
    print("pip install torch transformers")
  except ValueError as e:
    print(f"Configuration Error: {e}")
  except Exception as e:
    print(f"Error: {e}")
    print("This might be due to:")
    print("1. Model not found on Hugging Face Hub")
    print("2. Insufficient memory")
    print("3. Network connectivity issues")


def test_api_types():
  """Test that all API types are recognized."""
  print("Testing API type recognition...")

  # List of valid API types
  valid_api_types = [
      'amazon_bedrock',
      'google_aistudio_model',
      'google_cloud_custom_model',
      'huggingface',  # Our new addition
      'langchain_ollama',
      'mistral',
      'ollama',
      'openai',
      'pytorch_gemma',
      'together_ai'
  ]

  for api_type in valid_api_types:
    try:
      # Just test that the API type is recognized (will fail on missing dependencies/keys)
      utils.language_model_setup(
          api_type=api_type,
          model_name='dummy-model'
      )
      print(f"✓ {api_type}: Recognized")
    except ValueError as e:
      if 'Unrecognized api type' in str(e):
        print(f"✗ {api_type}: NOT recognized")
      else:
        print(f"✓ {api_type}: Recognized (failed for other reasons)")
    except Exception:
      print(f"✓ {api_type}: Recognized (failed for other reasons)")

  # Test an invalid API type
  try:
    utils.language_model_setup(
        api_type='invalid_api_type',
        model_name='dummy-model'
    )
    print("✗ invalid_api_type: Should have been rejected")
  except ValueError as e:
    if 'Unrecognized api type' in str(e):
      print("✓ invalid_api_type: Correctly rejected")
    else:
      print(f"? invalid_api_type: Unexpected error: {e}")


if __name__ == '__main__':
  print("=== Testing HuggingFace Model Integration ===")
  main()
  print("\n=== Testing API Type Recognition ===")
  test_api_types()
