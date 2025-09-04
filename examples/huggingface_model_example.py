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

"""Example script demonstrating how to use the HuggingFace language model."""

from concordia.language_model import huggingface_model


def main():
  """Example usage of HuggingFaceLanguageModel."""

  # Note: You need to install PyTorch and transformers first:
  # pip install torch transformers

  try:
    # Initialize the model with a small Hugging Face model
    # Examples of models you can use:
    # - 'microsoft/DialoGPT-small' (conversational)
    # - 'microsoft/Phi-3-mini-4k-instruct' (instruction following)
    # - 'google/flan-t5-small' (text-to-text)
    # - 'HuggingFaceH4/zephyr-7b-beta' (chat model)

    model = huggingface_model.HuggingFaceLanguageModel(
        model_name='microsoft/DialoGPT-small',  # A small model for testing
        system_message='You are a helpful assistant.',
        device='cpu',  # Use 'cuda' if you have a GPU
        load_in_8bit=False,  # Set to True for memory efficiency
        trust_remote_code=False,  # Set to True if the model requires it
    )

    # Test text generation
    print("Testing text generation:")
    prompt = "The weather today is"
    response = model.sample_text(
        prompt=prompt,
        max_tokens=50,
        temperature=0.7
    )
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    print()

    # Test choice selection using log probabilities
    print("Testing choice selection using log probabilities:")
    prompt = "What is the best programming language for beginners?"
    choices = ["Python", "JavaScript", "Java", "C++"]

    choice_idx, chosen_option, log_probs = model.sample_choice(
        prompt=prompt,
        responses=choices
    )

    print(f"Prompt: {prompt}")
    print(f"Choices: {choices}")
    print(f"Selected: {chosen_option} (index: {choice_idx})")
    print("Log probabilities for each choice:")
    for choice, log_prob in log_probs.items():
      print(f"  {choice}: {log_prob:.4f}")
    print()

    # Test another choice selection example
    print("Testing choice selection with Yes/No question:")
    prompt = "Is the sky blue?"
    yes_no_choices = ["Yes", "No"]

    choice_idx, chosen_option, log_probs = model.sample_choice(
        prompt=prompt,
        responses=yes_no_choices
    )

    print(f"Prompt: {prompt}")
    print(f"Choices: {yes_no_choices}")
    print(f"Selected: {chosen_option} (index: {choice_idx})")
    print("Log probabilities for each choice:")
    for choice, log_prob in log_probs.items():
      print(f"  {choice}: {log_prob:.4f}")
    print()

  except ImportError as e:
    print(f"Error: {e}")
    print("Please install the required dependencies:")
    print("pip install torch transformers")
  except Exception as e:
    print(f"Error initializing or using the model: {e}")
    print("This might be due to:")
    print("1. Model not found on Hugging Face Hub")
    print("2. Insufficient memory")
    print("3. Network connectivity issues")
    print("4. Model requires trust_remote_code=True")


if __name__ == '__main__':
  main()
