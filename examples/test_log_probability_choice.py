#!/usr/bin/env python3
"""Test script to verify the log probability-based choice selection."""

from concordia.language_model import huggingface_model


def test_log_probability_choice_selection():
  """Test the log probability-based choice selection method."""
  try:
    # Use a very small model for testing
    model = huggingface_model.HuggingFaceLanguageModel(
        model_name='gpt2',  # Small model available everywhere
        system_message='',  # Minimal system message for testing
        device='cpu',
    )

    print("Testing log probability-based choice selection...")

    # Test with a simple factual question
    prompt = "The capital of France is"
    choices = ["Paris", "London", "Berlin", "Madrid"]

    choice_idx, chosen_option, log_probs = model.sample_choice(
        prompt=prompt,
        responses=choices
    )

    print(f"Prompt: {prompt}")
    print(f"Choices: {choices}")
    print(f"Selected: {chosen_option} (index: {choice_idx})")
    print("Log probabilities:")
    for choice, log_prob in log_probs.items():
      print(f"  {choice}: {log_prob:.6f}")

    # Verify that the log probabilities make sense
    assert len(log_probs) == len(choices), "Should have log prob for each choice"
    assert chosen_option in choices, "Chosen option should be in choices"
    assert log_probs[chosen_option] == max(log_probs.values()), "Chosen option should have highest log prob"

    print("\n✓ Log probability-based choice selection working correctly!")

  except ImportError as e:
    print(f"Dependencies not available: {e}")
    print("This test requires: pip install torch transformers")
    return False
  except Exception as e:
    print(f"Test failed: {e}")
    return False

  return True


if __name__ == '__main__':
  test_log_probability_choice_selection()
