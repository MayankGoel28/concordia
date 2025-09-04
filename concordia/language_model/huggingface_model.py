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

"""Hugging Face Language Model, a wrapper for instruct models from Hugging Face."""

from collections.abc import Collection, Sequence
from typing import Any

from concordia.language_model import language_model
from concordia.utils import sampling
from concordia.utils.deprecated import measurements as measurements_lib
from typing_extensions import override


_MAX_MULTIPLE_CHOICE_ATTEMPTS = 20
_DEFAULT_TEMPERATURE = 0.5
_DEFAULT_TERMINATORS = ()
_DEFAULT_SYSTEM_MESSAGE = (
    'Continue the user\'s sentences. Never repeat their starts. For example, '
    'when you see \'Bob is\', you should continue the sentence after '
    'the word \'is\'. Here are some more examples: \'Question: Is Jake a '
    'turtle?\nAnswer: Jake is \' should be completed as \'not a turtle.\' and '
    '\'Question: What is Priya doing right now?\nAnswer: Priya is currently \' '
    'should be completed as \'working on repairing the sink.\'. Notice that '
    'it is OK to be creative with how you finish the user\'s sentences. The '
    'most important thing is to always continue in the same style as the user.'
)


def _check_dependencies():
  """Check if required dependencies are available."""
  try:
    import torch  # pylint: disable=import-outside-toplevel,unused-import
    import transformers  # pylint: disable=import-outside-toplevel,unused-import
    return True, None
  except ImportError as e:
    return False, str(e)


class HuggingFaceLanguageModel(language_model.LanguageModel):
  """Language Model that uses Hugging Face instruct models."""

  def __init__(
      self,
      model_name: str,
      *,
      system_message: str = _DEFAULT_SYSTEM_MESSAGE,
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      device: str | None = None,
      load_in_8bit: bool = False,
      load_in_4bit: bool = False,
      trust_remote_code: bool = False,
      torch_dtype: Any = None,
  ) -> None:
    """Initializes the instance.

    Args:
        model_name: The Hugging Face model name to use (e.g.,
          'microsoft/DialoGPT-medium', 'microsoft/Phi-3-mini-4k-instruct').
        system_message: System message to prefix to requests when prompting the
          model.
        measurements: The measurements object to log usage statistics to.
        channel: The channel to write the statistics to.
        device: Device to load the model on ('cuda', 'cpu', etc.). If None,
          automatically selects cuda if available.
        load_in_8bit: Whether to load the model in 8-bit precision.
        load_in_4bit: Whether to load the model in 4-bit precision.
        trust_remote_code: Whether to trust remote code when loading the model.
        torch_dtype: The torch dtype to use for the model (e.g., torch.float16).
    """
    deps_available, error_msg = _check_dependencies()
    if not deps_available:
      raise ImportError(
          f"PyTorch and Transformers are required for HuggingFaceLanguageModel. "
          f"Please install them using: pip install torch transformers. "
          f"Import error: {error_msg}"
      )

    # Import here to avoid errors when dependencies are not available
    import torch  # pylint: disable=import-outside-toplevel
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  # pylint: disable=import-outside-toplevel

    self._model_name = model_name
    self._system_message = system_message
    self._terminators = []

    # Set default torch_dtype if not provided
    if torch_dtype is None:
      torch_dtype = torch.float16

    # Set device
    if device is None:
      self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
      self._device = device

    # Configure quantization if requested
    quantization_config = None
    if load_in_4bit:
      quantization_config = BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_compute_dtype=torch_dtype,
          bnb_4bit_use_double_quant=True,
          bnb_4bit_quant_type="nf4"
      )
    elif load_in_8bit:
      quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load tokenizer and model
    try:
      self._tokenizer = AutoTokenizer.from_pretrained(
          model_name,
          trust_remote_code=trust_remote_code
      )

      # Set pad token if not present
      if self._tokenizer.pad_token is None:
        self._tokenizer.pad_token = self._tokenizer.eos_token

      self._model = AutoModelForCausalLM.from_pretrained(
          model_name,
          quantization_config=quantization_config,
          torch_dtype=torch_dtype,
          trust_remote_code=trust_remote_code,
          device_map='auto' if self._device == 'cuda' else None,
      )

      if self._device == 'cpu':
        self._model = self._model.to(self._device)

    except Exception as e:
      raise ValueError(f"Failed to load model {model_name}: {e}")

    self._measurements = measurements
    self._channel = channel

  def _format_prompt(self, prompt: str) -> str:
    """Format the prompt using the model's chat template if available."""
    # Try to use the chat template if available
    if hasattr(self._tokenizer, 'chat_template') and self._tokenizer.chat_template:
      messages = [
          {"role": "system", "content": self._system_message},
          {"role": "user", "content": prompt}
      ]
      try:
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
      except Exception:
        # Fallback to simple concatenation if chat template fails
        pass

    # Fallback formatting
    return f"{self._system_message}\n\nUser: {prompt}\nAssistant:"

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      terminators: Collection[str] = _DEFAULT_TERMINATORS,
      temperature: float = _DEFAULT_TEMPERATURE,
      timeout: float = -1,
      seed: int | None = None,
  ) -> str:
    del timeout  # Unused for local models

    import torch  # pylint: disable=import-outside-toplevel

    if seed is not None:
      torch.manual_seed(seed)

    formatted_prompt = self._format_prompt(prompt)
    terminators_list = self._terminators + list(terminators)

    # Tokenize input
    inputs = self._tokenizer.encode(formatted_prompt, return_tensors="pt").to(self._device)
    input_length = inputs.shape[1]

    # Generate response
    with torch.no_grad():
      outputs = self._model.generate(
          inputs,
          max_new_tokens=max_tokens,
          temperature=temperature,
          do_sample=temperature > 0,
          pad_token_id=self._tokenizer.eos_token_id,
          eos_token_id=self._tokenizer.eos_token_id,
          repetition_penalty=1.1,
      )

    # Decode only the generated part (exclude input)
    generated_tokens = outputs[0][input_length:]
    result = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Apply terminators
    for terminator in terminators_list:
      if terminator in result:
        result = result.split(terminator)[0]

    result = result.strip()

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel,
          {'raw_text_length': len(result)})

    return result

  def _get_response_log_probability(self, prompt: str, response: str) -> float:
    """Calculate the log probability of a response given a prompt.

    Args:
      prompt: The input prompt
      response: The response choice to evaluate

    Returns:
      The sum of log probabilities for tokens in the response
    """
    import torch  # pylint: disable=import-outside-toplevel

    formatted_prompt = self._format_prompt(prompt)
    full_text = formatted_prompt + response

    # Tokenize the full text (prompt + response)
    full_tokens = self._tokenizer.encode(full_text, return_tensors="pt").to(self._device)
    prompt_tokens = self._tokenizer.encode(formatted_prompt, return_tensors="pt").to(self._device)

    # Get the response tokens (tokens that are only in the response)
    response_start_idx = prompt_tokens.shape[1]

    with torch.no_grad():
      # Get logits for the full sequence
      outputs = self._model(full_tokens)
      logits = outputs.logits

      # Convert logits to log probabilities
      log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

      # Calculate log probability for each response token
      response_log_prob = 0.0
      for i in range(response_start_idx, full_tokens.shape[1]):
        token_id = full_tokens[0, i].item()
        # Use logits from the previous position to predict current token
        if i > 0:
          token_log_prob = log_probs[0, i-1, token_id].item()
          response_log_prob += token_log_prob

    return response_log_prob

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    import torch  # pylint: disable=import-outside-toplevel

    if seed is not None:
      torch.manual_seed(seed)

    # Calculate log probabilities for each response
    log_probs = {}
    for response in responses:
      try:
        log_prob = self._get_response_log_probability(prompt, response)
        log_probs[response] = log_prob
      except Exception as e:
        # If we can't calculate log probability for a response, assign very low probability
        log_probs[response] = float('-inf')
        if self._measurements is not None:
          self._measurements.publish_datum(
              self._channel, {'log_prob_calculation_error': str(e)}
          )

    # Find the response with the highest log probability
    if not log_probs or all(prob == float('-inf') for prob in log_probs.values()):
      raise language_model.InvalidResponseError(
          "Could not calculate log probabilities for any response choices"
      )

    best_response = max(log_probs.keys(), key=lambda r: log_probs[r])
    best_idx = responses.index(best_response)

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel, {
              'choice_selection_method': 'log_probability',
              'log_probs': log_probs
          }
      )

    return best_idx, best_response, log_probs
