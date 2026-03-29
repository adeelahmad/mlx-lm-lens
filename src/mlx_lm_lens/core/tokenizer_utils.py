"""Tokenizer wrapper with chat template support."""

import logging
from typing import Any, Dict, List, Optional

from mlx_lm.utils import load

logger = logging.getLogger(__name__)


def load_tokenizer(model_path: str) -> Any:
  """Load tokenizer from model path.

  Args:
    model_path: Path to MLX model or HuggingFace model ID.

  Returns:
    Tokenizer instance (wrapped for compatibility).
  """
  _, tokenizer = load(model_path, lazy=True)
  return TokenizerWrapper(tokenizer) if tokenizer else None


class TokenizerWrapper:
  """Wrapper around tokenizer with graceful fallbacks."""

  def __init__(self, tokenizer: Any) -> None:
    """Initialize tokenizer wrapper.

    Args:
      tokenizer: Tokenizer from mlx_lm.utils.load().
    """
    self.tokenizer = tokenizer

  def encode(self, text: str) -> List[int]:
    """Encode text to token IDs.

    Args:
      text: Text to encode.

    Returns:
      List of token IDs.

    Raises:
      AttributeError: If tokenizer has no encode method.
    """
    return self.tokenizer.encode(text)

  def decode(self, ids: List[int]) -> str:
    """Decode token IDs to text.

    Args:
      ids: List of token IDs.

    Returns:
      Decoded text.
    """
    if not ids:
      return ""
    if not hasattr(self.tokenizer, "decode"):
      return ""
    return self.tokenizer.decode(ids, skip_special_tokens=True)

  def apply_chat_template(
    self,
    messages: List[Dict[str, str]],
    chat_template: Optional[str] = None,
  ) -> str:
    """Apply chat template to messages with graceful fallback.

    If template is missing or fails, returns raw message text.

    Args:
      messages: List of {role, content} dicts.
      chat_template: Optional explicit template string.

    Returns:
      Formatted chat string.
    """
    if not hasattr(self.tokenizer, "apply_chat_template"):
      logger.warning(
        "Tokenizer has no apply_chat_template; using raw message text"
      )
      return self._raw_messages_to_text(messages)

    try:
      result = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=chat_template,
      )
      return result if result else self._raw_messages_to_text(messages)
    except Exception as e:
      logger.warning(
        f"Chat template failed ({e}); using raw message text"
      )
      return self._raw_messages_to_text(messages)

  @staticmethod
  def _raw_messages_to_text(messages: List[Dict[str, str]]) -> str:
    """Fallback: concatenate messages as plain text.

    Args:
      messages: List of {role, content} dicts.

    Returns:
      Concatenated message text.
    """
    lines = []
    for msg in messages:
      content = msg.get("content", "")
      if content:
        lines.append(content)
    return "\n".join(lines)
