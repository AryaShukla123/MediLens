"""
Thin wrapper around the Gemini API (google-genai SDK).

The rest of the chatbot only ever calls one function:

    reply = generate_reply(system_prompt, history, user_message)

and only ever has to catch one exception type: ChatbotError, which carries a
message that is safe to show to the user. Everything Gemini-specific (SDK
types, error classes, blocked responses, quota errors) stays in this file, so
if Google changes something again, this is the only file that needs touching.

Notes:
  - The model name comes from settings.gemini_model (GEMINI_MODEL in .env).
  - google-generativeai (the SDK this project originally listed) is the legacy
    package; google-genai is its replacement.
  - Gemini 3.x models "think" before answering and thinking tokens count
    towards max_output_tokens. We ask for low thinking effort (this is a
    short explanatory chat, not a reasoning task) and leave a generous output
    budget so an answer is never starved by its own thinking.
"""

import logging
from typing import Optional

from google import genai
from google.genai import errors, types

from app.config import settings

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_MS = 30_000
MAX_OUTPUT_TOKENS = 2048
TEMPERATURE = 0.4

# Flipped to False for the life of the process if the API ever rejects the
# thinking setting (e.g. a future model that doesn't accept it), so we stop
# sending it instead of failing every request.
_send_thinking_level = True

_client: Optional[genai.Client] = None


class ChatbotError(Exception):
    """An error whose `user_message` is safe to show in the chat window."""

    def __init__(self, user_message: str, status_code: int = 502):
        super().__init__(user_message)
        self.user_message = user_message
        self.status_code = status_code


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


# ---------------------------------------------------------------------------
# Building the request
# ---------------------------------------------------------------------------

def _build_contents(history: list[dict], user_message: str) -> list[types.Content]:
    """
    Turns our stored messages ({"role": "user"|"assistant", "content": str})
    into Gemini's format, which has two rules we enforce here:
      - the roles are "user" and "model" (not "assistant")
      - the conversation must start with a user turn, and the request must end
        with one (the new message)
    Consecutive turns from the same role are merged rather than rejected.
    """
    turns: list[list[str]] = []  # [role, text]

    def add(role: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if turns and turns[-1][0] == role:
            turns[-1][1] += "\n\n" + text
        else:
            turns.append([role, text])

    for message in history:
        add("user" if message["role"] == "user" else "model", message["content"])

    while turns and turns[0][0] != "user":
        turns.pop(0)

    add("user", user_message)

    return [
        types.Content(role=role, parts=[types.Part(text=text)])
        for role, text in turns
    ]


def _call(contents: list[types.Content], system_prompt: str, with_thinking: bool):
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
        thinking_config=(
            types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
            if with_thinking
            else None
        ),
    )
    return _get_client().models.generate_content(
        model=settings.gemini_model,
        contents=contents,
        config=config,
    )


# ---------------------------------------------------------------------------
# The one public function
# ---------------------------------------------------------------------------

def generate_reply(system_prompt: str, history: list[dict], user_message: str) -> str:
    """
    Sends the conversation to Gemini and returns the assistant's reply text.
    Raises ChatbotError (with a user-safe message) for anything that goes wrong.
    """
    global _send_thinking_level

    contents = _build_contents(history, user_message)

    try:
        try:
            response = _call(contents, system_prompt, _send_thinking_level)
        except errors.ClientError as exc:
            if _send_thinking_level and exc.code == 400 and "think" in str(exc).lower():
                logger.warning(
                    "Model %s rejected the thinking setting; retrying without it. (%s)",
                    settings.gemini_model, exc,
                )
                _send_thinking_level = False
                response = _call(contents, system_prompt, False)
            else:
                raise
    except errors.APIError as exc:
        raise _map_api_error(exc) from exc
    except Exception as exc:  # network failure, timeout, anything unexpected
        logger.exception("Unexpected error while calling Gemini")
        raise ChatbotError(
            "I couldn't reach the AI service just now. Please try again in a moment.",
            503,
        ) from exc

    return _extract_text(response)


# ---------------------------------------------------------------------------
# Turning Gemini's responses/errors into something the UI can show
# ---------------------------------------------------------------------------

_NOT_CONFIGURED = "The AI assistant isn't set up correctly on this server yet."


def _map_api_error(exc: errors.APIError) -> ChatbotError:
    code = getattr(exc, "code", None)
    detail = str(exc)

    if code == 429:
        logger.warning("Gemini quota / rate limit hit: %s", detail)
        return ChatbotError(
            "The assistant is getting a lot of requests right now. "
            "Please wait a minute and try again.",
            503,
        )
    if code in (401, 403) or (code == 400 and "api key" in detail.lower()):
        logger.error("Gemini rejected the API key (check GEMINI_API_KEY): %s", detail)
        return ChatbotError(_NOT_CONFIGURED, 503)
    if code == 404:
        logger.error(
            "Gemini model %r was not found or has been retired. "
            "Set GEMINI_MODEL in .env to a current model. Details: %s",
            settings.gemini_model, detail,
        )
        return ChatbotError(_NOT_CONFIGURED, 503)
    if code is not None and code >= 500:
        logger.warning("Gemini server error %s: %s", code, detail)
        return ChatbotError(
            "The AI service is temporarily unavailable. Please try again shortly.", 503
        )

    logger.error("Gemini request failed (%s): %s", code, detail)
    return ChatbotError("Sorry, I couldn't process that request. Please try again.", 502)


def _extract_text(response) -> str:
    text = (response.text or "").strip() if response is not None else ""
    if text:
        return text

    # No text came back: work out why so the log (and the user) get a useful answer.
    block_reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
    finish_reason = None
    if getattr(response, "candidates", None):
        finish_reason = response.candidates[0].finish_reason

    logger.warning(
        "Gemini returned no text (block_reason=%s, finish_reason=%s)",
        block_reason, finish_reason,
    )

    if block_reason or finish_reason in (
        types.FinishReason.SAFETY,
        types.FinishReason.PROHIBITED_CONTENT,
        types.FinishReason.BLOCKLIST,
        types.FinishReason.SPII,
    ):
        message = (
            "I'm not able to answer that one. Try rephrasing it, or ask a "
            "healthcare professional."
        )
    else:
        message = "I didn't manage to put an answer together. Please try asking again."
    raise ChatbotError(message, 502)
