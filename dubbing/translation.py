"""Duration-aware translation with an OpenAI chat model.

Segments are sent in batches so the model sees surrounding context, and each
segment carries its original duration so the translation can be kept short
enough to be spoken in the same time window.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List

from openai import OpenAI

from dubbing.config import Config

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a professional video dubbing translator translating from English to {target_language}.

RULES:
1. Translate the text naturally.
2. CRITICAL: The translated text length (syllables/speaking time) MUST match the 'duration_seconds' provided.
3. If the original duration is short, use short synonyms. If long, use slightly longer phrases but don't add unnecessary filler.
4. Maintain the context between sentences.
5. Output MUST be a valid JSON object with a key 'translations' containing a list of objects.

Output format example:
{{
    "translations": [
        {{"id": 1, "translated_text": "..."}},
        {{"id": 2, "translated_text": "..."}}
    ]
}}
"""


class Translator:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)

    def _translate_batch(self, batch: List[Dict]) -> Dict[int, str]:
        payload = [
            {
                "id": seg["id"],
                "text": seg["text"],
                "duration_seconds": round(seg["end"] - seg["start"], 2),
            }
            for seg in batch
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.config.translation_model,
                response_format={"type": "json_object"},
                temperature=self.config.translation_temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(
                        target_language=self.config.target_language)},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            items = json.loads(response.choices[0].message.content).get("translations", [])
            return {item["id"]: item["translated_text"] for item in items}
        except Exception:
            log.exception("Translation batch failed; keeping original text for this batch")
            return {}

    def translate(self, segments: List[Dict]) -> List[Dict]:
        """Return copies of ``segments`` with ``text`` translated and the
        source kept in ``source_text``. Untranslated segments fall back to
        the original text."""
        size = self.config.translation_batch_size
        result = []
        for i in range(0, len(segments), size):
            batch = segments[i: i + size]
            log.info("Translating segments %d-%d of %d", i + 1, i + len(batch), len(segments))
            translations = self._translate_batch(batch)
            for seg in batch:
                result.append({
                    **seg,
                    "source_text": seg["text"],
                    "text": translations.get(seg["id"], seg["text"]),
                })
        return result
