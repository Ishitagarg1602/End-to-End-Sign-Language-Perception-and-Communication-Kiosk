

import json
from pathlib import Path
from typing import Optional, Dict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TEMPLATES_PATH, MVP_TEMPLATES_PATH


class IntentMapper:
    """
    Maps predicted intent labels to human-readable sentences.

    Loads templates from templates.json and provides simple lookup.
    Falls back to a generic message if intent is not found.

    Attributes:
        templates: Dictionary mapping intent → sentence.
    """

    def __init__(self, templates_path: Optional[str] = None):
        """
        Initialize the mapper by loading templates.

        Args:
            templates_path: Optional path to templates.json.
                            If None, tries backend/nlp/templates.json
                            then falls back to mvp/templates.json.
        """
        self.templates: Dict[str, str] = {}

        if templates_path:
            self._load(Path(templates_path))
        elif TEMPLATES_PATH.exists():
            self._load(TEMPLATES_PATH)
        elif MVP_TEMPLATES_PATH.exists():
            self._load(MVP_TEMPLATES_PATH)
        else:
            print("[WARNING] No templates.json found. IntentMapper is empty.")

    def _load(self, path: Path) -> None:
        """Load templates from a JSON file."""
        with open(str(path), 'r', encoding='utf-8') as f:
            self.templates = json.load(f)
        print(f"[IntentMapper] Loaded {len(self.templates)} templates from {path}")

    def map(self, intent: str, **kwargs) -> str:
        """
        Map an intent label to a grammatically correct sentence.

        Supports dynamic placeholders: if the template contains {key},
        it will be replaced with the corresponding kwarg value.

        Args:
            intent: Predicted intent label (e.g., "balance").
            **kwargs: Dynamic placeholder values (e.g., amount=5000).

        Returns:
            Human-readable sentence string.
        """
        template = self.templates.get(intent)

        if template is None:
            return f"(Sign detected: {intent})"

        # Replace any dynamic placeholders
        if kwargs:
            try:
                return template.format(**kwargs)
            except KeyError:
                return template

        return template

    def get_all_intents(self) -> list:
        """Return all available intent labels."""
        return list(self.templates.keys())

    def has_intent(self, intent: str) -> bool:
        """Check if an intent is in the templates."""
        return intent in self.templates

    def __len__(self) -> int:
        return len(self.templates)
