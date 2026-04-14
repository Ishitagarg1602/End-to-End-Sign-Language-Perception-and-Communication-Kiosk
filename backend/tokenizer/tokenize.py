"""
backend/tokenizer/tokenize.py
==============================
Convert employee text replies into sign token sequences.

The tokenizer breaks an employee's text reply into a sequence of sign tokens
that can be played as avatar animations on the kiosk screen.

Tokenization strategy:
  1. Lowercase and clean the input text
  2. Match against known sign vocabulary (60 banking words)
  3. Map common phrases to predefined token sequences
  4. Unknown words are spelled out or skipped

Usage:
    from backend.tokenizer.tokenize import SignTokenizer
    tokenizer = SignTokenizer()
    tokens = tokenizer.tokenize("Please wait one moment")
    # → ["PLEASE", "WAIT"]
"""

import json
import re
from pathlib import Path
from typing import List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSES_PATH


# ─── Common phrase mappings ──────────────────────────────────────────────────
PHRASE_MAP = {
    "please wait": ["WAIT"],
    "one moment": ["WAIT"],
    "please show id": ["IDENTITY_CARD"],
    "please show your id": ["IDENTITY_CARD"],
    "show id": ["IDENTITY_CARD"],
    "thank you": ["THANK_YOU"],
    "thanks": ["THANK_YOU"],
    "good morning": ["GOOD_MORNING"],
    "good afternoon": ["HELLO"],
    "good evening": ["HELLO"],
    "credit card": ["CREDIT_CARD"],
    "debit card": ["DEBIT_CARD"],
    "savings account": ["SAVINGS_ACCOUNT"],
    "current account": ["CURRENT_ACCOUNT"],
    "joint account": ["JOINT_ACCOUNT"],
    "mobile banking": ["MOBILE_BANKING"],
    "interest rate": ["INTEREST_RATE"],
    "security deposit": ["SECURITY_DEPOSIT"],
    "account blocked": ["ACCOUNT_BLOCKED"],
    "account statement": ["ACCOUNT_STATEMENT"],
    "opening balance": ["OPENING_BALANCE"],
    "bank branch name": ["BANK_BRANCH_NAME"],
    "cif number": ["CIF_NUMBER"],
}

# ─── Stop words to skip ─────────────────────────────────────────────────────
STOP_WORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'it', 'its', 'this', 'that',
    'i', 'you', 'we', 'they', 'he', 'she', 'me', 'my', 'your', 'our',
    'and', 'or', 'but', 'if', 'then', 'so', 'just', 'very', 'really',
    'please', 'kindly', 'sir', 'madam', 'ma\'am', 'mr', 'mrs',
}


class SignTokenizer:
    """
    Converts employee text replies into sign language token sequences.

    Tokens correspond to available sign animations. Each token maps to
    a .glb animation file that the Three.js avatar can play.

    Attributes:
        vocabulary: Set of known sign words.
    """

    def __init__(self):
        """Initialize the tokenizer with the known sign vocabulary."""
        self.vocabulary = set()

        if CLASSES_PATH.exists():
            with open(str(CLASSES_PATH), 'r') as f:
                classes = json.load(f)
                self.vocabulary = {w.lower() for w in classes}
        else:
            # Fallback: use the 61-word vocabulary
            self.vocabulary = {
                'account', 'account_blocked', 'account_closing',
                'account_holder', 'account_statement', 'address',
                'affidavit', 'amount', 'atm', 'balance', 'bank',
                'bank_branch_name', 'cancel', 'cash', 'change',
                'cheque', 'cif_number', 'complain', 'credit_card',
                'current_account', 'debit_card', 'deposit', 'dividend',
                'expenditure', 'finish', 'form', 'fraud', 'good_morning',
                'hello', 'help', 'identity_card', 'income',
                'interest_rate', 'joint_account', 'kyc', 'loan', 'lose',
                'mobile_banking', 'money', 'mortgages', 'name', 'no',
                'nominee', 'number', 'open', 'opening_balance',
                'passbook', 'payment', 'paytm', 'phone', 'receive',
                'revenue', 'savings_account', 'security_deposit', 'send',
                'signature', 'thank_you', 'transfer', 'verification',
                'wait', 'withdraw'
            }

    def tokenize(self, text: str) -> List[str]:
        """
        Convert text to a sequence of sign tokens.

        Strategy:
          1. Check for known phrase matches first
          2. For remaining text, match individual words against vocabulary
          3. Skip stop words and unknown words

        Args:
            text: Input text string from employee.

        Returns:
            List of uppercase sign token strings.
        """
        if not text or not text.strip():
            return []

        text_lower = text.strip().lower()

        # ─── Phase 1: Check phrase matches ────────────────────────────
        for phrase, tokens in PHRASE_MAP.items():
            if phrase in text_lower:
                return tokens

        # ─── Phase 2: Word-level tokenization ─────────────────────────
        # Clean text: keep only letters, numbers, spaces, underscores
        cleaned = re.sub(r'[^a-z0-9\s_]', '', text_lower)
        words = cleaned.split()

        tokens = []
        i = 0
        while i < len(words):
            word = words[i]

            # Try two-word compound matches (e.g., "credit card" → "credit_card")
            if i + 1 < len(words):
                compound = f"{word}_{words[i+1]}"
                if compound in self.vocabulary:
                    tokens.append(compound.upper())
                    i += 2
                    continue

            # Single word match
            if word in self.vocabulary:
                tokens.append(word.upper())
            elif word not in STOP_WORDS and len(word) > 2:
                # Try partial matches or synonyms
                match = self._find_closest(word)
                if match:
                    tokens.append(match.upper())

            i += 1

        return tokens

    def _find_closest(self, word: str) -> Optional[str]:
        """
        Find the closest vocabulary word for a non-matching word.

        Simple heuristic: check if the word is a substring of any
        vocabulary word, or vice versa.

        Args:
            word: Input word to match.

        Returns:
            Closest vocabulary word, or None.
        """
        for vocab_word in self.vocabulary:
            if word in vocab_word or vocab_word in word:
                return vocab_word
        return None

    def is_known_word(self, word: str) -> bool:
        """Check if a word is in the sign vocabulary."""
        return word.lower() in self.vocabulary


# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    tokenizer = SignTokenizer()
    test_phrases = [
        "Please wait",
        "One moment",
        "Please show ID",
        "Your balance is ready",
        "I will help you with the transfer",
        "Thank you for visiting",
        "Please sign again slowly",
    ]
    for phrase in test_phrases:
        tokens = tokenizer.tokenize(phrase)
        print(f"  '{phrase}' → {tokens}")
