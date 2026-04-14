
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'backend'))

import pytest


class TestEventDefinitions:
    """Tests for Socket.IO event constant definitions."""

    def test_all_events_defined(self):
        """All required events should be defined."""
        from session import events as evt

        required = [
            'USER_DETECTED', 'MULTI_PERSON_ALERT', 'SIGN_DETECTED',
            'NO_HAND', 'LOW_CONFIDENCE', 'SIGN_TOKENS', 'SESSION_STATUS',
            'EMPLOYEE_MESSAGE', 'SESSION_REQUEST', 'MESSAGE_TO_EMPLOYEE',
            'USER_CONFIRMED', 'USER_RETRY', 'SESSION_ACCEPTED',
            'SESSION_DECLINED', 'EMPLOYEE_REPLY', 'SESSION_ENDED',
        ]

        for event_name in required:
            assert hasattr(evt, event_name), f"Missing event: {event_name}"
            value = getattr(evt, event_name)
            assert isinstance(value, str), f"{event_name} should be a string"
            assert len(value) > 0, f"{event_name} should not be empty"

    def test_room_names_defined(self):
        """Room names should be defined."""
        from session import events as evt
        assert evt.ROOM_KIOSK == 'kiosk'
        assert evt.ROOM_EMPLOYEE == 'employee'

    def test_payload_builders(self):
        """Payload builder functions should return valid dicts."""
        from session import events as evt

        # user_detected
        p = evt.user_detected_payload('test-session-id')
        assert p['session_id'] == 'test-session-id'

        # sign_detected
        p = evt.sign_detected_payload('balance', 'I want balance', 0.85, [])
        assert p['word'] == 'balance'
        assert p['confidence'] == 0.85

        # low_confidence
        p = evt.low_confidence_payload('hello', 0.3)
        assert p['word'] == 'hello'
        assert p['confidence'] == 0.3

        # sign_tokens
        p = evt.sign_tokens_payload(['PLEASE', 'WAIT'])
        assert p['tokens'] == ['PLEASE', 'WAIT']


class TestSessionManager:
    """Tests for the session manager."""

    def test_create_session(self):
        """Creating a session should return a Session with a UUID."""
        from session.manager import SessionManager
        mgr = SessionManager()
        session = mgr.create_session()
        assert session.session_id is not None
        assert session.status == 'active'
        assert mgr.active_count == 1

    def test_end_session(self):
        """Ending a session should remove it from active sessions."""
        from session.manager import SessionManager
        mgr = SessionManager()
        session = mgr.create_session()
        sid = session.session_id

        doc = mgr.end_session(sid)
        assert doc is not None
        assert doc['status'] == 'completed'
        assert mgr.active_count == 0

    def test_add_message(self):
        """Messages should be added to session history."""
        from session.manager import SessionManager
        mgr = SessionManager()
        session = mgr.create_session()

        session.add_message(
            direction='user_to_employee',
            text='I want to check my balance.',
            intent='balance',
            confidence=0.82
        )

        assert len(session.messages) == 1
        assert session.messages[0]['direction'] == 'user_to_employee'
        assert session.messages[0]['intent'] == 'balance'


class TestNLPMapper:
    """Tests for the intent mapper."""

    def test_known_intent(self):
        """Known intents should return the correct sentence."""
        from nlp.mapper import IntentMapper
        mapper = IntentMapper()
        if len(mapper) > 0:
            sentence = mapper.map('balance')
            assert 'balance' in sentence.lower()

    def test_unknown_intent(self):
        """Unknown intents should return a fallback message."""
        from nlp.mapper import IntentMapper
        mapper = IntentMapper()
        sentence = mapper.map('xyz_nonexistent_word')
        assert 'xyz_nonexistent_word' in sentence

    def test_has_intent(self):
        """has_intent should correctly check existence."""
        from nlp.mapper import IntentMapper
        mapper = IntentMapper()
        if len(mapper) > 0:
            assert mapper.has_intent('balance') is True
            assert mapper.has_intent('xyz_fake') is False


class TestSignTokenizer:
    """Tests for the sign tokenizer."""

    def test_known_phrase(self):
        """Known phrases should produce correct tokens."""
        from tokenizer.tokenize import SignTokenizer
        tokenizer = SignTokenizer()
        tokens = tokenizer.tokenize('Please wait')
        assert 'PLEASE' in tokens or 'WAIT' in tokens

    def test_empty_input(self):
        """Empty input should return empty list."""
        from tokenizer.tokenize import SignTokenizer
        tokenizer = SignTokenizer()
        assert tokenizer.tokenize('') == []
        assert tokenizer.tokenize('   ') == []

    def test_is_known_word(self):
        """is_known_word should identify vocabulary words."""
        from tokenizer.tokenize import SignTokenizer
        tokenizer = SignTokenizer()
        assert tokenizer.is_known_word('balance') is True
        assert tokenizer.is_known_word('xyzfake') is False


class TestTemplatesJSON:
    """Tests for templates.json content."""

    def test_templates_has_61_entries(self):
        """templates.json should have exactly 61 entries."""
        templates_path = Path(__file__).resolve().parent.parent / 'mvp' / 'templates.json'
        if templates_path.exists():
            with open(str(templates_path), 'r') as f:
                templates = json.load(f)
            assert len(templates) == 61

    def test_all_vocabulary_covered(self):
        """All 61 vocabulary words should be in templates."""
        templates_path = Path(__file__).resolve().parent.parent / 'mvp' / 'templates.json'
        if not templates_path.exists():
            pytest.skip("templates.json not found")

        with open(str(templates_path), 'r') as f:
            templates = json.load(f)

        expected_words = {
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

        for word in expected_words:
            assert word in templates, f"Missing template for: {word}"

