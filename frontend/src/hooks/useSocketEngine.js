import { useEffect, useRef, useState, useCallback } from 'react';
import { io } from 'socket.io-client';

const BACKEND_IP = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
// If we are served from Vite dev server (port 5173), point to backend at 8000.
// If we are served by the backend itself (e.g. via tunnel), use the current origin.
const isDevServer = typeof window !== 'undefined' && window.location.port === '5173';
const SOCKET_URL = import.meta.env.VITE_BACKEND_URL || (isDevServer ? `http://${BACKEND_IP}:8000` : window.location.origin);

// Derive API base URL for transcription endpoint
const API_BASE = SOCKET_URL.replace(/\/$/, '');

export function useSocketEngine(role) {
  const socketRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);

  // Session
  const [sessionId, setSessionId] = useState(null);
  const [sessionRequest, setSessionRequest] = useState(null);
  const [sessionActive, setSessionActive] = useState(false);
  const [waitingApproval, setWaitingApproval] = useState(false);
  const [sessionTaken, setSessionTaken] = useState(false);

  // Detection
  const [detectionState, setDetectionState] = useState('idle');
  const [latestSign, setLatestSign] = useState(null);
  const [confirmedWords, setConfirmedWords] = useState([]);

  // Messages / Log
  const [messages, setMessages] = useState([]);
  const [employeeMessage, setEmployeeMessage] = useState(null);

  // Alerts & Transcriptions
  const [multiPersonAlert, setMultiPersonAlert] = useState(null); // now stores the full alert data
  const [isTranscribing, setIsTranscribing] = useState(false);

  useEffect(() => {
    socketRef.current = io(SOCKET_URL, {
      transports: ['websocket', 'polling'],
      autoConnect: true,
      reconnection: true,
      reconnectionAttempts: 50,
      reconnectionDelay: 1000
    });

    const socket = socketRef.current;

    socket.on('connect', () => {
      setIsConnected(true);
      if (role === 'employee') socket.emit('join_employee');
      else if (role === 'kiosk') socket.emit('join_kiosk');
    });

    socket.on('disconnect', () => setIsConnected(false));

    // ── KIOSK events ──
    socket.on('user_detected', (data) => {
      if (role === 'kiosk') {
        setSessionId(data.session_id);
        setWaitingApproval(true);
        setSessionActive(false);
        setConfirmedWords([]);
        setDetectionState('waiting_approval');
        setLatestSign(null);
      }
    });

    socket.on('sign_detected', (data) => {
      if (role === 'kiosk') {
        setLatestSign(data);
        setDetectionState('paused');
      }
    });

    socket.on('intent_options_ready', (data) => {
      if (role === 'kiosk') {
        setLatestSign(prev => {
          if (prev && prev.word === data.word) {
            return { ...prev, intent_options: data.intent_options };
          }
          return prev;
        });
      }
    });

    socket.on('prediction_error', (data) => {
      if (role === 'kiosk') {
        setDetectionState('scanning');
      }
    });

    socket.on('detection_state', (data) => {
      if (role === 'kiosk') {
        setDetectionState(data.state === 'detecting' ? 'scanning' : data.state);
      }
    });

    socket.on('retry_ack', () => {
      if (role === 'kiosk') setDetectionState('scanning');
    });

    socket.on('multi_person_alert', (data) => {
      if (role === 'kiosk') {
        setMultiPersonAlert(data);
        setDetectionState('paused');
        setTimeout(() => setMultiPersonAlert(null), 3500);
      }
    });

    socket.on('employee_message', (data) => {
      if (role === 'kiosk') {
        setEmployeeMessage(data.reply_text);
        setDetectionState('paused');
        setMessages(prev => [...prev, {
          id: Date.now() + Math.random(),
          type: 'rx',
          text: data.reply_text,
          label: 'Bank Staff',
          time: new Date().toLocaleTimeString(),
          inputMode: 'voice'
        }]);
      }
    });

    // ── EMPLOYEE events ──
    socket.on('session_request', (data) => {
      if (role === 'employee') {
        setSessionId(data.session_id);
        setSessionRequest(data);
      }
    });

    socket.on('session_taken', () => {
      if (role === 'employee') {
        setSessionTaken(true);
        setSessionRequest(null);
        setTimeout(() => setSessionTaken(false), 4000);
      }
    });

    socket.on('message_to_employee', (data) => {
      if (role === 'employee') {
        setSessionId(data.session_id);
        setMessages(prev => [...prev, {
          id: Date.now() + Math.random(),
          type: 'rx',
          text: data.sentence,
          label: 'Kiosk Uplink',
          time: new Date(data.timestamp || Date.now()).toLocaleTimeString(),
          word: data.word,
          conf: data.confidence,
          inputMode: data.input_mode || (data.word === '(typed)' ? 'text' : 'sign')
        }]);
      }
    });

    socket.on('user_acknowledged', (data) => {
      if (role === 'employee') {
        setMessages(prev => [...prev, {
          id: Date.now() + Math.random(),
          type: 'sys',
          text: '✓ ' + (data.message || 'User acknowledged your message'),
          label: 'System',
          time: new Date(data.timestamp || Date.now()).toLocaleTimeString()
        }]);
      }
    });

    socket.on('voice_transcription_result', (data) => {
      if (role === 'employee') {
        setIsTranscribing(false);
        if (data.error) {
          alert(`Transcription Error: ${data.error}`);
          console.error("Transcription error:", data.error);
        } else if (data.text) {
          sendReply(data.text);
        }
      }
    });

    socket.on('session_status', (data) => {
      if (data.status === 'accepted') {
        setSessionActive(true);
        setSessionRequest(null);
        setWaitingApproval(false);
        setDetectionState('scanning');
      } else if (data.status === 'ended' || data.status === 'declined') {
        setSessionActive(false);
        setSessionRequest(null);
        setSessionId(null);
        setMessages([]);
        setDetectionState('idle');
        setWaitingApproval(false);
        setLatestSign(null);
        setConfirmedWords([]);
      } else if (data.status === 'claimed_elsewhere') {
        if (role === 'employee' && !sessionActive) {
          setSessionRequest(null);
          setSessionId(null);
        }
      }
    });

    return () => socket.disconnect();
  }, [role]);

  // ── Actions ──
  const acceptSession = useCallback(() => {
    if (socketRef.current && sessionId) {
      socketRef.current.emit('session_accepted', { session_id: sessionId });
      setSessionRequest(null);
      setSessionActive(true);
      setMessages(prev => [...prev, {
        id: Date.now(), type: 'sys', text: 'Session accepted — link established',
        label: 'System', time: new Date().toLocaleTimeString()
      }]);
    }
  }, [sessionId]);

  const declineSession = useCallback(() => {
    if (socketRef.current && sessionId) {
      socketRef.current.emit('session_declined', { session_id: sessionId });
      setSessionRequest(null);
      setSessionId(null);
    }
  }, [sessionId]);

  const sendReply = useCallback((text) => {
    if (socketRef.current && text.trim()) {
      socketRef.current.emit('employee_reply', { session_id: sessionId, reply_text: text });
      setMessages(prev => [...prev, {
        id: Date.now() + Math.random(), type: 'tx', text,
        label: 'You', time: new Date().toLocaleTimeString(),
        inputMode: 'voice'
      }]);
    }
  }, [sessionId]);

  const stopSigning = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.emit('stop_signing', { session_id: sessionId });
      setDetectionState('processing');
    }
  }, [sessionId]);

  const confirmSign = useCallback((word, sentence, confidence, selectedIntentLabel) => {
    if (socketRef.current) {
      socketRef.current.emit('user_confirmed', {
        session_id: sessionId, word, sentence, confidence,
        selected_intent: selectedIntentLabel || null
      });
      setConfirmedWords(prev => [...prev, word]);
      setLatestSign(null);
      setDetectionState('scanning');
      setMessages(prev => [...prev, {
        id: Date.now() + Math.random(), type: 'tx', text: sentence,
        label: 'You (Sign)', time: new Date().toLocaleTimeString(), word, conf: confidence,
        inputMode: 'sign'
      }]);
    }
  }, [sessionId]);

  const retrySign = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.emit('user_retry', { session_id: sessionId });
      setLatestSign(null);
      setDetectionState('scanning');
    }
  }, [sessionId]);

  const endSession = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.emit('stop_detection', { session_id: sessionId });
      setDetectionState('idle');
      setSessionActive(false);
      setWaitingApproval(false);
      setLatestSign(null);
      setConfirmedWords([]);
    }
  }, [sessionId]);

  const dismissEmployeeMessage = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.emit('user_acknowledged', { session_id: sessionId });
    }
    setEmployeeMessage(null);
    if (sessionActive) setDetectionState('scanning');
  }, [sessionActive, sessionId]);

  const sendTextMessage = useCallback((text) => {
      if (socketRef.current && text.trim()) {
        socketRef.current.emit('text_message', { session_id: sessionId, text });
        setMessages(prev => [...prev, {
          id: Date.now() + Math.random(), type: 'tx', text,
          label: 'You (Typed)', time: new Date().toLocaleTimeString(),
          inputMode: 'text'
        }]);
      }
    }, [sessionId]);

  const sendVoiceAudio = useCallback((blob) => {
    if (socketRef.current) {
      setIsTranscribing(true);
      socketRef.current.emit('transcribe_audio', { audio: blob });
    }
  }, []);

  return {
    socket: socketRef.current,
    isConnected,
    sessionId,
    sessionRequest,
    sessionActive,
    sessionTaken,
    waitingApproval,
    detectionState,
    latestSign,
    confirmedWords,
    messages,
    employeeMessage,
    multiPersonAlert,
    isTranscribing,
    acceptSession,
    declineSession,
    sendReply,
    stopSigning,
    confirmSign,
    retrySign,
    endSession,
    dismissEmployeeMessage,
    sendTextMessage,
    sendVoiceAudio,
    API_BASE,
  };
}
