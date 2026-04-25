import { useEffect, useRef, useState, useCallback } from 'react';
import { io } from 'socket.io-client';

// Dynamically use the host running the frontend to connect to the backend
const BACKEND_IP = window.location.hostname;
const SOCKET_URL = `http://${BACKEND_IP}:8000`;

export function useSocketEngine(role) {
  const socketRef = useRef(null);
  const [isConnected, setIsConnected] = useState(false);

  // Session
  const [sessionId, setSessionId] = useState(null);
  const [sessionRequest, setSessionRequest] = useState(null);
  const [sessionActive, setSessionActive] = useState(false);
  const [waitingApproval, setWaitingApproval] = useState(false);

  // Detection
  const [detectionState, setDetectionState] = useState('idle');
  const [latestSign, setLatestSign] = useState(null);
  const [confirmedWords, setConfirmedWords] = useState([]);

  // Messages / Log
  const [messages, setMessages] = useState([]);
  const [employeeMessage, setEmployeeMessage] = useState(null);

  // Alerts
  const [multiPersonAlert, setMultiPersonAlert] = useState(false);

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
      console.log(`[Socket] Connected as ${role}`);
      setIsConnected(true);
      if (role === 'employee') socket.emit('join_employee');
      else if (role === 'kiosk') socket.emit('join_kiosk');
    });

    socket.on('disconnect', () => {
      console.log('[Socket] Disconnected');
      setIsConnected(false);
    });

    // ── KIOSK events ──
    socket.on('user_detected', (data) => {
      console.log('[Socket] User detected:', data);
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

    socket.on('multi_person_alert', () => {
      if (role === 'kiosk') {
        setMultiPersonAlert(true);
        setDetectionState('paused');
        setTimeout(() => setMultiPersonAlert(false), 2500);
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
          time: new Date().toLocaleTimeString()
        }]);
      }
    });

    // ── EMPLOYEE events ──
    socket.on('session_request', (data) => {
      console.log('[Socket] Session request received:', data);
      if (role === 'employee') {
        setSessionId(data.session_id);
        setSessionRequest(data);
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
          conf: data.confidence
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

    socket.on('session_status', (data) => {
      console.log('[Socket] Session status update:', data);
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
      }
    });

    return () => socket.disconnect();
  }, [role]);

  // ── Actions ──
  const acceptSession = useCallback(() => {
    console.log('[Action] Accepting session:', sessionId);
    if (socketRef.current && sessionId) {
      socketRef.current.emit('session_accepted', { session_id: sessionId });
      console.log('[Action] session_accepted EMITTED');
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
        label: 'You', time: new Date().toLocaleTimeString()
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
        label: 'You (Sign)', time: new Date().toLocaleTimeString(), word, conf: confidence
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
        label: 'You (Typed)', time: new Date().toLocaleTimeString()
      }]);
    }
  }, [sessionId]);

  return {
    socket: socketRef.current,
    isConnected,
    sessionId,
    sessionRequest,
    sessionActive,
    waitingApproval,
    detectionState,
    latestSign,
    confirmedWords,
    messages,
    employeeMessage,
    multiPersonAlert,
    acceptSession,
    declineSession,
    sendReply,
    stopSigning,
    confirmSign,
    retrySign,
    endSession,
    dismissEmployeeMessage,
    sendTextMessage,
  };
}
