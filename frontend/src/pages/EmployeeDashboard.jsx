import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useSocketEngine } from '../hooks/useSocketEngine';
import { Briefcase, Mic, Square, Send, Activity, LogOut, Check, X, MessageSquare, Clock, Power, FileScan, Download, FileText } from 'lucide-react';

export default function EmployeeDashboard() {
  const {
    isConnected, sessionId, sessionRequest, sessionActive, sessionTaken,
    messages, acceptSession, declineSession, sendReply, endSession,
    multiPersonAlert, isTranscribing, sendVoiceAudio, API_BASE,
    transcribedText, setTranscribedText
  } = useSocketEngine('employee');

  const navigate = useNavigate();
  const [inputText, setInputText] = useState('');
  const logEndRef = useRef(null);
  const [sessionStart, setSessionStart] = useState(null);
  const [elapsed, setElapsed] = useState('0:00');
  const [showEndConfirm, setShowEndConfirm] = useState(false);
  const [showTranscriptBtn, setShowTranscriptBtn] = useState(false);
  const sessionMessagesRef = useRef([]);
  const sessionMetaRef = useRef({});

  // Mic state
  const [isMicRecording, setIsMicRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const [expandedImage, setExpandedImage] = useState(null);

  useEffect(() => {
    if (transcribedText) {
      setInputText(prev => prev ? prev + ' ' + transcribedText : transcribedText);
      setTranscribedText(null);
    }
  }, [transcribedText, setTranscribedText]);

  // Web Audio API Notification Sound
  const playNotificationSound = useCallback((type = 'message') => {
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) return;
      const ctx = new AudioContext();
      const osc = ctx.createOscillator();
      const gainNode = ctx.createGain();
      osc.connect(gainNode);
      gainNode.connect(ctx.destination);
      
      if (type === 'message') {
        osc.type = 'sine';
        osc.frequency.setValueAtTime(523.25, ctx.currentTime); // C5
        osc.frequency.exponentialRampToValueAtTime(659.25, ctx.currentTime + 0.1); // E5
        gainNode.gain.setValueAtTime(0.3, ctx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.3);
      } else if (type === 'alert') {
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(440, ctx.currentTime); // A4
        osc.frequency.setValueAtTime(880, ctx.currentTime + 0.1); // A5
        gainNode.gain.setValueAtTime(0.4, ctx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.4);
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 0.4);
      }
    } catch (e) {
      console.warn("Audio context failed:", e);
    }
  }, []);

  const lastKioskMsg = messages.filter(m => m.type === 'rx').slice(-1)[0];

  // Sound triggers
  useEffect(() => {
    if (sessionRequest) playNotificationSound('alert');
  }, [sessionRequest, playNotificationSound]);

  useEffect(() => {
    if (sessionActive) playNotificationSound('message');
  }, [sessionActive, playNotificationSound]);

  useEffect(() => {
    if (lastKioskMsg) playNotificationSound('message');
  }, [lastKioskMsg, playNotificationSound]);

  useEffect(() => {
    if (sessionActive && !sessionStart) {
      setSessionStart(Date.now());
      setShowTranscriptBtn(false);
      sessionMetaRef.current = { startTime: new Date().toISOString(), sessionId: sessionId };
    }
    if (!sessionActive && sessionStart) {
      sessionMetaRef.current.endTime = new Date().toISOString();
      sessionMetaRef.current.duration = elapsed;
      sessionMessagesRef.current = [...messages];
      if (messages.length > 0) setShowTranscriptBtn(true);
      setSessionStart(null);
      setElapsed('0:00');
    }
  }, [sessionActive]);

  useEffect(() => {
    if (!sessionStart) return;
    const timer = setInterval(() => {
      const diff = Math.floor((Date.now() - sessionStart) / 1000);
      const m = Math.floor(diff / 60);
      const s = diff % 60;
      setElapsed(`${m}:${s.toString().padStart(2, '0')}`);
    }, 1000);
    return () => clearInterval(timer);
  }, [sessionStart]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    if (!inputText.trim()) return;
    sendReply(inputText);
    setInputText('');
  };

  const quickReply = (text) => sendReply(text);

  const handleEndSession = () => {
    endSession();
    setShowEndConfirm(false);
  };

  const generateTranscript = () => {
    const meta = sessionMetaRef.current;
    const msgs = sessionMessagesRef.current;
    const startDate = meta.startTime ? new Date(meta.startTime) : new Date();
    const rows = msgs.map(msg => {
      const direction = msg.type === 'rx' ? 'KIOSK USER' : msg.type === 'tx' ? 'EMPLOYEE' : msg.type === 'doc' ? 'DOCUMENT SCAN' : 'SYSTEM';
      const mode = msg.inputMode === 'sign' ? 'Sign Language' : msg.inputMode === 'voice' ? 'Voice' : msg.inputMode === 'text' ? 'Keyboard' : msg.type === 'doc' ? 'Camera Scan' : '-';
      return `<tr>
        <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;font-size:12px;color:#6b7280;white-space:nowrap;">${msg.time || '-'}</td>
        <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;font-size:12px;font-weight:600;color:${msg.type === 'rx' ? '#7c3aed' : msg.type === 'tx' ? '#2563eb' : msg.type === 'doc' ? '#0369a1' : '#6b7280'};white-space:nowrap;">${direction}</td>
        <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;font-size:12px;color:#6b7280;white-space:nowrap;">${mode}</td>
        <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;font-size:13px;color:#111827;">${msg.text || '-'}</td>
      </tr>`;
    }).join('');

    const docScans = msgs.filter(m => m.type === 'doc' && m.images && m.images.length > 0);
    const docSection = docScans.length > 0 ? `
      <div style="margin-top:32px;">
        <h2 style="font-size:14px;font-weight:700;color:#111827;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;padding-bottom:8px;border-bottom:2px solid #111827;">Appendix: Scanned Documents</h2>
        ${docScans.map((d, di) => `
          <div style="margin-bottom:16px;">
            <p style="font-size:12px;font-weight:600;color:#374151;margin-bottom:6px;">Document Set ${di + 1} - ${d.time}</p>
            <p style="font-size:12px;color:#6b7280;margin-bottom:8px;">AI Analysis: ${d.text}</p>
            <div style="display:flex;gap:8px;flex-wrap:wrap;">
              ${d.images.map((img, ii) => `<img src="${img}" style="width:200px;height:140px;object-fit:cover;border:1px solid #d1d5db;border-radius:4px;" />`).join('')}
            </div>
          </div>
        `).join('')}
      </div>` : '';

    const html = `<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Session Transcript - ${meta.sessionId || 'N/A'}</title>
<style>
  @media print { body { -webkit-print-color-adjust: exact; print-color-adjust: exact; } }
  body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px; color: #111827; }
  table { width: 100%; border-collapse: collapse; }
</style></head><body>
  <div style="border-bottom:3px solid #111827;padding-bottom:16px;margin-bottom:24px;">
    <h1 style="font-size:20px;font-weight:700;margin:0 0 4px 0;letter-spacing:0.5px;">ISL BANKING KIOSK - SESSION TRANSCRIPT</h1>
    <p style="font-size:12px;color:#6b7280;margin:0;">Confidential - For Internal Banking Use Only</p>
  </div>

  <table style="margin-bottom:32px;">
    <tr><td style="padding:4px 12px 4px 0;font-size:12px;font-weight:600;color:#374151;width:140px;">Session ID</td><td style="padding:4px 0;font-size:12px;color:#111827;">${meta.sessionId || 'N/A'}</td></tr>
    <tr><td style="padding:4px 12px 4px 0;font-size:12px;font-weight:600;color:#374151;">Date</td><td style="padding:4px 0;font-size:12px;color:#111827;">${startDate.toLocaleDateString('en-IN', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</td></tr>
    <tr><td style="padding:4px 12px 4px 0;font-size:12px;font-weight:600;color:#374151;">Start Time</td><td style="padding:4px 0;font-size:12px;color:#111827;">${startDate.toLocaleTimeString('en-IN')}</td></tr>
    <tr><td style="padding:4px 12px 4px 0;font-size:12px;font-weight:600;color:#374151;">Duration</td><td style="padding:4px 0;font-size:12px;color:#111827;">${meta.duration || 'N/A'}</td></tr>
    <tr><td style="padding:4px 12px 4px 0;font-size:12px;font-weight:600;color:#374151;">Total Messages</td><td style="padding:4px 0;font-size:12px;color:#111827;">${msgs.length}</td></tr>
    <tr><td style="padding:4px 12px 4px 0;font-size:12px;font-weight:600;color:#374151;">Documents Scanned</td><td style="padding:4px 0;font-size:12px;color:#111827;">${docScans.length}</td></tr>
  </table>

  <h2 style="font-size:14px;font-weight:700;color:#111827;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;padding-bottom:8px;border-bottom:2px solid #111827;">Communication Log</h2>
  <table>
    <thead><tr style="background:#f9fafb;">
      <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:700;color:#374151;text-transform:uppercase;letter-spacing:0.5px;border-bottom:2px solid #d1d5db;">Time</th>
      <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:700;color:#374151;text-transform:uppercase;letter-spacing:0.5px;border-bottom:2px solid #d1d5db;">Source</th>
      <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:700;color:#374151;text-transform:uppercase;letter-spacing:0.5px;border-bottom:2px solid #d1d5db;">Input Mode</th>
      <th style="padding:8px 12px;text-align:left;font-size:11px;font-weight:700;color:#374151;text-transform:uppercase;letter-spacing:0.5px;border-bottom:2px solid #d1d5db;">Message</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>
  ${docSection}

  <div style="margin-top:40px;padding-top:16px;border-top:1px solid #e5e7eb;font-size:11px;color:#9ca3af;text-align:center;">
    Generated automatically by ISL Banking Kiosk System | ${new Date().toLocaleString('en-IN')}
  </div>
</body></html>`;

    const w = window.open('', '_blank');
    w.document.write(html);
    w.document.close();
    setTimeout(() => w.print(), 400);
  };

  // Whisper Mic — uses dynamic API_BASE
  const initMic = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      let options = { mimeType: 'audio/webm' };
      if (typeof MediaRecorder !== 'undefined') {
        if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
          options = { mimeType: 'audio/webm;codecs=opus' };
        }
      }
      
      const recorder = new MediaRecorder(stream, options);
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, { type: options.mimeType });
        audioChunksRef.current = [];
        sendVoiceAudio(blob);
        setIsMicRecording(false);
      };
      mediaRecorderRef.current = recorder;
    } catch (err) {
      alert('Microphone access denied or unavailable.');
    }
  }, [sendVoiceAudio]);

  const toggleMic = useCallback(async () => {
    if (isTranscribing) return;
    if (!mediaRecorderRef.current) {
      await initMic();
      if (!mediaRecorderRef.current) return;
    }
    if (isMicRecording) {
      mediaRecorderRef.current.stop();
    } else {
      audioChunksRef.current = [];
      mediaRecorderRef.current.start();
      setIsMicRecording(true);
    }
  }, [isMicRecording, isTranscribing, initMic]);

  const quickReplies = ['Please wait', 'One moment', 'Please show ID', 'Proceeding with request', 'Please sign again', 'Your request is complete'];

  const getInputModeIcon = (mode) => {
    if (mode === 'sign') return '🤟';
    if (mode === 'text') return '⌨️';
    if (mode === 'voice') return '🎙️';
    return '';
  };

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>

      {/* End Session Confirm Dialog */}
      {showEndConfirm && (
        <div style={{ position: 'fixed', inset: 0, zIndex: 5000, background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(4px)' }}>
          <div className="animate-enter surface-card" style={{ padding: 32, maxWidth: 380, textAlign: 'center' }}>
            <Power size={36} style={{ color: 'var(--danger)', marginBottom: 16 }} />
            <h3 style={{ marginBottom: 8 }}>End Session?</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: 14, marginBottom: 24 }}>This will disconnect the kiosk user. Are you sure?</p>
            <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
              <button className="btn" onClick={handleEndSession} style={{ background: 'var(--danger)', color: 'white', padding: '10px 28px' }}>End Session</button>
              <button className="btn btn-secondary" onClick={() => setShowEndConfirm(false)} style={{ padding: '10px 28px' }}>Cancel</button>
            </div>
          </div>
        </div>
      )}

      {/* Download Transcript Banner */}
      {showTranscriptBtn && !sessionActive && (
        <div style={{ position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)', zIndex: 4000, background: 'var(--bg-surface)', border: '1px solid var(--border-light)', borderRadius: 16, padding: '16px 24px', display: 'flex', alignItems: 'center', gap: 16, boxShadow: '0 20px 40px rgba(0,0,0,0.15)' }}>
          <FileText size={24} style={{ color: 'var(--accent)' }} />
          <div>
            <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-main)' }}>Session Complete</div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Download the compliance transcript for this session.</div>
          </div>
          <button onClick={generateTranscript} className="btn" style={{ background: 'var(--accent)', color: 'white', padding: '10px 20px', fontSize: 13, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8, whiteSpace: 'nowrap' }}>
            <Download size={16} /> Download Transcript
          </button>
          <button onClick={() => setShowTranscriptBtn(false)} style={{ background: 'transparent', border: 'none', color: 'var(--text-faint)', cursor: 'pointer' }}><X size={18} /></button>
        </div>
      )}

      {/* Sidebar */}
      <nav style={{ width: 72, background: '#0d0d0d', borderRight: '1px solid #1f1f1f', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '20px 0', gap: 28, flexShrink: 0 }}>
        <div style={{ width: 40, height: 40, background: '#2a2a2a', borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'rgba(255,255,255,0.85)', boxShadow: '0 2px 8px rgba(0,0,0,0.4)' }}>
          <Briefcase size={20} />
        </div>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ width: 44, height: 44, display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 12, background: 'rgba(255,255,255,0.08)', color: '#D6C2A8', cursor: 'pointer' }}>
            <Activity size={20} />
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: isConnected ? '#4ade80' : '#f87171', boxShadow: isConnected ? '0 0 6px #4ade80' : 'none' }} />
            <span style={{ fontSize: 9, fontWeight: 600, color: 'rgba(255,255,255,0.4)', letterSpacing: 0.5 }}>{isConnected ? 'LIVE' : 'OFF'}</span>
          </div>
          <div onClick={() => navigate('/login')} style={{ width: 44, height: 44, display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 12, color: 'rgba(255,255,255,0.45)', cursor: 'pointer' }}>
            <LogOut size={20} />
          </div>
        </div>
      </nav>

      {/* Main Workspace */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

        {/* Header */}
        <header style={{ padding: '16px 36px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #2a2a2a', background: '#1a1a1a', flexShrink: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <h1 className="heading-display" style={{ fontSize: 18, letterSpacing: 0.5, color: '#ffffff', fontWeight: 700 }}>ISL Banking Interface</h1>
            {sessionActive && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '4px 12px', borderRadius: 20, background: 'rgba(255,255,255,0.1)', color: '#D6C2A8', fontSize: 11, fontWeight: 700 }}>
                <Clock size={12} /> {elapsed}
              </div>
            )}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            {sessionActive && (
              <button className="btn" onClick={() => setShowEndConfirm(true)} style={{ background: 'rgba(220,38,38,0.15)', color: '#f87171', border: '1px solid rgba(220,38,38,0.3)', padding: '6px 16px', fontSize: 11 }}>
                <Power size={14} /> End Session
              </button>
            )}
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '6px 18px', borderRadius: 30, background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.12)' }}>
              <div style={{ width: 7, height: 7, borderRadius: '50%', background: isConnected ? '#4ade80' : '#f87171', boxShadow: isConnected ? '0 0 6px #4ade80' : 'none' }} />
              <span style={{ fontSize: 11, fontWeight: 700, color: '#ffffff', letterSpacing: 1, textTransform: 'uppercase' }}>
                {isConnected ? 'Data Link Secure' : 'Offline'}
              </span>
            </div>
          </div>
        </header>

        {/* Grid */}
        <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, padding: '24px 36px', overflow: 'hidden' }}>

          {/* Left Panel: Kiosk Downlink */}
          <div className="surface-card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-light)', display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>
              <MessageSquare size={14} color="var(--accent)" /> Kiosk Downlink
            </div>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 20 }}>

              {/* Session Request Alert */}
              {sessionRequest && !sessionActive && !sessionTaken && (
                <div className="animate-enter" style={{ background: 'var(--accent-light)', border: '1px solid rgba(37,99,235,0.3)', borderRadius: 16, padding: 24, textAlign: 'center', marginBottom: 20, flexShrink: 0, boxShadow: 'var(--shadow-lg)' }}>
                  <div style={{ color: 'var(--accent)', fontSize: 14, fontWeight: 700, letterSpacing: 1.5, textTransform: 'uppercase', marginBottom: 20 }}>
                    🔔 New Communication Request
                  </div>
                  <p style={{ color: 'var(--text-muted)', fontSize: 13, marginBottom: 20 }}>A deaf user at the kiosk needs assistance.</p>
                  <div style={{ display: 'flex', gap: 16, justifyContent: 'center' }}>
                    <button className="btn" onClick={acceptSession} style={{ background: 'var(--success-bg)', color: 'var(--success)', border: '1px solid rgba(5,150,105,0.3)', padding: '12px 28px', fontSize: 12, fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase' }}>
                      <Check size={16} /> Accept
                    </button>
                    <button className="btn btn-secondary" onClick={declineSession} style={{ padding: '12px 28px', fontSize: 12, fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase' }}>
                      <X size={16} /> Decline
                    </button>
                  </div>
                </div>
              )}

              {sessionTaken && !sessionActive && (
                <div className="animate-enter" style={{ background: 'var(--danger-bg)', border: '1px solid rgba(220,38,38,0.3)', borderRadius: 16, padding: 24, textAlign: 'center', marginBottom: 20 }}>
                  <div style={{ color: 'var(--danger)', fontSize: 14, fontWeight: 700, marginBottom: 10 }}>Session Taken</div>
                  <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>Another representative accepted this session.</div>
                </div>
              )}

              {/* Message Display */}
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center', background: '#e0e0e0', borderRadius: 16, border: '1px solid #cccccc', boxShadow: 'inset 0 2px 6px rgba(0,0,0,0.06)' }}>
                {lastKioskMsg ? (
                  <div className="animate-enter">
                    <div style={{ fontSize: 32, fontWeight: 700, lineHeight: 1.3, color: 'var(--text-main)', marginBottom: 20, padding: '0 20px' }}>
                      "{lastKioskMsg.text}"
                    </div>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 14, flexWrap: 'wrap', justifyContent: 'center' }}>
                      {lastKioskMsg.word && (
                        <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase', padding: '6px 14px', borderRadius: 8, background: 'rgba(139,92,246,0.08)', color: '#7C3AED', border: '1px solid rgba(139,92,246,0.2)' }}>
                          {getInputModeIcon(lastKioskMsg.inputMode)} {lastKioskMsg.word}
                        </span>
                      )}
                      {lastKioskMsg.conf != null && (
                        <span style={{ fontSize: 11, fontWeight: 700, padding: '6px 14px', borderRadius: 8, background: lastKioskMsg.conf >= 0.75 ? 'var(--success-bg)' : 'rgba(245,158,11,0.1)', color: lastKioskMsg.conf >= 0.75 ? 'var(--success)' : '#F59E0B' }}>
                          {Math.round(lastKioskMsg.conf * 100)}%
                        </span>
                      )}
                      <span style={{ fontSize: 11, color: 'var(--text-faint)', letterSpacing: 1 }}>{lastKioskMsg.time}</span>
                    </div>
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: '#222222', letterSpacing: 2, textTransform: 'uppercase' }}>
                      {sessionActive ? 'Waiting for user message...' : 'No Signal Detected'}
                    </div>
                    {!sessionActive && (
                      <div style={{ fontSize: 12, color: '#777777', fontWeight: 400, letterSpacing: 0.5 }}>Waiting for user to begin sign language...</div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right Panel: Transmission Control */}
          <div className="surface-card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-light)', display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>
              <Send size={14} color="var(--accent)" /> Transmission Control
            </div>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 20, overflow: 'hidden' }}>

              {/* Input Bar */}
              <div style={{ marginBottom: 20, flexShrink: 0 }}>
                <div style={{ fontSize: 10, fontWeight: 600, color: '#999999', letterSpacing: 1, textTransform: 'uppercase', marginBottom: 6, textAlign: 'right' }}>
                  Voice-to-Text Active
                </div>
                <div style={{ display: 'flex', gap: 10, background: '#e0e0e0', padding: 8, borderRadius: 16, border: '1px solid #cccccc', alignItems: 'center' }}>
                  <input type="text" value={inputText} onChange={e => setInputText(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleSend()}
                    placeholder="Type response or use microphone..."
                    style={{ flex: 1, background: 'transparent', border: 'none', padding: '0 16px', fontSize: 15, outline: 'none', boxShadow: 'none', color: '#333333' }} />

                  <button onClick={toggleMic} disabled={isTranscribing}
                    style={{ width: 48, height: 48, borderRadius: 12, border: `1px solid ${isMicRecording ? 'var(--danger)' : '#bbbbbb'}`, background: isMicRecording ? 'var(--danger-bg)' : '#f0f0f0', color: isMicRecording ? 'var(--danger)' : '#555555', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', transition: '0.3s', flexShrink: 0 }}>
                    {isTranscribing ? (
                      <div style={{ width: 18, height: 18, border: '2px solid #999', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
                    ) : isMicRecording ? (
                      <Square size={20} fill="currentColor" />
                    ) : (
                      <Mic size={20} />
                    )}
                  </button>

                  <button className="btn" onClick={handleSend} style={{ height: 48, padding: '0 20px', borderRadius: 12, background: '#111111', color: '#ffffff', border: 'none' }}>
                    <Send size={18} />
                  </button>
                </div>
              </div>

              {/* Quick Replies */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20, paddingBottom: 20, borderBottom: '1px solid #e5e5e5', flexShrink: 0 }}>
                {quickReplies.map(text => (
                  <button key={text} onClick={() => quickReply(text)}
                    style={{ padding: '8px 16px', borderRadius: 30, background: '#D6C2A8', border: '1px solid #c4ae93', color: '#1a1a1a', fontSize: 13, fontWeight: 600, cursor: 'pointer', transition: '0.2s', fontFamily: 'var(--font-sans)', boxShadow: '0 1px 3px rgba(0,0,0,0.08)' }}>
                    {text}
                  </button>
                ))}
              </div>

              {/* Event Log */}
              <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 12, paddingRight: 6, minHeight: 0 }}>
                {messages.length === 0 ? (
                  <div style={{ textAlign: 'center', padding: '50px 20px', color: 'var(--text-faint)', fontSize: 12, fontWeight: 600, letterSpacing: 2, textTransform: 'uppercase' }}>
                    Session Event Log Empty
                  </div>
                ) : (
                  messages.map((msg) => (
                    <div key={msg.id} className="animate-enter" style={{
                      padding: '14px 18px', borderRadius: 12,
                      background: msg.type === 'rx' ? 'linear-gradient(90deg, rgba(139,92,246,0.04) 0%, transparent 100%)' : msg.type === 'sys' ? 'var(--bg-subtle)' : msg.type === 'doc' ? 'rgba(59,130,246,0.05)' : 'var(--bg-page)',
                      border: msg.type === 'doc' ? '1px solid rgba(59,130,246,0.2)' : '1px solid transparent',
                      borderLeft: `3px solid ${msg.type === 'rx' ? '#8B5CF6' : msg.type === 'tx' ? 'var(--accent)' : msg.type === 'doc' ? '#3B82F6' : msg.text.startsWith('✓') ? 'var(--success)' : 'var(--text-faint)'}`,
                      display: 'flex', flexDirection: 'column', gap: 8
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: 4,
                          color: msg.type === 'rx' ? '#8B5CF6' : msg.type === 'tx' ? 'var(--accent)' : msg.type === 'doc' ? '#3B82F6' : msg.text.startsWith('✓') ? 'var(--success)' : 'var(--text-faint)' }}>
                          {msg.type === 'doc' ? <FileScan size={12} /> : msg.inputMode ? getInputModeIcon(msg.inputMode) : null} {msg.label}
                        </span>
                        <span style={{ fontSize: 10, color: 'var(--text-faint)', letterSpacing: 1 }}>{msg.time}</span>
                      </div>
                      <div style={{ fontSize: 14, fontWeight: 400, lineHeight: 1.5, color: 'var(--text-main)' }}>{msg.text}</div>
                      
                      {msg.type === 'doc' && msg.images && msg.images.length > 0 && (
                        <div style={{ marginTop: 8, display: 'flex', gap: 8, overflowX: 'auto', paddingBottom: 4 }}>
                          {msg.images.map((img, imgIdx) => (
                            <div key={imgIdx} style={{ flexShrink: 0, cursor: 'pointer', borderRadius: 8, overflow: 'hidden', border: '1px solid var(--border-light)', width: 120 }} onClick={() => setExpandedImage(img)}>
                              <img src={img} alt={`Page ${imgIdx + 1}`} style={{ width: '100%', height: 80, objectFit: 'cover', display: 'block' }} />
                              <div style={{ background: 'var(--bg-subtle)', padding: '4px 8px', fontSize: 10, color: 'var(--text-muted)', textAlign: 'center', fontWeight: 600 }}>Page {imgIdx + 1}</div>
                            </div>
                          ))}
                        </div>
                      )}
                      {/* Legacy single image support */}
                      {msg.type === 'doc' && msg.image && !msg.images && (
                        <div style={{ marginTop: 8, cursor: 'pointer', borderRadius: 8, overflow: 'hidden', border: '1px solid var(--border-light)' }} onClick={() => setExpandedImage(msg.image)}>
                          <img src={msg.image} alt="Scanned Document Thumbnail" style={{ width: '100%', maxHeight: 150, objectFit: 'cover', display: 'block' }} />
                          <div style={{ background: 'var(--bg-subtle)', padding: '6px 12px', fontSize: 11, color: 'var(--text-muted)', textAlign: 'center', fontWeight: 600 }}>Click to expand image</div>
                        </div>
                      )}
                    </div>
                  ))
                )}
                <div ref={logEndRef} />
              </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        body { background-color: #f5f5f5; }
      `}</style>
      
      {/* Expanded Image Modal */}
      {expandedImage && (
        <div style={{ position: 'fixed', inset: 0, zIndex: 3000, background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(4px)', display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '16px 24px', display: 'flex', justifyContent: 'flex-end', gap: 12 }}>
            <button onClick={() => {
              const link = document.createElement('a');
              link.href = expandedImage;
              link.download = `scanned_document_${Date.now()}.jpg`;
              link.click();
            }} style={{ background: 'rgba(59,130,246,0.8)', border: 'none', color: 'white', borderRadius: 8, height: 40, padding: '0 16px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, cursor: 'pointer', fontSize: 13, fontWeight: 600 }}>
              <Download size={18} /> Download
            </button>
            <button onClick={() => setExpandedImage(null)} style={{ background: 'rgba(255,255,255,0.1)', border: 'none', color: 'white', borderRadius: '50%', width: 40, height: 40, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}>
              <X size={24} />
            </button>
          </div>
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24, overflow: 'hidden' }}>
            <img src={expandedImage} alt="Expanded Document" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', borderRadius: 8, boxShadow: '0 25px 50px rgba(0,0,0,0.5)' }} />
          </div>
        </div>
      )}
    </div>
  );
}
