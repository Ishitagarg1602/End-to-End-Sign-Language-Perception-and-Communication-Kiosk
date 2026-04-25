import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useSocketEngine } from '../hooks/useSocketEngine';
import { Briefcase, Mic, Square, Send, Activity, LogOut, Check, X, MessageSquare } from 'lucide-react';

export default function EmployeeDashboard() {
  const {
    isConnected, sessionId, sessionRequest, sessionActive,
    messages, acceptSession, declineSession, sendReply,
    multiPersonAlert
  } = useSocketEngine('employee');

  const navigate = useNavigate();
  const [inputText, setInputText] = useState('');
  const logEndRef = useRef(null);

  // Mic state
  const [isMicRecording, setIsMicRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const lastKioskMsg = messages.filter(m => m.type === 'rx').slice(-1)[0];

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = () => {
    if (!inputText.trim()) return;
    sendReply(inputText);
    setInputText('');
  };

  const quickReply = (text) => sendReply(text);

  // Whisper Mic
  const initMic = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        audioChunksRef.current = [];
        setIsTranscribing(true);
        try {
          const form = new FormData();
          form.append('audio', blob, 'recording.webm');
          const res = await fetch('http://localhost:8000/api/transcribe', { method: 'POST', body: form });
          const data = await res.json();
          if (data.text) sendReply(data.text);
        } catch (err) {
          console.error('Transcription error:', err);
        } finally {
          setIsTranscribing(false);
          setIsMicRecording(false);
        }
      };
      mediaRecorderRef.current = recorder;
    } catch (err) {
      alert('Microphone access denied or unavailable.');
    }
  }, [sendReply]);

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

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>

      {/* ── Sidebar ── */}
      <nav style={{ width: 72, background: 'var(--bg-surface)', borderRight: '1px solid var(--border-light)', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '20px 0', gap: 28, flexShrink: 0 }}>
        <div style={{ width: 40, height: 40, background: 'var(--primary)', borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', boxShadow: 'var(--shadow-md)' }}>
          <Briefcase size={20} />
        </div>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ width: 44, height: 44, display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 12, background: 'var(--bg-subtle)', color: 'var(--accent)', cursor: 'pointer' }}>
            <Activity size={20} />
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: isConnected ? 'var(--success)' : 'var(--danger)', boxShadow: isConnected ? '0 0 8px var(--success)' : 'none' }} />
            <span style={{ fontSize: 9, fontWeight: 600, color: 'var(--text-muted)', letterSpacing: 0.5 }}>{isConnected ? 'LIVE' : 'OFF'}</span>
          </div>
          <div onClick={() => navigate('/login')} style={{ width: 44, height: 44, display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 12, color: 'var(--danger)', cursor: 'pointer' }}>
            <LogOut size={20} />
          </div>
        </div>
      </nav>

      {/* ── Main Workspace ── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

        {/* Header */}
        <header style={{ padding: '20px 36px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border-light)', flexShrink: 0 }}>
          <h1 className="heading-display" style={{ fontSize: 18, letterSpacing: 0.5 }}>ISL Banking Interface</h1>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '6px 16px', borderRadius: 30, background: isConnected ? 'var(--success-bg)' : 'var(--danger-bg)', border: `1px solid ${isConnected ? 'rgba(5,150,105,0.2)' : 'rgba(220,38,38,0.2)'}` }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: isConnected ? 'var(--success)' : 'var(--danger)', boxShadow: isConnected ? '0 0 10px var(--success)' : 'none' }} />
            <span style={{ fontSize: 11, fontWeight: 700, color: isConnected ? 'var(--success)' : 'var(--danger)', letterSpacing: 1, textTransform: 'uppercase' }}>
              {isConnected ? 'Data Link Secure' : 'Offline'}
            </span>
          </div>
        </header>

        {/* Grid */}
        <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, padding: '24px 36px', overflow: 'hidden' }}>

          {/* ══ Left Panel: Kiosk Downlink ══ */}
          <div className="surface-card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-light)', display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>
              <MessageSquare size={14} color="var(--accent)" /> Kiosk Downlink
            </div>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 20 }}>

              {/* Session Request Alert */}
              {sessionRequest && (
                <div className="animate-enter" style={{
                  background: 'var(--accent-light)', border: '1px solid rgba(37,99,235,0.3)',
                  borderRadius: 16, padding: 24, textAlign: 'center', marginBottom: 20, flexShrink: 0,
                  boxShadow: 'var(--shadow-lg)'
                }}>
                  <div style={{ color: 'var(--accent)', fontSize: 14, fontWeight: 700, letterSpacing: 1.5, textTransform: 'uppercase', marginBottom: 20 }}>
                    New Communication Request
                  </div>
                  <div style={{ display: 'flex', gap: 16, justifyContent: 'center' }}>
                    <button className="btn" onClick={acceptSession}
                      style={{ background: 'var(--success-bg)', color: 'var(--success)', border: '1px solid rgba(5,150,105,0.3)', padding: '12px 28px', fontSize: 12, fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase' }}>
                      <Check size={16} /> Accept Uplink
                    </button>
                    <button className="btn btn-secondary" onClick={declineSession}
                      style={{ padding: '12px 28px', fontSize: 12, fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase' }}>
                      <X size={16} /> Reject
                    </button>
                  </div>
                </div>
              )}

              {/* Message Display */}
              <div style={{
                flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                textAlign: 'center', background: 'var(--bg-page)', borderRadius: 16, border: '1px dashed var(--border-light)'
              }}>
                {lastKioskMsg ? (
                  <div className="animate-enter">
                    <div style={{ fontSize: 36, fontWeight: 700, lineHeight: 1.3, letterSpacing: -0.5, color: 'var(--text-main)', marginBottom: 28 }}>
                      "{lastKioskMsg.text}"
                    </div>
                    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 14 }}>
                      {lastKioskMsg.word && (
                        <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase', padding: '6px 14px', borderRadius: 8, background: 'rgba(139,92,246,0.08)', color: '#7C3AED', border: '1px solid rgba(139,92,246,0.2)' }}>
                          {lastKioskMsg.word}
                        </span>
                      )}
                      {lastKioskMsg.conf != null && (
                        <span style={{ fontSize: 11, fontWeight: 700, padding: '6px 14px', borderRadius: 8,
                          background: lastKioskMsg.conf >= 0.75 ? 'var(--success-bg)' : 'rgba(245,158,11,0.1)',
                          color: lastKioskMsg.conf >= 0.75 ? 'var(--success)' : '#F59E0B' }}>
                          {Math.round(lastKioskMsg.conf * 100)}%
                        </span>
                      )}
                      <span style={{ fontSize: 11, color: 'var(--text-faint)', letterSpacing: 1 }}>{lastKioskMsg.time}</span>
                    </div>
                  </div>
                ) : (
                  <div style={{ fontSize: 15, color: 'var(--text-faint)', fontWeight: 500, letterSpacing: 2, textTransform: 'uppercase' }}>
                    No Signal Detected
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* ══ Right Panel: Transmission Control ══ */}
          <div className="surface-card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--border-light)', display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1 }}>
              <Send size={14} color="var(--accent)" /> Transmission Control
            </div>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 20, overflow: 'hidden' }}>

              {/* Input Bar */}
              <div style={{ marginBottom: 20, flexShrink: 0 }}>
                <div style={{ fontSize: 10, fontWeight: 600, color: 'var(--text-faint)', letterSpacing: 1, textTransform: 'uppercase', marginBottom: 6, textAlign: 'right' }}>
                  Voice-to-Text Active
                </div>
                <div style={{ display: 'flex', gap: 10, background: 'var(--bg-subtle)', padding: 8, borderRadius: 16, border: '1px solid var(--border-light)', alignItems: 'center' }}>
                  <input type="text" value={inputText} onChange={e => setInputText(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleSend()}
                    placeholder="Type response or use microphone..."
                    style={{ flex: 1, background: 'transparent', border: 'none', padding: '0 16px', fontSize: 15, outline: 'none', boxShadow: 'none' }} />

                  <button onClick={toggleMic} disabled={isTranscribing}
                    style={{
                      width: 48, height: 48, borderRadius: 12, border: `1px solid ${isMicRecording ? 'var(--danger)' : 'var(--border-light)'}`,
                      background: isMicRecording ? 'var(--danger-bg)' : 'var(--bg-surface)',
                      color: isMicRecording ? 'var(--danger)' : 'var(--text-muted)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', transition: '0.3s', flexShrink: 0
                    }}>
                    {isTranscribing ? (
                      <div style={{ width: 18, height: 18, border: '2px solid var(--text-muted)', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
                    ) : isMicRecording ? (
                      <Square size={20} fill="currentColor" />
                    ) : (
                      <Mic size={20} />
                    )}
                  </button>

                  <button className="btn btn-blue" onClick={handleSend} style={{ height: 48, padding: '0 20px', borderRadius: 12 }}>
                    <Send size={18} />
                  </button>
                </div>
              </div>

              {/* Quick Replies */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20, paddingBottom: 20, borderBottom: '1px solid var(--border-light)', flexShrink: 0 }}>
                {quickReplies.map(text => (
                  <button key={text} onClick={() => quickReply(text)}
                    style={{ padding: '8px 16px', borderRadius: 30, background: 'var(--bg-surface)', border: '1px solid var(--border-light)', color: 'var(--text-muted)', fontSize: 13, fontWeight: 500, cursor: 'pointer', transition: '0.2s', fontFamily: 'var(--font-sans)' }}>
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
                      background: msg.type === 'rx' ? 'linear-gradient(90deg, rgba(139,92,246,0.04) 0%, transparent 100%)' : msg.type === 'sys' ? 'var(--bg-subtle)' : 'var(--bg-page)',
                      border: '1px solid transparent',
                      borderLeft: `3px solid ${msg.type === 'rx' ? '#8B5CF6' : msg.type === 'tx' ? 'var(--accent)' : msg.text.startsWith('✓') ? 'var(--success)' : 'var(--text-faint)'}`,
                      display: 'flex', flexDirection: 'column', gap: 5
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase',
                          color: msg.type === 'rx' ? '#8B5CF6' : msg.type === 'tx' ? 'var(--accent)' : msg.text.startsWith('✓') ? 'var(--success)' : 'var(--text-faint)' }}>
                          {msg.label}
                        </span>
                        <span style={{ fontSize: 10, color: 'var(--text-faint)', letterSpacing: 1 }}>{msg.time}</span>
                      </div>
                      <div style={{ fontSize: 14, fontWeight: 400, lineHeight: 1.5, color: 'var(--text-main)' }}>{msg.text}</div>
                    </div>
                  ))
                )}
                <div ref={logEndRef} />
              </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
