import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useSocketEngine } from '../hooks/useSocketEngine';
import { Briefcase, Mic, Square, Send, Activity, LogOut, Check, X, MessageSquare, Terminal, Zap, ShieldCheck, RotateCcw, AlertTriangle } from 'lucide-react';

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
          const backendIp = window.location.hostname;
          const res = await fetch(`http://${backendIp}:8000/api/transcribe`, { method: 'POST', body: form });
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
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: 'var(--bg-page)' }}>

      {/* ── Sidebar ── */}
      <nav style={{ width: 80, background: 'rgba(2,6,23,0.95)', borderRight: '1px solid var(--border-light)', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '32px 0', gap: 32, flexShrink: 0, zIndex: 100 }}>
        <div style={{ width: 48, height: 48, background: 'linear-gradient(135deg, var(--primary), var(--accent))', borderRadius: 14, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'white', boxShadow: '0 4px 15px var(--primary-glow)' }}>
          <Briefcase size={22} />
        </div>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div style={{ width: 48, height: 48, display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 14, background: 'rgba(59,130,246,0.1)', color: 'var(--primary)', border: '1px solid var(--primary-glow)' }}>
            <Terminal size={22} />
          </div>
          <div style={{ width: 48, height: 48, display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 14, color: 'var(--text-faint)', cursor: 'pointer' }}>
            <Zap size={22} />
          </div>
        </div>
        <div onClick={() => navigate('/login')} style={{ width: 48, height: 48, display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 14, color: 'var(--danger)', cursor: 'pointer', background: 'rgba(239,68,68,0.05)' }}>
          <LogOut size={22} />
        </div>
      </nav>

      {/* ── Main Workspace ── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: '32px 40px', gap: 32 }}>

        {/* Header */}
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h1 style={{ fontFamily: 'var(--font-mono)', fontSize: 24, fontWeight: 700, letterSpacing: -0.5, textTransform: 'uppercase' }}>Command Interface</h1>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '8px 20px', borderRadius: 30, background: 'rgba(16,185,129,0.05)', border: '1px solid rgba(16,185,129,0.2)' }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--success)', boxShadow: '0 0 10px var(--success)' }} />
            <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--success)', letterSpacing: 1, textTransform: 'uppercase' }}>
              Satellite Link Secure {sessionId ? `[ID: ${sessionId.slice(0,8)}]` : ''}
            </span>
          </div>
        </header>

        {/* Grid */}
        <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 400px', gap: 32, overflow: 'hidden' }}>

          {/* ══ Left Panel: Kiosk Downlink ══ */}
          <div className="surface-card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '20px 28px', borderBottom: '1px solid var(--border-light)', display: 'flex', alignItems: 'center', gap: 12, fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 2 }}>
              <ShieldCheck size={16} color="var(--primary)" /> Live Kiosk Transmission
            </div>
            
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', padding: 40, position: 'relative' }}>
              {/* Session Request Alert */}
              {sessionRequest && (
                <div className="animate-enter" style={{
                  position: 'absolute', top: 32, left: '50%', transform: 'translateX(-50%)',
                  background: 'rgba(59,130,246,0.1)', border: '1px solid var(--primary-glow)',
                  padding: '24px 40px', borderRadius: 20, textAlign: 'center', zIndex: 100,
                  boxShadow: '0 10px 40px rgba(0,0,0,0.5)', backdropFilter: 'blur(10px)'
                }}>
                  <div style={{ color: 'var(--primary)', fontSize: 14, fontWeight: 700, letterSpacing: 2, textTransform: 'uppercase', marginBottom: 20 }}>
                    Inbound Handshake Request
                    <div style={{ fontSize: 18, color: 'white', marginTop: 10, fontFamily: 'var(--font-mono)' }}>
                      ID: {sessionRequest?.session_id?.slice(0, 8).toUpperCase() || 'UNKNOWN'}
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 16, justifyContent: 'center' }}>
                    <button className="btn btn-primary" onClick={acceptSession} style={{ padding: '12px 28px', fontSize: 12 }}>
                      Establish Link
                    </button>
                    <button className="btn btn-secondary" onClick={declineSession} style={{ padding: '12px 28px', fontSize: 12 }}>
                      Abort
                    </button>
                  </div>
                </div>
              )}

              {/* Multi-Person Alert */}
              {multiPersonAlert && (
                <div className="animate-enter" style={{
                  position: 'absolute', top: 20, left: '50%', transform: 'translateX(-50%)',
                  background: 'var(--danger)', color: 'white', padding: '10px 24px', borderRadius: 12,
                  fontSize: 12, fontWeight: 700, zIndex: 200, display: 'flex', alignItems: 'center', gap: 10
                }}>
                  <AlertTriangle size={16} /> MULTIPLE USERS DETECTED AT KIOSK
                </div>
              )}

              {/* Message Display */}
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
                {lastKioskMsg ? (
                  <div className="animate-enter">
                    <div style={{ fontSize: 56, fontWeight: 800, lineHeight: 1.1, letterSpacing: -2, background: 'linear-gradient(135deg, #fff, var(--primary))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: 32 }}>
                      "{lastKioskMsg.text}"
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 16, marginBottom: 32 }}>
                      {lastKioskMsg.word && (
                        <span className="badge badge-purple">{lastKioskMsg.word}</span>
                      )}
                      {lastKioskMsg.conf != null && (
                        <span className="badge badge-green">{Math.round(lastKioskMsg.conf * 100)}% Match</span>
                      )}
                    </div>
                    <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
                      <button className="btn btn-primary" onClick={() => sendReply('Yes, that is correct.')} style={{ background: 'var(--success)', border: 'none', padding: '10px 20px', fontSize: 12 }}>
                        <Check size={14} /> CORRECT
                      </button>
                      <button className="btn btn-secondary" onClick={() => sendReply('Could you please sign that again?')} style={{ background: 'rgba(239,68,68,0.1)', color: 'var(--danger)', border: '1px solid var(--danger)', padding: '10px 20px', fontSize: 12 }}>
                        <RotateCcw size={14} /> RETRY
                      </button>
                    </div>
                  </div>
                ) : (
                  <div style={{ opacity: 0.2, fontSize: 18, fontWeight: 700, letterSpacing: 3, textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                    Awaiting Uplink Signal
                  </div>
                )}
              </div>
            </div>

            {/* Bottom Control Bar */}
            <div style={{ padding: 32, borderTop: '1px solid var(--border-light)', display: 'flex', flexDirection: 'column', gap: 24 }}>
              <div style={{ display: 'flex', gap: 12, background: 'rgba(0,0,0,0.2)', padding: 8, borderRadius: 20, border: '1px solid var(--border-light)' }}>
                <input type="text" value={inputText} onChange={e => setInputText(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleSend()}
                  placeholder="Manual override transmission..."
                  style={{ flex: 1, background: 'transparent', border: 'none', padding: '0 20px', fontSize: 16, outline: 'none' }} />

                <button onClick={toggleMic} disabled={isTranscribing}
                  style={{
                    width: 48, height: 48, borderRadius: 14, border: `1px solid ${isMicRecording ? 'var(--danger)' : 'var(--border-light)'}`,
                    background: isMicRecording ? 'var(--danger-bg)' : 'rgba(255,255,255,0.03)',
                    color: isMicRecording ? 'var(--danger)' : 'var(--text-muted)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', transition: '0.3s'
                  }}>
                  {isTranscribing ? <div className="spin-icon" style={{ width: 20, height: 20, border: '2px solid currentColor', borderTopColor: 'transparent', borderRadius: '50%' }} /> : isMicRecording ? <Square size={20} fill="currentColor" /> : <Mic size={20} />}
                </button>

                <button className="btn btn-primary" onClick={handleSend} style={{ width: 48, height: 48, padding: 0, borderRadius: 14 }}>
                  <Send size={20} />
                </button>
              </div>

              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
                {quickReplies.map(text => (
                  <button key={text} onClick={() => quickReply(text)}
                    style={{ padding: '10px 20px', borderRadius: 14, background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-light)', color: 'var(--text-muted)', fontSize: 13, fontWeight: 500, cursor: 'pointer', transition: '0.2s' }}>
                    {text}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* ══ Right Panel: Transmission Logs ══ */}
          <div className="surface-card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ padding: '20px 28px', borderBottom: '1px solid var(--border-light)', display: 'flex', alignItems: 'center', gap: 12, fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 2 }}>
              <MessageSquare size={16} color="var(--primary)" /> Transmission Logs
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: 24, display: 'flex', flexDirection: 'column', gap: 16 }}>
              {messages.length === 0 ? (
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-faint)', fontSize: 12, textTransform: 'uppercase', letterSpacing: 2 }}>
                  Log Empty
                </div>
              ) : (
                messages.map((msg) => (
                  <div key={msg.id} className="animate-enter" style={{
                    padding: 20, borderRadius: 20, background: 'rgba(255,255,255,0.02)',
                    borderLeft: `3px solid ${msg.type === 'rx' ? 'var(--accent)' : msg.type === 'tx' ? 'var(--primary)' : 'var(--text-faint)'}`
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                      <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase', color: msg.type === 'rx' ? 'var(--accent)' : 'var(--primary)' }}>{msg.label}</span>
                      <span style={{ fontSize: 10, color: 'var(--text-faint)' }}>{msg.time}</span>
                    </div>
                    <div style={{ fontSize: 15, lineHeight: 1.5 }}>{msg.text}</div>
                  </div>
                ))
              )}
              <div ref={logEndRef} />
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .badge { padding: 8px 16px; border-radius: 10px; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 1.5px; }
        .badge-purple { background: rgba(139,92,246,0.1); color: #c4b5fd; border: 1px solid rgba(139,92,246,0.2); }
        .badge-green { background: rgba(16,185,129,0.1); color: var(--success); border: 1px solid rgba(16,185,129,0.2); }
        .spin-icon { animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
