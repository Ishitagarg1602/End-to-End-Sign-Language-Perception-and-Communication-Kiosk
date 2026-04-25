import React, { useEffect, useState, useRef, useCallback, Suspense } from 'react';
import { useSocketEngine } from '../hooks/useSocketEngine';
import { Camera, HandMetal, Send, RotateCcw, Square, MessageSquare, AlertTriangle, Loader2, MessageCircle, Sparkles, Activity, ArrowRight } from 'lucide-react';
import AvatarScene, { getGestureForText } from '../components/AvatarScene';

export default function KioskDashboard() {
  const {
    socket, isConnected, sessionId, sessionActive, waitingApproval,
    detectionState, latestSign, confirmedWords, messages,
    employeeMessage, multiPersonAlert,
    stopSigning, confirmSign, retrySign, endSession, dismissEmployeeMessage
  } = useSocketEngine('kiosk');

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [selectedIntent, setSelectedIntent] = useState(null);
  const [curSentence, setCurSentence] = useState('');

  // Typewriter effect for employee message
  const [displayedText, setDisplayedText] = useState('');
  const [typewriterDone, setTypewriterDone] = useState(false);

  // Camera init
  useEffect(() => {
    async function setupCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (err) {
        console.error('Camera error:', err);
      }
    }
    setupCamera();
    return () => {
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(t => t.stop());
      }
    };
  }, []);

  // Frame broadcast
  useEffect(() => {
    let intervalId;
    if (socket && isConnected) {
      intervalId = setInterval(() => {
        if (videoRef.current && canvasRef.current && videoRef.current.readyState >= 2) {
          const canvas = canvasRef.current;
          canvas.width = 320;
          canvas.height = 240;
          canvas.getContext('2d').drawImage(videoRef.current, 0, 0, 320, 240);
          socket.emit('video_frame', { image: canvas.toDataURL('image/jpeg', 0.6) });
        }
      }, 100);
    }
    return () => { if (intervalId) clearInterval(intervalId); };
  }, [socket, isConnected]);

  // Sign detected → set default intent
  useEffect(() => {
    if (latestSign) {
      if (latestSign.intent_options?.length > 0) {
        setSelectedIntent(latestSign.intent_options[0]);
        setCurSentence(latestSign.intent_options[0].sentence);
      } else {
        setSelectedIntent(null);
        setCurSentence(latestSign.sentence);
      }
    }
  }, [latestSign]);

  // Typewriter effect for employee message
  useEffect(() => {
    if (employeeMessage) {
      // Play notification tune
      try {
        const audio = new Audio('https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3');
        audio.volume = 0.4;
        audio.play();
      } catch (e) { console.error('Audio play failed', e); }

      setDisplayedText('');
      setTypewriterDone(false);
      let i = 0;
      const interval = setInterval(() => {
        if (i < employeeMessage.length) {
          setDisplayedText(employeeMessage.slice(0, i + 1));
          i++;
        } else {
          setTypewriterDone(true);
          clearInterval(interval);
        }
      }, 35);
      return () => clearInterval(interval);
    }
  }, [employeeMessage]);

  const getInstructionCard = (text) => {
    if (!text) return null;
    const lower = text.toLowerCase();
    if (lower.includes('room') || lower.includes('floor') || lower.includes('counter') || lower.includes('go to')) {
      return {
        title: 'Directional Instruction',
        desc: 'Please follow the directions to the specified location.',
        icon: <ArrowRight size={24} className="animate-pulse" />
      };
    }
    if (lower.includes('wait') || lower.includes('moment')) {
      return {
        title: 'Please Wait',
        desc: 'A representative is processing your request.',
        icon: <Loader2 size={24} className="spin-icon" />
      };
    }
    return null;
  };
  const instruction = getInstructionCard(employeeMessage);

  const handleConfirm = () => {
    if (!latestSign) return;
    confirmSign(latestSign.word, curSentence, latestSign.confidence, selectedIntent?.label);
    setSelectedIntent(null);
    setCurSentence('');
  };

  const handleRetry = () => {
    retrySign();
    setSelectedIntent(null);
    setCurSentence('');
  };

  const handleIntentSelect = (opt) => {
    setSelectedIntent(opt);
    setCurSentence(opt.sentence);
  };

  // Status bar config
  const statusConfig = {
    idle: { text: 'Awaiting User…', bg: 'rgba(0,0,0,0.6)', anim: false, label: 'STANDBY' },
    waiting_approval: { text: 'Establishing Link…', bg: '#F59E0B', anim: true, label: 'HANDSHAKE' },
    scanning: { text: 'AI Recording Active', bg: 'var(--primary)', anim: true, label: 'SCANNING' },
    paused: { text: 'Analysis Complete', bg: '#F59E0B', anim: false, label: 'PAUSED' },
    processing: { text: 'Processing AI Inference…', bg: '#8B5CF6', anim: true, label: 'AI THINKING' }
  };
  const status = statusConfig[detectionState] || statusConfig.idle;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '24px', padding: '24px', height: '100vh', overflow: 'hidden', position: 'relative' }}>

      {/* ── MULTI-PERSON ALERT ── */}
      {multiPersonAlert && (
        <div className="animate-enter" style={{ position: 'fixed', top: 56, left: '50%', transform: 'translateX(-50%)', zIndex: 3000,
          background: 'var(--danger)', color: 'white', border: 'none',
          padding: '12px 32px', borderRadius: '16px', fontWeight: 700, fontSize: 14, display: 'flex', alignItems: 'center', gap: 12, boxShadow: '0 8px 32px rgba(239, 68, 68, 0.4)' }}>
          <AlertTriangle size={18} /> Multiple users detected. Please remain alone.
        </div>
      )}

      {/* ── EMPLOYEE MESSAGE MODAL ── */}
      {employeeMessage && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(3,7,18,0.9)', zIndex: 4000, display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(12px)' }}>
          <div className="animate-enter" style={{
            width: '90vw', maxWidth: 720, background: 'var(--bg-surface)',
            borderRadius: 32, overflow: 'hidden', boxShadow: '0 0 50px var(--primary-glow)',
            border: '1px solid var(--primary)'
          }}>
            <div style={{ display: 'flex', height: 320, position: 'relative' }}>
              <div style={{ flex: 1 }}>
                <Suspense fallback={
                  <div style={{ height: 320, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000' }}>
                    <Loader2 size={32} className="spin-icon" style={{ color: 'var(--primary)' }} />
                  </div>
                }>
                  <AvatarScene gesture={getGestureForText(employeeMessage)} isActive={true} />
                </Suspense>
              </div>

              {instruction && (
                <div className="animate-enter" style={{
                  width: 240, background: 'rgba(99, 102, 241, 0.1)', borderLeft: '1px solid var(--primary)',
                  display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 24, textAlign: 'center'
                }}>
                  <div style={{ color: 'var(--primary)', marginBottom: 16 }}>{instruction.icon}</div>
                  <div style={{ fontSize: 13, fontWeight: 800, color: 'var(--primary)', textTransform: 'uppercase', marginBottom: 8 }}>{instruction.title}</div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{instruction.desc}</div>
                </div>
              )}
              <div style={{
                position: 'absolute', top: 20, left: '50%', transform: 'translateX(-50%)',
                display: 'flex', alignItems: 'center', gap: 10,
                background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(8px)',
                padding: '10px 24px', borderRadius: 30, color: 'var(--primary)',
                fontSize: 12, fontWeight: 700, letterSpacing: 2, textTransform: 'uppercase'
              }}>
                <MessageSquare size={16} /> Staff Communication
              </div>
            </div>

            <div style={{ padding: '40px' }}>
              <div style={{
                fontSize: 32, fontWeight: 700, lineHeight: 1.3, color: 'var(--text-main)',
                marginBottom: 32, minHeight: 90, textAlign: 'center'
              }}>
                "{displayedText}"
                {!typewriterDone && <span style={{ animation: 'blink 0.7s infinite', color: 'var(--primary)' }}>|</span>}
              </div>

              <button className="btn btn-primary" onClick={dismissEmployeeMessage} style={{ width: '100%', padding: 18, fontSize: 18 }}>
                ACKNOWLEDGE & CONTINUE
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ══════ LEFT: Camera Feed ══════ */}
      <div className="surface-card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '16px 24px', borderBottom: '1px solid var(--border-light)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 12, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1.5 }}>
            <Camera size={16} color="var(--primary)" /> Vision Input
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: isConnected ? 'var(--success)' : 'var(--danger)', boxShadow: isConnected ? '0 0 10px var(--success)' : 'none' }} />
            <span style={{ fontSize: 11, fontWeight: 700, color: isConnected ? 'var(--success)' : 'var(--danger)' }}>
              {isConnected ? `LINK ACTIVE ${sessionId ? `[ID: ${sessionId.slice(0,8)}]` : ''}` : 'OFFLINE'}
            </span>
          </div>
        </div>

        <div style={{ flex: 1, position: 'relative', background: '#000', margin: 20, borderRadius: 20, overflow: 'hidden', border: '1px solid var(--border-light)', boxShadow: detectionState === 'scanning' ? '0 0 30px var(--primary-glow)' : 'none', transition: 'box-shadow 0.5s ease' }}>
          <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }} />
          <canvas ref={canvasRef} style={{ display: 'none' }} />

          {/* Status Label Overlay */}
          <div style={{
            position: 'absolute', top: 20, left: 20,
            padding: '8px 16px', borderRadius: 12,
            fontSize: 12, fontWeight: 700, color: 'white',
            background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(8px)',
            border: '1px solid rgba(255,255,255,0.1)',
            display: 'flex', alignItems: 'center', gap: 10
          }}>
            {detectionState === 'scanning' && <div style={{ width: 8, height: 8, background: 'var(--danger)', borderRadius: '50%', animation: 'pulse 1s infinite' }} />}
            {status.label}
          </div>

          {/* Status Bar Bottom */}
          <div style={{
            position: 'absolute', bottom: 20, left: 20, right: 20,
            padding: '12px 24px', borderRadius: 16,
            fontSize: 14, fontWeight: 600, textAlign: 'center',
            background: 'rgba(0,0,0,0.7)', color: '#fff', backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255,255,255,0.05)',
            boxShadow: '0 8px 32px rgba(0,0,0,0.4)'
          }}>
            {status.text}
          </div>
        </div>

        {/* Done Signing button */}
        {sessionActive && detectionState === 'scanning' && (
          <button className="btn btn-primary" onClick={stopSigning}
            style={{ margin: '0 20px 20px', padding: 16, fontSize: 16, borderRadius: 16 }}>
            <Square size={20} fill="white" /> FINISH SIGNING
          </button>
        )}
      </div>

      {/* ══════ RIGHT: Recognition Results ══════ */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 24, overflow: 'hidden' }}>
        <div className="surface-card" style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{ padding: '16px 24px', borderBottom: '1px solid var(--border-light)', display: 'flex', alignItems: 'center', gap: 10, fontSize: 12, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1.5 }}>
            <Sparkles size={16} color="var(--primary)" /> AI Recognition Engine
          </div>

          <div style={{ flex: 1, padding: 24, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 20 }}>
            {/* IDLE state */}
            {!latestSign && (
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16, color: 'var(--text-faint)', textAlign: 'center' }}>
                <Activity size={48} strokeWidth={1} />
                <p style={{ fontSize: 14, lineHeight: 1.6 }}>
                  Position your hands within the frame<br />and begin signing clearly.
                </p>
              </div>
            )}

            {/* DETECTED state */}
            {latestSign && (
              <div className="animate-enter" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 48, fontWeight: 800, letterSpacing: -1, background: 'linear-gradient(135deg, #fff, var(--primary))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', marginBottom: 12 }}>
                    {latestSign.word.toUpperCase()}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <div style={{ flex: 1, height: 6, background: 'var(--bg-subtle)', borderRadius: 3, overflow: 'hidden' }}>
                      <div style={{ height: '100%', background: 'var(--primary)', width: `${Math.round(latestSign.confidence * 100)}%`, transition: 'width 0.8s cubic-bezier(0.16, 1, 0.3, 1)' }} />
                    </div>
                    <span style={{ fontSize: 14, fontWeight: 800, color: 'var(--primary)' }}>{Math.round(latestSign.confidence * 100)}%</span>
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1.5 }}>Select Intent</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                    {(latestSign.intent_options || []).map((opt, idx) => (
                      <button key={idx} onClick={() => handleIntentSelect(opt)}
                        style={{
                          display: 'block', width: '100%', padding: 20, borderRadius: 16, textAlign: 'left',
                          border: selectedIntent?.label === opt.label ? '1px solid var(--primary)' : '1px solid var(--border-light)',
                          background: selectedIntent?.label === opt.label ? 'var(--primary-glow)' : 'rgba(255,255,255,0.02)',
                          color: 'var(--text-main)', cursor: 'pointer', transition: '0.3s',
                          boxShadow: selectedIntent?.label === opt.label ? '0 0 20px var(--primary-glow)' : 'none'
                        }}>
                        <div style={{ fontWeight: 800, color: 'var(--primary)', fontSize: 11, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>{opt.label}</div>
                        <div style={{ fontSize: 15, fontWeight: 500 }}>{opt.sentence}</div>
                      </button>
                    ))}
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1.5fr 1fr', gap: 12 }}>
                  <button className="btn btn-primary" onClick={handleConfirm} style={{ padding: 18, fontSize: 16 }}>
                    TRANSMIT
                  </button>
                  <button className="btn btn-secondary" onClick={handleRetry} style={{ padding: 18, fontSize: 16 }}>
                    RETRY
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Confirmed History */}
          <div style={{ padding: '20px 24px', borderTop: '1px solid var(--border-light)', background: 'rgba(0,0,0,0.1)' }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 1.5, marginBottom: 12 }}>Communication Log</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {confirmedWords.map((w, i) => (
                <span key={i} style={{ padding: '6px 14px', background: 'var(--success-bg)', color: 'var(--success)', borderRadius: 12, fontSize: 13, fontWeight: 600, border: '1px solid rgba(16,185,129,0.2)' }}>
                  {w}
                </span>
              ))}
              {messages.filter(m => m.type === 'rx').map((m, i) => (
                <span key={'rx' + i} style={{ padding: '6px 14px', background: 'var(--primary-glow)', color: 'var(--primary)', borderRadius: 12, fontSize: 13, fontWeight: 600, border: '1px solid var(--primary-glow)' }}>
                  Staff: {m.text}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes pulse { 0%,100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.1); opacity: 0.5; } }
        @keyframes blink { 0%,100% { opacity: 1; } 50% { opacity: 0; } }
        .spin-icon { animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
