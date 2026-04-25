import React, { useEffect, useState, useRef, useCallback, Suspense } from 'react';
import { useSocketEngine } from '../hooks/useSocketEngine';
import { Camera, HandMetal, Send, RotateCcw, Square, MessageSquare, AlertTriangle, Loader2, MessageCircle } from 'lucide-react';
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
  const [cameraError, setCameraError] = useState(null);

  // Camera init
  useEffect(() => {
    async function setupCamera() {
      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          throw new Error('Camera API not supported or blocked by browser security (HTTP).');
        }
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 640, height: 480, frameRate: { ideal: 30 } } 
        });
        if (videoRef.current) videoRef.current.srcObject = stream;
        setCameraError(null);
      } catch (err) {
        console.error('Camera error:', err);
        setCameraError(err.message);
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
    idle: { text: 'Waiting for user…', bg: 'rgba(100,116,139,0.85)', anim: false },
    waiting_approval: { text: 'Connecting to representative…', bg: '#F59E0B', anim: true },
    scanning: { text: 'Recording — sign now, click "Done Signing" when finished', bg: 'var(--accent)', anim: true },
    paused: { text: 'Select intent and send, or retry', bg: '#F59E0B', anim: false },
    processing: { text: 'Processing sign…', bg: '#8B5CF6', anim: true }
  };
  const status = statusConfig[detectionState] || statusConfig.idle;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', padding: '12px', height: '100vh', overflow: 'hidden', position: 'relative' }}>

      {/* ── WAITING APPROVAL OVERLAY ── */}
      {waitingApproval && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 3000,
          background: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(12px)',
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 24
        }}>
          <div style={{ width: 280, height: 280 }}>
            <Suspense fallback={null}>
              <AvatarScene gesture="wave" isActive={true} />
            </Suspense>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <Loader2 size={24} className="spin-icon" style={{ color: 'var(--accent)' }} />
            <span style={{ fontSize: 20, fontWeight: 600, color: 'var(--text-main)', letterSpacing: -0.3 }}>
              Connecting to bank representative…
            </span>
          </div>
          <p style={{ fontSize: 14, color: 'var(--text-muted)', maxWidth: 400, textAlign: 'center', lineHeight: 1.6 }}>
            Please wait while a bank employee accepts your session. You will be able to sign once connected.
          </p>
        </div>
      )}

      {/* ── MULTI-PERSON ALERT ── */}
      {multiPersonAlert && (
        <div className="animate-enter" style={{ position: 'fixed', top: 56, left: '50%', transform: 'translateX(-50%)', zIndex: 1000,
          background: 'var(--danger-bg)', color: 'var(--danger)', border: '1px solid var(--danger)',
          padding: '10px 28px', borderRadius: '12px', fontWeight: 600, fontSize: 13, display: 'flex', alignItems: 'center', gap: 8, boxShadow: 'var(--shadow-lg)' }}>
          <AlertTriangle size={16} /> Multiple people detected — only one user allowed
        </div>
      )}

      {/* ── EMPLOYEE MESSAGE + AVATAR MODAL ── */}
      {employeeMessage && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(4px)' }}>
          <div className="animate-enter" style={{
            width: '90vw', maxWidth: 720, background: 'var(--bg-surface)',
            borderRadius: 24, overflow: 'hidden', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.25)',
            border: '1px solid var(--border-light)'
          }}>
            {/* Avatar Section */}
            <div style={{ height: 300, position: 'relative' }}>
              <Suspense fallback={
                <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#0B1120' }}>
                  <Loader2 size={32} className="spin-icon" style={{ color: '#60A5FA' }} />
                </div>
              }>
                <AvatarScene gesture={getGestureForText(employeeMessage)} isActive={true} />
              </Suspense>
              {/* Header badge */}
              <div style={{
                position: 'absolute', top: 16, left: '50%', transform: 'translateX(-50%)',
                display: 'flex', alignItems: 'center', gap: 8,
                background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(8px)',
                padding: '8px 20px', borderRadius: 30, color: '#60A5FA',
                fontSize: 12, fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase'
              }}>
                <MessageSquare size={14} /> Bank Representative
              </div>
            </div>

            {/* Message Section */}
            <div style={{ padding: '28px 32px' }}>
              <div style={{
                fontSize: 24, fontWeight: 600, lineHeight: 1.5, color: 'var(--text-main)',
                marginBottom: 8, minHeight: 72
              }}>
                "{displayedText}"
                {!typewriterDone && <span style={{ animation: 'blink 0.7s infinite', color: 'var(--accent)' }}>|</span>}
              </div>

              {/* Action button — user replies only through sign language */}
              <div style={{ marginTop: 20 }}>
                <button className="btn btn-primary" onClick={dismissEmployeeMessage} style={{ width: '100%', padding: 16, fontSize: 16, borderRadius: 14 }}>
                  <MessageCircle size={18} /> Acknowledge & Continue Signing
                </button>
                <p style={{ textAlign: 'center', fontSize: 12, color: 'var(--text-faint)', marginTop: 10 }}>
                  Use sign language to reply to the representative
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ══════ LEFT: Camera Feed ══════ */}
      <div className="surface-card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border-light)', display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.6 }}>
          <Camera size={14} color="var(--accent)" /> Camera Feed
        </div>

        <div style={{ flex: 1, position: 'relative', background: '#000', margin: 8, borderRadius: 'var(--radius-sm)', overflow: 'hidden', minHeight: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {cameraError && (
            <div style={{ padding: 20, textAlign: 'center', color: '#f87171', fontSize: 13 }}>
              <AlertTriangle size={32} style={{ marginBottom: 12, margin: '0 auto' }} />
              <p style={{ fontWeight: 600, marginBottom: 8 }}>Camera Access Failed</p>
              <p style={{ opacity: 0.8 }}>{cameraError}</p>
              <p style={{ marginTop: 12, fontSize: 11, color: 'var(--text-muted)' }}>
                Tip: If using an IP address (not localhost), Chrome requires HTTPS or a "Secure Origins" override.
              </p>
            </div>
          )}
          <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)', display: cameraError ? 'none' : 'block' }} />
          <canvas ref={canvasRef} style={{ display: 'none' }} />

          {/* Status bar */}
          <div style={{
            position: 'absolute', bottom: 8, left: 8, right: 8,
            padding: '8px 16px', borderRadius: 8,
            fontSize: 12, fontWeight: 600, textAlign: 'center',
            background: status.bg, color: '#fff',
            animation: status.anim ? 'pulse 1.5s infinite' : 'none'
          }}>
            {status.text}
          </div>

          {/* Connection badge */}
          <div style={{ position: 'absolute', top: 10, left: 10, display: 'flex', alignItems: 'center', gap: 8,
            background: 'rgba(255,255,255,0.9)', padding: '6px 14px', borderRadius: 20, fontSize: 12, fontWeight: 600, boxShadow: 'var(--shadow-md)' }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: isConnected ? 'var(--success)' : 'var(--danger)' }} />
            {isConnected ? 'Connected' : 'Disconnected'}
          </div>

          {/* End Session */}
          {sessionActive && (
            <button onClick={endSession}
              style={{ position: 'absolute', top: 10, right: 10, padding: '5px 14px', borderRadius: 8, border: '1px solid rgba(220,38,38,0.4)', background: 'rgba(220,38,38,0.1)', color: '#f87171', fontFamily: 'var(--font-sans)', fontSize: 11, fontWeight: 600, cursor: 'pointer' }}>
              End Session
            </button>
          )}
        </div>

        {/* Done Signing button — only when session accepted and scanning */}
        {sessionActive && detectionState === 'scanning' && (
          <button className="btn" onClick={stopSigning}
            style={{ margin: '0 8px 8px', padding: 10, fontSize: 14, background: 'var(--danger)', color: 'white', borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
            <Square size={16} fill="white" /> Done Signing
          </button>
        )}
      </div>

      {/* ══════ RIGHT: Recognition Results ══════ */}
      <div className="surface-card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border-light)', display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.6 }}>
          <HandMetal size={14} color="var(--accent)" /> Recognition Results
        </div>

        <div style={{ flex: 1, padding: '12px 16px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 10 }}>
          {/* IDLE / WAITING */}
          {!latestSign && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10, color: 'var(--text-faint)' }}>
              <HandMetal size={36} />
              <div style={{ fontSize: 13, textAlign: 'center', lineHeight: 1.7 }}>
                {sessionActive
                  ? <>Show your hands to the camera.<br />Sign clearly, then click <b style={{ color: 'var(--accent)' }}>"Done Signing"</b>.</>
                  : 'Waiting for session to start…'
                }
              </div>
            </div>
          )}

          {/* DETECTED state */}
          {latestSign && (
            <div className="animate-enter" style={{ display: 'flex', flexDirection: 'column', gap: 12, alignItems: 'center' }}>
              {/* Word */}
              <div style={{
                fontSize: 32, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 2, fontFamily: 'var(--font-display)',
                background: 'linear-gradient(135deg, var(--accent), #6366f1)',
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text'
              }}>
                {latestSign.word}
              </div>

              {/* Intent + Category badges */}
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
                {latestSign.intent && latestSign.intent !== 'unknown' && (
                  <span style={{ padding: '4px 12px', borderRadius: 8, fontSize: 11, fontWeight: 700, textTransform: 'uppercase',
                    background: 'var(--accent-light)', color: 'var(--accent)', border: '1px solid rgba(37,99,235,0.2)' }}>
                    Intent: {latestSign.intent.replace(/_/g, ' ')}
                  </span>
                )}
                {latestSign.category && latestSign.category !== 'general' && (
                  <span style={{ padding: '3px 10px', borderRadius: 6, fontSize: 11, fontWeight: 600, textTransform: 'uppercase',
                    background: 'rgba(139,92,246,0.08)', color: '#7C3AED' }}>
                    {latestSign.category}
                  </span>
                )}
              </div>

              {/* Confidence */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{
                  padding: '3px 10px', borderRadius: 10, fontSize: 12, fontWeight: 700,
                  background: latestSign.confidence >= 0.85 ? 'var(--success-bg)' : 'rgba(245,158,11,0.1)',
                  color: latestSign.confidence >= 0.85 ? 'var(--success)' : '#F59E0B'
                }}>
                  {Math.round(latestSign.confidence * 100)}%
                </span>
                <div style={{ width: 130, height: 4, background: 'var(--bg-subtle)', borderRadius: 4, overflow: 'hidden' }}>
                  <div style={{
                    height: '100%', borderRadius: 4, transition: 'width 0.5s',
                    width: `${Math.round(latestSign.confidence * 100)}%`,
                    background: latestSign.confidence >= 0.85 ? 'var(--success)' : '#F59E0B'
                  }} />
                </div>
              </div>

              {/* Intent Options */}
              {latestSign.intent_options?.length > 0 && (
                <div style={{ width: '100%' }}>
                  <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: 0.6, marginBottom: 6, textAlign: 'center' }}>
                    Select what you want to communicate
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {latestSign.intent_options.map((opt, idx) => (
                      <button key={idx} onClick={() => handleIntentSelect(opt)}
                        style={{
                          display: 'block', width: '100%', padding: '8px 14px', borderRadius: 8, textAlign: 'left',
                          border: selectedIntent?.label === opt.label ? '1px solid var(--accent)' : '1px solid var(--border-light)',
                          background: selectedIntent?.label === opt.label ? 'var(--accent-light)' : 'var(--bg-surface)',
                          color: 'var(--text-main)', fontFamily: 'var(--font-sans)', fontSize: 12, cursor: 'pointer', transition: '0.15s'
                        }}>
                        <span style={{ fontWeight: 700, color: 'var(--accent)', fontSize: 11, textTransform: 'uppercase', letterSpacing: 0.3 }}>
                          {opt.label}
                        </span><br />
                        {opt.sentence}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Top 3 */}
              {latestSign.top3?.length > 0 && (
                <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap', justifyContent: 'center' }}>
                  {latestSign.top3.map((t, i) => (
                    <span key={i} style={{
                      padding: '3px 8px', borderRadius: 6, fontSize: 11, fontWeight: i === 0 ? 600 : 500,
                      background: i === 0 ? 'var(--accent-light)' : 'var(--bg-subtle)',
                      color: i === 0 ? 'var(--accent)' : 'var(--text-muted)',
                      border: `1px solid ${i === 0 ? 'rgba(37,99,235,0.25)' : 'var(--border-light)'}`
                    }}>
                      {t.word} ({Math.round(t.confidence * 100)}%)
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Confirmed Words */}
        <div style={{ padding: '6px 16px', borderTop: '1px solid var(--border-light)' }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.6, marginBottom: 4 }}>
            Confirmed Words
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, minHeight: 20 }}>
            {confirmedWords.map((w, i) => (
              <span key={i} style={{ padding: '2px 8px', background: 'var(--success-bg)', color: 'var(--success)', borderRadius: 4, fontSize: 11, fontWeight: 600 }}>
                {w}
              </span>
            ))}
            {messages.filter(m => m.type === 'rx').map((m, i) => (
              <span key={'rx' + i} style={{ padding: '2px 8px', background: 'var(--accent-light)', color: 'var(--accent)', borderRadius: 4, fontSize: 11, fontWeight: 600, border: '1px solid rgba(37,99,235,0.2)' }}>
                Bank: {m.text}
              </span>
            ))}
          </div>
        </div>

        {/* Action Buttons */}
        <div style={{ display: 'flex', gap: 8, padding: '6px 16px 10px', borderTop: '1px solid var(--border-light)' }}>
          <button className="btn btn-blue" onClick={handleConfirm} disabled={!latestSign}
            style={{ flex: 1, padding: 10, fontSize: 13 }}>
            <Send size={14} /> Send to Employee
          </button>
          <button className="btn btn-secondary" onClick={handleRetry} disabled={!latestSign}
            style={{ flex: 1, padding: 10, fontSize: 13 }}>
            <RotateCcw size={14} /> Retry
          </button>
        </div>
      </div>

      {/* Animations */}
      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.7} }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
        .spin-icon { animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
