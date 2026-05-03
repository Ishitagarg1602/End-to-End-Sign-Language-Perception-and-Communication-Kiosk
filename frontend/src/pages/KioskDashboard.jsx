import React, { useEffect, useState, useRef, Suspense, useCallback } from 'react';
import { useSocketEngine } from '../hooks/useSocketEngine';
import { Camera, HandMetal, Send, RotateCcw, Square, MessageSquare, AlertTriangle, Loader2, MessageCircle, Keyboard, CheckCircle2, ScanLine, X } from 'lucide-react';
import AvatarScene, { getGestureForText } from '../components/AvatarScene';

export default function KioskDashboard() {
  const {
    socket, isConnected, sessionId, sessionActive, waitingApproval,
    detectionState, latestSign, confirmedWords, messages,
    employeeMessage, multiPersonAlert,
    stopSigning, confirmSign, retrySign, endSession, dismissEmployeeMessage, sendTextMessage, scanDocument
  } = useSocketEngine('kiosk');

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const chatEndRef = useRef(null);
  const [selectedIntent, setSelectedIntent] = useState(null);
  const [curSentence, setCurSentence] = useState('');
  const [typedText, setTypedText] = useState('');
  const [showTextInput, setShowTextInput] = useState(false);
  const [scannedImage, setScannedImage] = useState(null);
  const [showFlash, setShowFlash] = useState(false);

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
      setShowTextInput(false);
      if (latestSign.intent_options?.length > 0) {
        setSelectedIntent(latestSign.intent_options[0]);
        setCurSentence(latestSign.intent_options[0].sentence);
      } else {
        setSelectedIntent(null);
        setCurSentence(latestSign.sentence);
      }
    }
  }, [latestSign]);

  // Web Audio API Notification Sound
  const playNotificationSound = useCallback(() => {
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) return;
      const ctx = new AudioContext();
      const osc = ctx.createOscillator();
      const gainNode = ctx.createGain();
      osc.connect(gainNode);
      gainNode.connect(ctx.destination);
      
      osc.type = 'sine';
      osc.frequency.setValueAtTime(523.25, ctx.currentTime); // C5
      osc.frequency.exponentialRampToValueAtTime(659.25, ctx.currentTime + 0.1); // E5
      gainNode.gain.setValueAtTime(0.3, ctx.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.3);
    } catch (e) {
      console.warn("Audio context failed:", e);
    }
  }, []);

  // Typewriter effect
  useEffect(() => {
    if (employeeMessage) {
      playNotificationSound();
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
  }, [employeeMessage, playNotificationSound]);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleConfirm = () => {
    if (!latestSign) return;
    confirmSign(latestSign.word, curSentence, latestSign.confidence, selectedIntent?.label);
    setSelectedIntent(null);
    setCurSentence('');
    setShowTextInput(false);
  };

  const handleRetry = () => {
    retrySign();
    setSelectedIntent(null);
    setCurSentence('');
    setShowTextInput(false);
  };

  const handleIntentSelect = (opt) => {
    setSelectedIntent(opt);
    setCurSentence(opt.sentence);
  };

  const handleSendText = () => {
    if (!typedText.trim()) return;
    sendTextMessage(typedText.trim());
    setTypedText('');
    setShowTextInput(false);
  };

  const handleCaptureDocument = () => {
    if (canvasRef.current) {
      const dataUrl = canvasRef.current.toDataURL('image/jpeg', 0.8);
      setShowFlash(true);
      setTimeout(() => setShowFlash(false), 200);
      setScannedImage(dataUrl);
    }
  };

  const handleSendDocument = () => {
    if (scannedImage) {
      scanDocument(scannedImage);
      setScannedImage(null);
    }
  };

  // Status bar config
  const statusConfig = {
    idle: { text: 'Waiting for user…', bg: 'rgba(100,116,139,0.85)', anim: false },
    waiting_approval: { text: 'Connecting to representative…', bg: '#F59E0B', anim: true },
    scanning: { text: 'Recording — sign now, click "Done Signing" when finished', bg: 'var(--accent)', anim: true },
    paused: { text: 'Review detected sign below', bg: '#F59E0B', anim: false },
    processing: { text: 'Processing sign…', bg: '#8B5CF6', anim: true }
  };
  const status = statusConfig[detectionState] || statusConfig.idle;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', padding: '12px', height: '100vh', overflow: 'hidden', position: 'relative' }}>

      {/* WAITING APPROVAL OVERLAY */}
      {waitingApproval && (
        <div style={{ position: 'fixed', inset: 0, zIndex: 3000, background: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(12px)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 24 }}>
          <div style={{ width: 280, height: 280 }}>
            <Suspense fallback={null}><AvatarScene gesture="wave" isActive={true} /></Suspense>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <Loader2 size={24} className="spin-icon" style={{ color: 'var(--accent)' }} />
            <span style={{ fontSize: 20, fontWeight: 600, color: 'var(--text-main)' }}>Connecting to bank representative…</span>
          </div>
          <p style={{ fontSize: 14, color: 'var(--text-muted)', maxWidth: 400, textAlign: 'center', lineHeight: 1.6 }}>
            Please wait while a bank employee accepts your session.
          </p>
        </div>
      )}

      {/* MULTI-PERSON ALERT — Full overlay */}
      {multiPersonAlert && (
        <div style={{ position: 'fixed', inset: 0, zIndex: 2500, background: 'rgba(220,38,38,0.12)', backdropFilter: 'blur(4px)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div className="animate-enter" style={{ background: 'white', borderRadius: 20, padding: '40px 48px', textAlign: 'center', boxShadow: '0 25px 50px rgba(0,0,0,0.15)', border: '2px solid var(--danger)', maxWidth: 420 }}>
            <AlertTriangle size={48} style={{ color: 'var(--danger)', marginBottom: 16 }} />
            <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--danger)', marginBottom: 8 }}>Multiple Users Detected</div>
            <p style={{ fontSize: 14, color: 'var(--text-muted)', lineHeight: 1.6 }}>
              {multiPersonAlert.message || 'Only one person should be at the kiosk at a time. Please ensure you are alone.'}
            </p>
            <div style={{ marginTop: 16, display: 'flex', gap: 12, justifyContent: 'center', fontSize: 12, color: 'var(--text-faint)' }}>
              {multiPersonAlert.faces > 0 && <span>👤 {multiPersonAlert.faces} faces</span>}
              {multiPersonAlert.hands > 0 && <span>✋ {multiPersonAlert.hands} hands</span>}
            </div>
          </div>
        </div>
      )}

      {/* EMPLOYEE MESSAGE + AVATAR MODAL */}
      {employeeMessage && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(4px)' }}>
          <div className="animate-enter" style={{ width: '90vw', maxWidth: 720, background: 'var(--bg-surface)', borderRadius: 24, overflow: 'hidden', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.25)', border: '1px solid var(--border-light)' }}>
            <div style={{ height: 280, position: 'relative' }}>
              <Suspense fallback={<div style={{ height: 280, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#0B1120' }}><Loader2 size={32} className="spin-icon" style={{ color: '#60A5FA' }} /></div>}>
                <AvatarScene gesture={getGestureForText(employeeMessage)} isActive={true} />
              </Suspense>
              <div style={{ position: 'absolute', top: 16, left: '50%', transform: 'translateX(-50%)', display: 'flex', alignItems: 'center', gap: 8, background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(8px)', padding: '8px 20px', borderRadius: 30, color: '#60A5FA', fontSize: 12, fontWeight: 700, letterSpacing: 1, textTransform: 'uppercase' }}>
                <MessageSquare size={14} /> Bank Representative
              </div>
            </div>
            <div style={{ padding: '28px 32px' }}>
              <div style={{ fontSize: 24, fontWeight: 600, lineHeight: 1.5, color: 'var(--text-main)', marginBottom: 8, minHeight: 72 }}>
                "{displayedText}"
                {!typewriterDone && <span style={{ animation: 'blink 0.7s infinite', color: 'var(--accent)' }}>|</span>}
              </div>
              <div style={{ marginTop: 20 }}>
                <button className="btn btn-primary" onClick={dismissEmployeeMessage} style={{ width: '100%', padding: 16, fontSize: 16, borderRadius: 14 }}>
                  <MessageCircle size={18} /> Acknowledge & Continue Signing
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* LEFT: Camera Feed */}
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
            </div>
          )}
          <video ref={videoRef} autoPlay playsInline muted style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)', display: cameraError ? 'none' : 'block' }} />
          <canvas ref={canvasRef} style={{ display: 'none' }} />

          {/* Camera Flash Effect */}
          <div style={{ position: 'absolute', inset: 0, background: 'white', opacity: showFlash ? 0.9 : 0, transition: 'opacity 0.1s ease-out', pointerEvents: 'none', zIndex: 100 }} />

          {/* Status bar */}
          <div style={{ position: 'absolute', bottom: 8, left: 8, right: 8, padding: '8px 16px', borderRadius: 8, fontSize: 12, fontWeight: 600, textAlign: 'center', background: status.bg, color: '#fff', animation: status.anim ? 'pulse 1.5s infinite' : 'none' }}>
            {status.text}
          </div>

          {/* Connection badge */}
          <div style={{ position: 'absolute', top: 10, left: 10, display: 'flex', alignItems: 'center', gap: 8, background: 'rgba(255,255,255,0.9)', padding: '6px 14px', borderRadius: 20, fontSize: 12, fontWeight: 600, boxShadow: 'var(--shadow-md)' }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: isConnected ? 'var(--success)' : 'var(--danger)' }} />
            {isConnected ? 'Connected' : 'Disconnected'}
          </div>

          {sessionActive && (
            <button onClick={endSession} style={{ position: 'absolute', top: 10, right: 10, padding: '5px 14px', borderRadius: 8, border: '1px solid rgba(220,38,38,0.4)', background: 'rgba(220,38,38,0.1)', color: '#f87171', fontFamily: 'var(--font-sans)', fontSize: 11, fontWeight: 600, cursor: 'pointer' }}>
              End Session
            </button>
          )}
        </div>

        {/* Document Scanner Overlay */}
        {scannedImage && (
          <div style={{ position: 'absolute', inset: 0, zIndex: 1000, background: '#000', display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '12px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(0,0,0,0.8)' }}>
              <div style={{ color: 'white', fontWeight: 600, fontSize: 14 }}>Document Preview</div>
              <button onClick={() => setScannedImage(null)} style={{ background: 'transparent', border: 'none', color: 'white', cursor: 'pointer' }}><X size={20} /></button>
            </div>
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16 }}>
              <img src={scannedImage} alt="Scanned Document" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain', borderRadius: 8 }} />
            </div>
            <div style={{ padding: 16, display: 'flex', gap: 12, background: 'rgba(0,0,0,0.8)' }}>
              <button onClick={handleSendDocument} style={{ flex: 1, padding: 12, background: 'var(--accent)', color: 'white', border: 'none', borderRadius: 8, fontWeight: 600, fontSize: 14, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                <Send size={16} /> Send to Employee
              </button>
              <button onClick={() => setScannedImage(null)} style={{ flex: 1, padding: 12, background: 'rgba(255,255,255,0.1)', color: 'white', border: '1px solid rgba(255,255,255,0.2)', borderRadius: 8, fontWeight: 600, fontSize: 14, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                <RotateCcw size={16} /> Retake
              </button>
            </div>
          </div>
        )}

        {/* Buttons */}
        {sessionActive && detectionState === 'scanning' && !scannedImage && (
          <div style={{ display: 'flex', gap: 8, margin: '0 8px 8px' }}>
            <button className="btn" onClick={stopSigning} style={{ flex: 2, padding: 10, fontSize: 14, background: 'var(--danger)', color: 'white', borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
              <Square size={16} fill="white" /> Done Signing
            </button>
            <button className="btn" onClick={handleCaptureDocument} style={{ flex: 1, padding: 10, fontSize: 14, background: 'var(--bg-surface)', color: 'var(--text-main)', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
              <ScanLine size={16} /> Scan Doc
            </button>
          </div>
        )}
      </div>

      {/* RIGHT: Recognition & Chat */}
      <div className="surface-card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border-light)', display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.6 }}>
          <HandMetal size={14} color="var(--accent)" /> Recognition & Chat
        </div>

        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>

          {/* DETECTED STATE — Intent selector */}
          {latestSign && (
            <div className="animate-enter" style={{ padding: '12px 16px', borderBottom: '1px solid var(--border-light)', flexShrink: 0 }}>
              {/* Word + Confidence */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
                <div style={{ fontSize: 24, fontWeight: 700, textTransform: 'uppercase', letterSpacing: 1, background: 'linear-gradient(135deg, var(--accent), #6366f1)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>
                  {latestSign.word}
                </div>
                <span style={{ padding: '3px 10px', borderRadius: 10, fontSize: 12, fontWeight: 700, background: latestSign.confidence >= 0.85 ? 'var(--success-bg)' : 'rgba(245,158,11,0.1)', color: latestSign.confidence >= 0.85 ? 'var(--success)' : '#F59E0B' }}>
                  {Math.round(latestSign.confidence * 100)}%
                </span>
              </div>

              {/* Intent Options */}
              {latestSign.intent_options && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: 0.8, marginBottom: 8 }}>
                    {latestSign.intent_options.length === 0 ? 'Generating options...' : `Select what you want to say (${latestSign.intent_options.length} options)`}
                  </div>
                  <div style={{ maxHeight: 200, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {latestSign.intent_options.length === 0 ? (
                      <div style={{ padding: '16px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 13, background: 'var(--bg-surface)', borderRadius: 8, border: '1px solid var(--border-light)' }}>
                        <Loader2 className="spin-icon" size={20} style={{ margin: '0 auto 8px auto', color: 'var(--accent)' }} />
                        Analyzing sign and generating smart options...
                      </div>
                    ) : (
                      latestSign.intent_options.map((opt, idx) => (
                        <button key={idx} onClick={() => handleIntentSelect(opt)}
                          style={{ display: 'block', width: '100%', padding: '8px 12px', borderRadius: 8, textAlign: 'left',
                            border: selectedIntent?.label === opt.label ? '2px solid var(--accent)' : '1px solid var(--border-light)',
                            background: selectedIntent?.label === opt.label ? 'var(--accent-light)' : 'var(--bg-surface)',
                            color: 'var(--text-main)', fontFamily: 'var(--font-sans)', fontSize: 12, cursor: 'pointer', transition: '0.15s' }}>
                          <span style={{ fontWeight: 700, color: 'var(--accent)', fontSize: 10, textTransform: 'uppercase' }}>{opt.label}</span>
                          <br />{opt.sentence}
                        </button>
                      ))
                    )}
                  </div>
                </div>
              )}

              {/* Type instead toggle */}
              <button onClick={() => setShowTextInput(!showTextInput)} style={{ marginTop: 8, padding: '6px 14px', borderRadius: 8, border: '1px solid var(--border-light)', background: showTextInput ? 'var(--accent-light)' : 'transparent', color: 'var(--text-muted)', fontSize: 11, fontWeight: 600, cursor: 'pointer', fontFamily: 'var(--font-sans)', display: 'flex', alignItems: 'center', gap: 6, width: '100%', justifyContent: 'center' }}>
                <Keyboard size={14} /> {showTextInput ? 'Hide text input' : "None of these? Type your own message"}
              </button>

              {/* Action Buttons */}
              <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
                <button className="btn btn-primary" onClick={handleConfirm} disabled={!selectedIntent && !curSentence} style={{ flex: 1, padding: 10, fontSize: 13 }}>
                  <CheckCircle2 size={14} /> Confirm & Send
                </button>
                <button className="btn btn-secondary" onClick={handleRetry} style={{ flex: 1, padding: 10, fontSize: 13 }}>
                  <RotateCcw size={14} /> Retry
                </button>
              </div>
            </div>
          )}

          {/* Text Input (shown when toggled or no sign detected during active session) */}
          {(showTextInput || (!latestSign && sessionActive)) && (
            <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border-light)', flexShrink: 0 }}>
              {!latestSign && !showTextInput && (
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8, textAlign: 'center' }}>
                  Sign to the camera and click <b>Done Signing</b>, or type below:
                </div>
              )}
              <div style={{ display: 'flex', gap: 8 }}>
                <input type="text" value={typedText} onChange={e => setTypedText(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleSendText()}
                  placeholder="Type your message here..."
                  style={{ flex: 1, padding: '10px 14px', borderRadius: 10, border: '1px solid var(--border-light)', fontSize: 14, background: 'var(--bg-surface)', outline: 'none', fontFamily: 'var(--font-sans)' }} />
                <button className="btn btn-blue" onClick={handleSendText} disabled={!typedText.trim()} style={{ padding: '10px 18px', borderRadius: 10 }}>
                  <Send size={16} />
                </button>
              </div>
            </div>
          )}

          {/* IDLE state message */}
          {!latestSign && !sessionActive && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 10, color: 'var(--text-faint)', padding: 20 }}>
              <HandMetal size={36} />
              <div style={{ fontSize: 13, textAlign: 'center', lineHeight: 1.7 }}>Waiting for session to start…</div>
            </div>
          )}

          {/* Chat History */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '10px 16px', display: 'flex', flexDirection: 'column', gap: 8, minHeight: 0 }}>
            {messages.length === 0 && sessionActive && !latestSign && (
              <div style={{ textAlign: 'center', padding: 20, color: 'var(--text-faint)', fontSize: 12 }}>
                Sign to the camera, then click "Done Signing"
              </div>
            )}
            {messages.map((msg) => (
              <div key={msg.id} className="animate-enter" style={{
                padding: '10px 14px', borderRadius: 12, maxWidth: '85%',
                alignSelf: msg.type === 'tx' ? 'flex-end' : msg.type === 'rx' ? 'flex-start' : 'center',
                background: msg.type === 'tx' ? 'var(--accent)' : msg.type === 'rx' ? 'var(--bg-subtle)' : 'transparent',
                color: msg.type === 'tx' ? 'white' : 'var(--text-main)',
                border: msg.type === 'sys' ? 'none' : `1px solid ${msg.type === 'tx' ? 'var(--accent)' : 'var(--border-light)'}`
              }}>
                <div style={{ fontSize: 10, fontWeight: 600, marginBottom: 3, opacity: 0.7, display: 'flex', alignItems: 'center', gap: 4 }}>
                  {msg.inputMode === 'sign' && '🤟'}
                  {msg.inputMode === 'text' && '⌨️'}
                  {msg.inputMode === 'voice' && '🔊'}
                  {msg.label} · {msg.time}
                </div>
                <div style={{ fontSize: 14, lineHeight: 1.4 }}>{msg.text}</div>
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>
        </div>
      </div>

      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.7} }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
        .spin-icon { animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
