import React, { useEffect, useState, useRef, Suspense, useCallback } from 'react';
import { useSocketEngine } from '../hooks/useSocketEngine';
import { Camera, HandMetal, Send, RotateCcw, Square, MessageSquare, AlertTriangle, Loader2, MessageCircle, Keyboard, CheckCircle2, ScanLine, X, Plus, Trash2, Images } from 'lucide-react';
import AvatarScene, { getGestureForText } from '../components/AvatarScene';

export default function KioskDashboard() {
  const {
    socket, isConnected, sessionId, sessionActive, waitingApproval,
    detectionState, latestSign, confirmedWords, messages,
    employeeMessage, multiPersonAlert,
    stopSigning, confirmSign, retrySign, endSession, dismissEmployeeMessage, sendTextMessage, scanDocument, resumeAfterMultiPerson
  } = useSocketEngine('kiosk');

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const chatEndRef = useRef(null);
  const [selectedIntent, setSelectedIntent] = useState(null);
  const [curSentence, setCurSentence] = useState('');
  const [typedText, setTypedText] = useState('');
  const [showTextInput, setShowTextInput] = useState(false);
  const [scannedImages, setScannedImages] = useState([]);
  const [showFlash, setShowFlash] = useState(false);
  const [scannerMode, setScannerMode] = useState(null); // null | 'capturing' | 'gallery'
  const MAX_SCAN_PAGES = 5;

  // Onboarding state
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [onboardingCompleted, setOnboardingCompleted] = useState(false);
  const [onboardingName, setOnboardingName] = useState('');
  const [onboardingDOB, setOnboardingDOB] = useState('');
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

  // Trigger onboarding when session becomes active
  useEffect(() => {
    if (sessionActive && !onboardingCompleted) {
      setShowOnboarding(true);
    }
    if (!sessionActive) {
      setShowOnboarding(false);
      setOnboardingCompleted(false);
      setOnboardingName('');
      setOnboardingDOB('');
    }
  }, [sessionActive, onboardingCompleted]);

  const handleOnboardingSubmit = (e) => {
    e.preventDefault();
    if (!onboardingName.trim() || !onboardingDOB) return;
    
    setShowOnboarding(false);
    setOnboardingCompleted(true);
    
    sendTextMessage(`User details received: ${onboardingName.trim()}, DOB: ${onboardingDOB}`);
  };
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
    if (videoRef.current && canvasRef.current && scannedImages.length < MAX_SCAN_PAGES) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
      setShowFlash(true);
      setTimeout(() => setShowFlash(false), 200);
      setScannedImages(prev => [...prev, dataUrl]);
      setScannerMode('gallery');
    }
  };

  const handleStartCapture = () => {
    setScannerMode('capturing');
  };

  const handleRemoveScannedImage = (index) => {
    setScannedImages(prev => {
      const next = prev.filter((_, i) => i !== index);
      if (next.length === 0) setScannerMode(null);
      return next;
    });
  };

  const handleSendDocument = () => {
    if (scannedImages.length > 0) {
      scanDocument(scannedImages);
      setScannedImages([]);
      setScannerMode(null);
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

    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden', position: 'relative' }}>

      {/* TOP HEADER BAR */}
      <header style={{ flexShrink: 0, background: '#1a1a1a', borderBottom: '1px solid #2a2a2a', padding: '0 24px', height: 48, display: 'flex', alignItems: 'center', justifyContent: 'space-between', zIndex: 100 }}>
        <span style={{ color: '#ffffff', fontWeight: 700, fontSize: 15, letterSpacing: 0.3, fontFamily: 'var(--font-sans)' }}>ISL Banking Interface</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '5px 16px', borderRadius: 30, background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.12)' }}>
          <div style={{ width: 7, height: 7, borderRadius: '50%', background: isConnected ? '#4ade80' : '#f87171', boxShadow: isConnected ? '0 0 6px #4ade80' : 'none' }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: '#ffffff', letterSpacing: 1, textTransform: 'uppercase' }}>{isConnected ? 'Online' : 'Offline'}</span>
        </div>
      </header>

      {/* BODY: sidebar + panels */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* LEFT SIDEBAR */}
        <nav style={{ width: 60, flexShrink: 0, background: '#0d0d0d', borderRight: '1px solid #1f1f1f', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '16px 0', gap: 24 }}>
          <div style={{ width: 36, height: 36, background: '#2a2a2a', borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'rgba(255,255,255,0.8)' }}>
            <Camera size={18} />
          </div>
          <div style={{ width: 36, height: 36, display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: 10, background: 'rgba(255,255,255,0.08)', color: '#D6C2A8' }}>
            <HandMetal size={18} />
          </div>
          <div style={{ flex: 1 }} />
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: isConnected ? '#4ade80' : '#f87171' }} />
            <span style={{ fontSize: 8, fontWeight: 700, color: 'rgba(255,255,255,0.4)', letterSpacing: 0.5 }}>{isConnected ? 'LIVE' : 'OFF'}</span>
          </div>
        </nav>

        {/* MAIN PANELS GRID */}
        <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', padding: '12px', overflow: 'hidden', position: 'relative' }}>

      {/* WAITING APPROVAL OVERLAY */}
      {waitingApproval && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 3000,
          background: '#111111',
          backgroundImage: `repeating-linear-gradient(-45deg, transparent, transparent 38px, rgba(255,255,255,0.018) 38px, rgba(255,255,255,0.018) 40px), repeating-linear-gradient(45deg, transparent, transparent 38px, rgba(255,255,255,0.018) 38px, rgba(255,255,255,0.018) 40px)`,
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        }}>
          {/* Dark top bar inside overlay */}
          <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 48, background: '#1a1a1a', borderBottom: '1px solid #2a2a2a', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 24px' }}>
            <span style={{ color: '#ffffff', fontWeight: 700, fontSize: 15, letterSpacing: 0.3, fontFamily: 'var(--font-sans)' }}>ISL Banking Interface</span>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '5px 16px', borderRadius: 30, background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.12)' }}>
              <div style={{ width: 7, height: 7, borderRadius: '50%', background: isConnected ? '#4ade80' : '#f87171', boxShadow: isConnected ? '0 0 6px #4ade80' : 'none' }} />
              <span style={{ fontSize: 11, fontWeight: 700, color: '#ffffff', letterSpacing: 1, textTransform: 'uppercase' }}>{isConnected ? 'Online' : 'Offline'}</span>
            </div>
          </div>

          {/* White card */}
          <div className="animate-enter" style={{
            background: '#ffffff',
            borderRadius: 20,
            boxShadow: '0 32px 64px rgba(0,0,0,0.5)',
            width: '90%', maxWidth: 420,
            overflow: 'hidden',
            display: 'flex', flexDirection: 'column', alignItems: 'center',
          }}>
            {/* Image section */}
            <div style={{ width: '100%', background: '#f5f5f5', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '36px 0 28px' }}>
              <img
                src="/src/assets/hero.png"
                alt="Waiting illustration"
                style={{ width: 160, height: 160, objectFit: 'contain', borderRadius: 12 }}
              />
            </div>

            {/* Text section */}
            <div style={{ padding: '28px 36px 36px', textAlign: 'center', width: '100%', boxSizing: 'border-box' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10, marginBottom: 10 }}>
                <Loader2 size={18} className="spin-icon" style={{ color: '#111111', flexShrink: 0 }} />
                <span style={{ fontSize: 18, fontWeight: 700, color: '#0d0d0d', letterSpacing: -0.2 }}>Connecting to bank representative</span>
              </div>
              <p style={{ fontSize: 13.5, color: '#888888', lineHeight: 1.65, margin: 0 }}>
                Please wait while a bank employee accepts your session.
              </p>
              {/* Animated loading dots */}
              <div style={{ display: 'flex', justifyContent: 'center', gap: 6, marginTop: 24 }}>
                {[0, 1, 2].map(i => (
                  <div key={i} style={{ width: 8, height: 8, borderRadius: '50%', background: '#D6C2A8', animation: `waitDot 1.2s ease-in-out ${i * 0.2}s infinite` }} />
                ))}
              </div>
            </div>
          </div>

          <style>{`
            @keyframes waitDot {
              0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
              40% { transform: scale(1); opacity: 1; }
            }
          `}</style>
        </div>
      )}


      {/* ONBOARDING OVERLAY */}
      {showOnboarding && (
        <div style={{ position: 'fixed', inset: 0, zIndex: 4000, background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(8px)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <div className="animate-enter" style={{ background: 'var(--bg-surface, white)', padding: '40px', borderRadius: '24px', boxShadow: '0 25px 50px -12px rgba(0,0,0,0.25)', border: '1px solid var(--border-light, #e5e7eb)', width: '90%', maxWidth: '440px' }}>
            <h2 style={{ margin: '0 0 8px 0', fontSize: '24px', fontWeight: 700, color: 'var(--text-main, #111827)', textAlign: 'center' }}>
              Verify Your Identity
            </h2>
            <p style={{ margin: '0 0 24px 0', fontSize: '14px', color: 'var(--text-muted, #6b7280)', textAlign: 'center', lineHeight: 1.5 }}>
              Please enter your basic details to continue
            </p>
            <form onSubmit={handleOnboardingSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <label style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-main, #374151)' }}>Full Name</label>
                <input 
                  type="text" 
                  required 
                  value={onboardingName} 
                  onChange={e => setOnboardingName(e.target.value)}
                  placeholder="e.g. John Doe"
                  style={{ padding: '12px 16px', borderRadius: '12px', border: '1px solid var(--border-light, #d1d5db)', fontSize: '15px', outline: 'none', background: 'var(--bg-main, #ffffff)', color: 'var(--text-main, #111827)' }} 
                />
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                <label style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-main, #374151)' }}>Date of Birth</label>
                <input 
                  type="date" 
                  required 
                  value={onboardingDOB} 
                  onChange={e => setOnboardingDOB(e.target.value)}
                  style={{ padding: '12px 16px', borderRadius: '12px', border: '1px solid var(--border-light, #d1d5db)', fontSize: '15px', outline: 'none', background: 'var(--bg-main, #ffffff)', color: 'var(--text-main, #111827)' }} 
                />
              </div>
              <button 
                type="submit" 
                disabled={!onboardingName.trim() || !onboardingDOB}
                className="btn btn-primary" 
                style={{ marginTop: '8px', padding: '14px', fontSize: '16px', borderRadius: '12px', width: '100%', fontWeight: 600, cursor: (!onboardingName.trim() || !onboardingDOB) ? 'not-allowed' : 'pointer', opacity: (!onboardingName.trim() || !onboardingDOB) ? 0.6 : 1 }}
              >
                Continue
              </button>
            </form>
          </div>
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
            <button
              onClick={resumeAfterMultiPerson}
              style={{ marginTop: 20, padding: '10px 28px', borderRadius: 10, border: 'none', background: 'var(--accent)', color: 'white', fontSize: 14, fontWeight: 600, cursor: 'pointer', fontFamily: 'var(--font-sans)' }}
            >
              Continue Signing
            </button>
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
            <Camera size={14} color="#111111" /> Camera Feed
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

        {/* Document Scanner — Camera Capture Mode */}
        {scannerMode === 'capturing' && (
          <div style={{ position: 'absolute', inset: 0, zIndex: 1000, display: 'flex', flexDirection: 'column' }}>
            {/* Top bar */}
            <div style={{ position: 'absolute', top: 0, left: 0, right: 0, zIndex: 10, padding: '12px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'linear-gradient(to bottom, rgba(0,0,0,0.7), transparent)' }}>
              <div style={{ color: 'white', fontWeight: 600, fontSize: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
                <ScanLine size={16} /> Position document in view
              </div>
              <button onClick={() => setScannerMode(scannedImages.length > 0 ? 'gallery' : null)} style={{ background: 'rgba(255,255,255,0.15)', border: 'none', color: 'white', borderRadius: 8, padding: '6px 14px', cursor: 'pointer', fontSize: 12, fontWeight: 600 }}>Cancel</button>
            </div>
            {/* Capture button at bottom */}
            <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, zIndex: 10, padding: 20, display: 'flex', justifyContent: 'center', background: 'linear-gradient(to top, rgba(0,0,0,0.7), transparent)' }}>
              <button onClick={handleCaptureDocument} style={{ width: 72, height: 72, borderRadius: '50%', border: '4px solid white', background: 'rgba(255,255,255,0.2)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 4px 20px rgba(0,0,0,0.4)', transition: 'transform 0.1s' }} onMouseDown={e => e.currentTarget.style.transform = 'scale(0.9)'} onMouseUp={e => e.currentTarget.style.transform = 'scale(1)'}>
                <div style={{ width: 56, height: 56, borderRadius: '50%', background: 'white' }} />
              </button>
            </div>
            {/* Page count badge */}
            {scannedImages.length > 0 && (
              <div style={{ position: 'absolute', bottom: 24, right: 24, zIndex: 10, background: 'var(--accent)', color: 'white', borderRadius: 20, padding: '6px 14px', fontSize: 12, fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6 }} onClick={() => setScannerMode('gallery')}>
                <Images size={14} /> {scannedImages.length} page{scannedImages.length > 1 ? 's' : ''}
              </div>
            )}
          </div>
        )}

        {/* Document Scanner — Gallery Review Mode */}
        {scannerMode === 'gallery' && scannedImages.length > 0 && (
          <div style={{ position: 'absolute', inset: 0, zIndex: 1000, background: '#0a0a0a', display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '12px 16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(0,0,0,0.8)' }}>
              <div style={{ color: 'white', fontWeight: 600, fontSize: 14 }}>📄 {scannedImages.length} / {MAX_SCAN_PAGES} Pages Scanned</div>
              <button onClick={() => { setScannedImages([]); setScannerMode(null); }} style={{ background: 'transparent', border: 'none', color: 'white', cursor: 'pointer' }}><X size={20} /></button>
            </div>
            <div style={{ flex: 1, display: 'flex', gap: 12, alignItems: 'center', justifyContent: 'center', padding: 16, overflowX: 'auto' }}>
              {scannedImages.map((img, idx) => (
                <div key={idx} style={{ position: 'relative', flexShrink: 0, width: 140, height: 180, borderRadius: 8, overflow: 'hidden', border: '2px solid rgba(255,255,255,0.3)' }}>
                  <img src={img} alt={`Page ${idx + 1}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, background: 'rgba(0,0,0,0.7)', color: 'white', fontSize: 10, fontWeight: 700, textAlign: 'center', padding: 3 }}>Page {idx + 1}</div>
                  <button onClick={() => handleRemoveScannedImage(idx)} style={{ position: 'absolute', top: 4, right: 4, background: 'rgba(220,38,38,0.85)', border: 'none', color: 'white', borderRadius: '50%', width: 24, height: 24, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}><Trash2 size={12} /></button>
                </div>
              ))}
              {scannedImages.length < MAX_SCAN_PAGES && (
                <button onClick={handleStartCapture} style={{ flexShrink: 0, width: 140, height: 180, borderRadius: 8, border: '2px dashed rgba(255,255,255,0.3)', background: 'transparent', color: 'rgba(255,255,255,0.6)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 8, cursor: 'pointer', fontSize: 12, fontWeight: 600 }}>
                  <Plus size={28} />
                  Add Page
                </button>
              )}
            </div>
            <div style={{ padding: 16, display: 'flex', gap: 12, background: 'rgba(0,0,0,0.8)' }}>
              <button onClick={handleSendDocument} style={{ flex: 1, padding: 12, background: 'var(--accent)', color: 'white', border: 'none', borderRadius: 8, fontWeight: 600, fontSize: 14, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                <Send size={16} /> Send {scannedImages.length} Page{scannedImages.length > 1 ? 's' : ''}
              </button>
              <button onClick={() => { setScannedImages([]); setScannerMode(null); }} style={{ flex: 1, padding: 12, background: 'rgba(255,255,255,0.1)', color: 'white', border: '1px solid rgba(255,255,255,0.2)', borderRadius: 8, fontWeight: 600, fontSize: 14, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                <Trash2 size={16} /> Clear All
              </button>
            </div>
          </div>
        )}


          {/* Buttons */}
          {sessionActive && detectionState === 'scanning' && !scannerMode && (
            <div style={{ display: 'flex', gap: 8, margin: '0 8px 8px' }}>
              <button className="btn" onClick={stopSigning} style={{ flex: 2, padding: 10, fontSize: 14, background: 'var(--danger)', color: 'white', borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                <Square size={16} fill="white" /> Done Signing
              </button>
              <button className="btn" onClick={handleStartCapture} style={{ flex: 1, padding: 10, fontSize: 14, background: 'var(--bg-surface)', color: 'var(--text-main)', border: '1px solid var(--border-light)', borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                <ScanLine size={16} /> Scan Doc
              </button>
            </div>
          )}
        </div>

        {/* RIGHT: Recognition & Chat */}
        <div className="surface-card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border-light)', display: 'flex', alignItems: 'center', gap: 8, fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 0.6 }}>
            <HandMetal size={14} color="#111111" /> Recognition & Chat
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
                  style={{ flex: 1, padding: '10px 14px', borderRadius: 10, border: '1px solid #d0d0d0', fontSize: 14, background: '#efefef', outline: 'none', fontFamily: 'var(--font-sans)', color: '#111111' }} />
                <button className="btn" onClick={handleSendText} disabled={!typedText.trim()} style={{ padding: '10px 18px', borderRadius: 10, background: '#111111', color: '#ffffff', border: 'none' }}>
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
                background: msg.type === 'tx' ? '#111111' : msg.type === 'rx' ? '#D6C2A8' : 'transparent',
                color: msg.type === 'tx' ? '#ffffff' : msg.type === 'rx' ? '#1a1a1a' : 'var(--text-main)',
                border: msg.type === 'sys' ? 'none' : `1px solid ${msg.type === 'tx' ? '#111111' : msg.type === 'rx' ? '#c4ae93' : 'var(--border-light)'}`
              }}>
                <div style={{ fontSize: 10, fontWeight: 600, marginBottom: 3, opacity: 0.65, display: 'flex', alignItems: 'center', gap: 4 }}>
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

        </div> {/* end MAIN PANELS GRID */}
      </div> {/* end BODY */}

      <style>{`
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.7} }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
        .spin-icon { animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
