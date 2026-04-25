import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ShieldCheck, User, Lock, ArrowRight } from 'lucide-react';

export default function LoginScreen() {
  const [role, setRole] = useState('employee');
  const navigate = useNavigate();

  const handleLogin = (e) => {
    e.preventDefault();
    navigate(`/${role}`);
  };

  return (
    <div style={{ 
      height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'var(--bg-page)', position: 'relative', overflow: 'hidden' 
    }}>
      {/* Decorative Blobs */}
      <div style={{ position: 'absolute', top: '-10%', left: '-10%', width: '40%', height: '40%', background: 'var(--primary-glow)', filter: 'blur(100px)', borderRadius: '50%' }} />
      <div style={{ position: 'absolute', bottom: '-10%', right: '-10%', width: '40%', height: '40%', background: 'rgba(139, 92, 246, 0.1)', filter: 'blur(100px)', borderRadius: '50%' }} />

      <div className="surface-card animate-enter" style={{ width: '100%', maxWidth: 440, padding: 48, position: 'relative', zIndex: 10 }}>
        <div style={{ textAlign: 'center', marginBottom: 40 }}>
          <div style={{ 
            width: 64, height: 64, background: 'linear-gradient(135deg, var(--primary), var(--accent))', 
            borderRadius: 20, display: 'flex', alignItems: 'center', justifyContent: 'center', 
            margin: '0 auto 24px', boxShadow: '0 8px 25px var(--primary-glow)' 
          }}>
            <ShieldCheck size={32} color="white" />
          </div>
          <h1 className="heading-display" style={{ fontSize: 28, marginBottom: 8 }}>Authentication</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: 14 }}>Secure portal for bank staff and kiosks</p>
        </div>

        <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          <div style={{ display: 'flex', background: 'rgba(255,255,255,0.03)', padding: 4, borderRadius: 12, border: '1px solid var(--border-light)' }}>
            <button type="button" onClick={() => setRole('employee')} style={{ 
              flex: 1, padding: '10px', borderRadius: 8, border: 'none', cursor: 'pointer',
              background: role === 'employee' ? 'rgba(255,255,255,0.08)' : 'transparent',
              color: role === 'employee' ? 'white' : 'var(--text-muted)',
              fontSize: 13, fontWeight: 600, transition: '0.3s'
            }}>Employee</button>
            <button type="button" onClick={() => setRole('kiosk')} style={{ 
              flex: 1, padding: '10px', borderRadius: 8, border: 'none', cursor: 'pointer',
              background: role === 'kiosk' ? 'rgba(255,255,255,0.08)' : 'transparent',
              color: role === 'kiosk' ? 'white' : 'var(--text-muted)',
              fontSize: 13, fontWeight: 600, transition: '0.3s'
            }}>Kiosk</button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div style={{ position: 'relative' }}>
              <User size={18} style={{ position: 'absolute', left: 16, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-faint)' }} />
              <input type="text" placeholder="Access ID" style={{ paddingLeft: 48 }} defaultValue="ADMIN_CORE" />
            </div>
            <div style={{ position: 'relative' }}>
              <Lock size={18} style={{ position: 'absolute', left: 16, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-faint)' }} />
              <input type="password" placeholder="Passcode" style={{ paddingLeft: 48 }} defaultValue="••••••••" />
            </div>
          </div>

          <button type="submit" className="btn btn-primary" style={{ width: '100%', padding: 16, fontSize: 15 }}>
            Authorized Login <ArrowRight size={18} />
          </button>
        </form>
      </div>
    </div>
  );
}
