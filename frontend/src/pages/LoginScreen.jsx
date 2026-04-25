import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Fingerprint, ArrowRight } from 'lucide-react';

export default function LoginScreen() {
  const navigate = useNavigate();
  const [username, setUsername] = useState('admin');
  const [password, setPassword] = useState('password123');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = (e) => {
    e.preventDefault();
    setError('');
    
    if (username === 'admin' && password === 'password123') {
      setIsLoading(true);
      setTimeout(() => {
        navigate('/employee');
      }, 800); 
    } else {
      setError('Invalid identity credentials.');
    }
  };

  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
      <div className="surface-card animate-enter" style={{ width: '420px', padding: '48px', display: 'flex', flexDirection: 'column', gap: '32px' }}>
        
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
          <div style={{ padding: '12px', background: 'var(--bg-page)', border: '1px solid var(--border-light)', borderRadius: '16px', color: 'var(--primary)' }}>
            <Fingerprint size={32} />
          </div>
          <div style={{ textAlign: 'center' }}>
             <h2 className="heading-display" style={{ fontSize: '24px' }}>Teller Authentication</h2>
             <p style={{ color: 'var(--text-muted)', fontSize: '14px', marginTop: '6px' }}>Sign in to manage client interactions.</p>
          </div>
        </div>

        <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
             <input type="text" placeholder="Employee ID" value={username} onChange={e => setUsername(e.target.value)} required />
             <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} required />
          </div>

          {error && <div style={{ color: 'var(--danger)', fontSize: '13px', textAlign: 'center', padding: '8px', background: 'var(--danger-bg)', borderRadius: 'var(--radius-sm)' }}>{error}</div>}

          <button type="submit" className="btn btn-primary" disabled={isLoading} style={{ width: '100%', padding: '14px' }}>
            {isLoading ? 'Verifying...' : 'Sign In'}
            {!isLoading && <ArrowRight size={16} />}
          </button>
        </form>

      </div>
    </div>
  );
}
