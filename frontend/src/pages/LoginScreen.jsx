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
    <>
      <style>{`
        .login-page {
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          background-color: #111111;
          background-image:
            repeating-linear-gradient(
              -45deg,
              transparent,
              transparent 38px,
              rgba(255,255,255,0.018) 38px,
              rgba(255,255,255,0.018) 40px
            ),
            repeating-linear-gradient(
              45deg,
              transparent,
              transparent 38px,
              rgba(255,255,255,0.018) 38px,
              rgba(255,255,255,0.018) 40px
            );
          padding: 24px;
          font-family: var(--font-sans, 'Inter', system-ui, sans-serif);
        }

        .login-card {
          display: flex;
          width: 100%;
          max-width: 900px;
          min-height: 560px;
          border-radius: 20px;
          overflow: hidden;
          box-shadow: 0 32px 64px rgba(0,0,0,0.6);
        }

        /* LEFT PANEL */
        .login-left {
          flex: 1;
          background: linear-gradient(160deg, #1a1a1a 0%, #0d0d0d 100%);
          display: flex;
          flex-direction: column;
          justify-content: flex-end;
          padding: 48px;
          position: relative;
        }


        .login-left-content {
          position: relative;
          z-index: 1;
        }

        .login-left-badge {
          display: inline-block;
          padding: 5px 14px;
          border: 1px solid rgba(255,255,255,0.18);
          border-radius: 100px;
          font-size: 11px;
          font-weight: 600;
          letter-spacing: 1.2px;
          text-transform: uppercase;
          color: rgba(255,255,255,0.55);
          margin-bottom: 20px;
        }

        .login-left h1 {
          font-size: 30px;
          font-weight: 700;
          color: #ffffff;
          line-height: 1.3;
          margin: 0 0 14px 0;
          letter-spacing: -0.3px;
        }

        .login-left p {
          font-size: 14px;
          color: rgba(255,255,255,0.48);
          line-height: 1.65;
          margin: 0;
          max-width: 280px;
        }

        .login-left-divider {
          width: 36px;
          height: 2px;
          background: rgba(255,255,255,0.25);
          border-radius: 2px;
          margin-top: 28px;
        }

        /* RIGHT PANEL */
        .login-right {
          flex: 1;
          background: #ffffff;
          display: flex;
          flex-direction: column;
          justify-content: center;
          padding: 56px 48px;
        }

        .login-right-header {
          margin-bottom: 36px;
        }

        .login-right-header h2 {
          font-size: 22px;
          font-weight: 700;
          color: #0d0d0d;
          margin: 0 0 6px 0;
          letter-spacing: -0.2px;
        }

        .login-right-header p {
          font-size: 13.5px;
          color: #888888;
          margin: 0;
          line-height: 1.5;
        }

        .login-form {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .login-field {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }

        .login-field label {
          font-size: 12px;
          font-weight: 600;
          color: #444444;
          letter-spacing: 0.3px;
          text-transform: uppercase;
        }

        .login-field input {
          padding: 13px 16px;
          border: 1.5px solid #e5e5e5;
          border-radius: 12px;
          font-size: 14px;
          color: #0d0d0d;
          background: #fafafa;
          outline: none;
          transition: border-color 0.15s, box-shadow 0.15s;
          font-family: inherit;
          width: 100%;
          box-sizing: border-box;
        }

        .login-field input:focus {
          border-color: #333333;
          background: #ffffff;
          box-shadow: 0 0 0 3px rgba(0,0,0,0.06);
        }

        .login-error {
          font-size: 13px;
          color: #c0392b;
          background: #fdf2f2;
          border: 1px solid #f5c6c6;
          border-radius: 10px;
          padding: 10px 14px;
          text-align: center;
        }

        .login-submit {
          margin-top: 4px;
          padding: 14px 20px;
          background: #0d0d0d;
          color: #ffffff;
          border: none;
          border-radius: 12px;
          font-size: 14px;
          font-weight: 600;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          transition: background 0.15s, transform 0.1s;
          font-family: inherit;
          letter-spacing: 0.2px;
        }

        .login-submit:hover:not(:disabled) {
          background: #222222;
        }

        .login-submit:active:not(:disabled) {
          transform: scale(0.99);
        }

        .login-submit:disabled {
          opacity: 0.55;
          cursor: not-allowed;
        }

        @media (max-width: 680px) {
          .login-card {
            flex-direction: column;
            max-width: 420px;
          }
          .login-left {
            padding: 36px 32px 32px;
            min-height: 200px;
            justify-content: flex-end;
          }
          .login-left h1 {
            font-size: 22px;
          }
          .login-right {
            padding: 40px 32px;
          }
        }
      `}</style>

      <div className="login-page">
        <div className="login-card">

          {/* LEFT: Visual Panel */}
          <div className="login-left">
            <div className="login-left-content">
              <div className="login-left-badge">Employee Portal</div>
              <h1>Sign Language Banking Kiosk</h1>
              <p>Accessible banking communication for all users</p>
              <div className="login-left-divider" />
            </div>
          </div>

          {/* RIGHT: Login Form */}
          <div className="login-right">
            <div className="login-right-header">
              <h2>Authentication</h2>
              <p>Sign in to manage client interactions</p>
            </div>

            <form onSubmit={handleLogin} className="login-form">
              <div className="login-field">
                <label>Employee ID</label>
                <input
                  type="text"
                  placeholder="Enter your employee ID"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  required
                />
              </div>

              <div className="login-field">
                <label>Password</label>
                <input
                  type="password"
                  placeholder="Enter your password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                />
              </div>

              {error && <div className="login-error">{error}</div>}

              <button type="submit" className="login-submit" disabled={isLoading}>
                {isLoading ? 'Verifying...' : 'Sign In'}
                {!isLoading && <ArrowRight size={16} />}
              </button>
            </form>
          </div>

        </div>
      </div>
    </>
  );
}
