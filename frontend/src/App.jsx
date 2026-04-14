import { Routes, Route, Navigate } from 'react-router-dom';
import LoginScreen from './pages/LoginScreen';
import EmployeeDashboard from './pages/EmployeeDashboard';
import KioskDashboard from './pages/KioskDashboard';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/login" replace />} />
      <Route path="/login" element={<LoginScreen />} />
      <Route path="/employee" element={<EmployeeDashboard />} />
      <Route path="/kiosk" element={<KioskDashboard />} />
    </Routes>
  );
}

export default App;
