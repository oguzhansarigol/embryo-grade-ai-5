import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import SinglePrediction from './pages/SinglePrediction';
import BatchProcessing from './pages/BatchProcessing';
import History from './pages/History';
import Reporting from './pages/Reporting';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/single" replace />} />
          <Route path="/single" element={<SinglePrediction />} />
          <Route path="/batch" element={<BatchProcessing />} />
          <Route path="/history" element={<History />} />
          <Route path="/reporting" element={<Reporting />} />
        </Routes>
      </Layout>
    </Router>
  );
}

export default App;
