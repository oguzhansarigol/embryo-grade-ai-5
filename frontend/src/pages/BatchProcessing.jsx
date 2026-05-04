import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, Download } from 'lucide-react';

export default function BatchProcessing() {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
    }
  };

  const handleUpload = async () => {
    if (files.length === 0) return;
    setLoading(true);
    const formData = new FormData();
    files.forEach(f => formData.append('images', f));

    try {
      const res = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.results) {
        setResults(data.results);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="flex-between" style={{marginBottom: '32px'}}>
        <div>
          <h1 className="page-title">Toplu İşlem</h1>
          <p className="page-subtitle" style={{marginBottom: 0}}>Birden fazla görseli aynı anda analiz edin.</p>
        </div>
        <div className="flex-gap">
          <button className="btn-secondary" style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
            <FileText size={16} /> PDF Raporu Oluştur
          </button>
          <button className="btn-secondary" style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
            <Download size={16} /> CSV Dışa Aktar
          </button>
        </div>
      </div>

      <div className="card" style={{marginBottom: '24px'}}>
        <div className="flex-gap">
          <div 
            className="upload-area"
            style={{flex: 1, padding: '24px'}}
            onClick={() => fileInputRef.current.click()}
          >
            <UploadCloud size={32} className="upload-icon" style={{marginBottom: '8px'}} />
            <h4>Klasör veya Birden Fazla Dosya Seçin</h4>
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{display: 'none'}} 
              onChange={handleFileChange}
              accept=".jpg,.jpeg,.png,.tif,.tiff,.bmp"
              multiple
              webkitdirectory="true"
            />
          </div>
          <div style={{width: '200px', display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: '16px'}}>
            <div style={{fontSize: '14px'}}>
              <strong>Seçilen Dosya:</strong> {files.length}
            </div>
            <button 
              className="btn-primary" 
              onClick={handleUpload} 
              disabled={files.length === 0 || loading}
            >
              {loading ? 'İşleniyor...' : 'Toplu Analizi Başlat'}
            </button>
          </div>
        </div>
      </div>

      <div className="card">
        <h3 style={{marginBottom: '16px'}}>Sonuçlar</h3>
        {results.length > 0 ? (
          <table className="data-table">
            <thead>
              <tr>
                <th>Dosya Adı</th>
                <th>Tahmin</th>
                <th>Güven Skoru</th>
                <th>Uyarı</th>
                <th>Önizleme</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r, idx) => (
                <tr key={idx}>
                  <td style={{fontWeight: 500}}>{r.filename}</td>
                  <td>
                    <span className="badge outline-blue">{r.predicted_class}</span>
                  </td>
                  <td>{(r.confidence * 100).toFixed(1)}%</td>
                  <td>
                    {r.warning ? (
                      <span className="badge amber">Manuel Kontrol</span>
                    ) : (
                      <span className="badge green">Güvenilir</span>
                    )}
                  </td>
                  <td>
                    <img src={r.gradcam_url} alt="preview" style={{width: '40px', height: '40px', borderRadius: '4px', objectFit: 'cover'}} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p style={{textAlign: 'center', padding: '32px 0'}}>Henüz analiz yapılmadı.</p>
        )}
      </div>
    </div>
  );
}
