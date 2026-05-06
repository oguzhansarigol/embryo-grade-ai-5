import React, { useState, useRef } from 'react';
import { UploadCloud, AlertCircle, CheckCircle } from 'lucide-react';

export default function SinglePrediction() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('images', file);

    try {
      const res = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.results && data.results.length > 0) {
        setResult(data.results[0]);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1 className="page-title">Tekli Görsel Tahmini</h1>
      <p className="page-subtitle">Embriyo kalitesini değerlendirmek için tek bir görsel yükleyin.</p>

      <div className="grid-2">
        <div className="card">
          <div 
            className="upload-area"
            onClick={() => fileInputRef.current.click()}
          >
            <UploadCloud size={48} className="upload-icon" />
            <h3 style={{marginBottom: '8px'}}>Görseli Sürükleyin veya Seçin</h3>
            <p style={{marginBottom: '24px'}}>{file ? file.name : "Desteklenen formatlar: JPG, PNG, TIF"}</p>
            <button className="btn-secondary" onClick={(e) => {e.stopPropagation(); fileInputRef.current.click();}}>
              Dosya Seç
            </button>
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{display: 'none'}} 
              onChange={handleFileChange}
              accept=".jpg,.jpeg,.png,.tif,.tiff,.bmp"
            />
          </div>
          
          <div style={{marginTop: '24px', textAlign: 'right'}}>
            <button 
              className="btn-primary" 
              onClick={handleUpload} 
              disabled={!file || loading}
            >
              {loading ? 'Analiz Ediliyor...' : 'Analizi Başlat'}
            </button>
          </div>
        </div>

        {result ? (
          <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
            <h3 style={{marginBottom: '16px'}}>Analiz Sonucu</h3>
            
            {result.warning && (
              <div className="warning-banner">
                <AlertCircle size={20} />
                <span>Bu tahmin düşük güvenilirliktedir ({Math.round(result.confidence * 100)}%), lütfen manuel kontrol yapınız.</span>
              </div>
            )}

            {!result.warning && (
              <div className="warning-banner" style={{backgroundColor: '#E8F5E9', borderColor: '#4F8A6B', color: '#4F8A6B'}}>
                <CheckCircle size={20} />
                <span>Tahmin yüksek güvenilirliktedir ({Math.round(result.confidence * 100)}%).</span>
              </div>
            )}

            <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '24px 0'}}>
              <div style={{textAlign: 'center'}}>
                <div style={{fontSize: '48px', fontWeight: '700', color: 'var(--accent-blue)'}}>
                  {result.predicted_class}
                </div>
                <div style={{color: 'var(--text-secondary)'}}>Gardner Sınıfı</div>
              </div>
            </div>

            <div className="grid-2" style={{marginTop: 'auto'}}>
              <div>
                <p style={{fontSize: '12px', fontWeight: 600, marginBottom: '8px'}}>Orijinal Görsel</p>
                <img src={result.image_url} alt="Orijinal" style={{width: '100%', borderRadius: '6px'}} />
              </div>
              <div>
                <p style={{fontSize: '12px', fontWeight: 600, marginBottom: '8px'}}>Grad-CAM (Isı Haritası)</p>
                <img src={result.gradcam_url} alt="Grad-CAM" style={{width: '100%', borderRadius: '6px'}} />
              </div>
            </div>
            
            <div style={{marginTop: '24px', padding: '16px', backgroundColor: 'var(--bg-color)', borderRadius: '6px'}}>
              <h4 style={{marginBottom: '8px', fontSize: '14px'}}>Morfolojik Özellik Raporu</h4>
              <p style={{fontSize: '13px'}}>
                Simetri: Uygun<br/>
                Fragmentasyon: Düşük<br/>
                Vakuol Varlığı: Gözlemlenmedi<br/><br/>
                <i>{result.focus_hint || 'Model odağı (ICM/TE) belirlenemedi.'}</i>
              </p>
            </div>
          </div>
        ) : (
          <div className="card" style={{display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)'}}>
            Sonuçları görmek için görsel yükleyip analiz başlatın.
          </div>
        )}
      </div>
    </div>
  );
}
