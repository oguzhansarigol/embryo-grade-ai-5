import React, { useEffect, useState } from 'react';
import { Filter, ChevronRight } from 'lucide-react';

export default function History() {
  const [rows, setRows] = useState([]);
  
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await fetch('http://localhost:5000/api/history');
        const data = await res.json();
        setRows(data);
      } catch (error) {
        console.error(error);
      }
    };
    fetchHistory();
  }, []);

  return (
    <div>
      <h1 className="page-title">Geçmiş ve Veritabanı</h1>
      <p className="page-subtitle">Önceki analizleri inceleyin ve klinik sonuçları (follow-up) kaydedin.</p>

      <div className="card" style={{marginBottom: '24px'}}>
        <div className="flex-gap" style={{alignItems: 'flex-end'}}>
          <div style={{flex: 1}}>
            <label style={{display: 'block', fontSize: '12px', marginBottom: '4px', fontWeight: 600}}>Tarih Aralığı</label>
            <input type="date" className="input-field" />
          </div>
          <div style={{flex: 1}}>
            <label style={{display: 'block', fontSize: '12px', marginBottom: '4px', fontWeight: 600}}>Sınıf</label>
            <select className="input-field">
              <option>Tümü</option>
              <option>5AA</option>
              <option>4AA</option>
              <option>3CC</option>
            </select>
          </div>
          <div style={{flex: 1}}>
            <label style={{display: 'block', fontSize: '12px', marginBottom: '4px', fontWeight: 600}}>Güven Eşiği</label>
            <select className="input-field">
              <option>Tümü</option>
              <option>&lt; %70 (Uyarılı)</option>
              <option>&ge; %70 (Güvenilir)</option>
            </select>
          </div>
          <div>
            <button className="btn-secondary" style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
              <Filter size={16} /> Filtrele
            </button>
          </div>
        </div>
      </div>

      <div className="card">
        <table className="data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Tarih</th>
              <th>Dosya</th>
              <th>Tahmin</th>
              <th>Güven Skoru</th>
              <th>Uyarı Durumu</th>
              <th>Gerçek Sonuç</th>
              <th>Detay</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, idx) => (
              <tr key={idx}>
                <td>#{r.id}</td>
                <td>{r.timestamp.split('T')[0]}</td>
                <td>{r.image_filename}</td>
                <td>
                  <span className="badge outline-blue">{r.predicted_class}</span>
                </td>
                <td>{(r.confidence * 100).toFixed(1)}%</td>
                <td>
                  {r.warning_flag ? (
                    <span className="badge amber">Uyarı</span>
                  ) : (
                    <span className="badge green">Uygun</span>
                  )}
                </td>
                <td>{r.actual_class || '-'}</td>
                <td>
                  <button className="btn-secondary" style={{padding: '4px 8px'}}>
                    <ChevronRight size={16} />
                  </button>
                </td>
              </tr>
            ))}
            {rows.length === 0 && (
              <tr>
                <td colSpan="8" style={{textAlign: 'center', padding: '32px'}}>Kayıt bulunamadı.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
