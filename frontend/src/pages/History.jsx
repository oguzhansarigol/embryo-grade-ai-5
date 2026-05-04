import React, { useEffect, useState } from 'react';
import { Filter, ChevronRight } from 'lucide-react';

export default function History() {
  const [rows, setRows] = useState([]);
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [classFilter, setClassFilter] = useState('Tümü');
  const [confidenceFilter, setConfidenceFilter] = useState('Tümü');
  const [filteredRows, setFilteredRows] = useState([]);
  
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await fetch('http://localhost:5000/api/history');
        const data = await res.json();
        setRows(Array.isArray(data) ? data : []);
        setFilteredRows(Array.isArray(data) ? data : []);
      } catch (error) {
        console.error(error);
      }
    };
    fetchHistory();
  }, []);

  const applyFilters = () => {
    const from = dateFrom ? new Date(`${dateFrom}T00:00:00`) : null;
    const to = dateTo ? new Date(`${dateTo}T23:59:59`) : null;

    const next = rows.filter((r) => {
      const ts = r.timestamp ? new Date(r.timestamp) : null;
      if (from && ts && ts < from) return false;
      if (to && ts && ts > to) return false;

      if (classFilter !== 'Tümü' && r.predicted_class !== classFilter) return false;

      if (confidenceFilter === '<70') {
        if (!(typeof r.confidence === 'number' && r.confidence < 0.7)) return false;
      }
      if (confidenceFilter === '>=70') {
        if (!(typeof r.confidence === 'number' && r.confidence >= 0.7)) return false;
      }

      return true;
    });

    setFilteredRows(next);
  };

  return (
    <div>
      <h1 className="page-title">Geçmiş ve Veritabanı</h1>
      <p className="page-subtitle">Önceki analizleri inceleyin ve klinik sonuçları (follow-up) kaydedin.</p>

      <div className="card" style={{marginBottom: '24px'}}>
        <div className="flex-gap" style={{alignItems: 'flex-end'}}>
          <div style={{flex: 1}}>
            <label style={{display: 'block', fontSize: '12px', marginBottom: '4px', fontWeight: 600}}>Tarih Aralığı</label>
            <div className="flex-gap">
              <input type="date" className="input-field" value={dateFrom} onChange={(e) => setDateFrom(e.target.value)} />
              <input type="date" className="input-field" value={dateTo} onChange={(e) => setDateTo(e.target.value)} />
            </div>
          </div>
          <div style={{flex: 1}}>
            <label style={{display: 'block', fontSize: '12px', marginBottom: '4px', fontWeight: 600}}>Sınıf</label>
            <select className="input-field" value={classFilter} onChange={(e) => setClassFilter(e.target.value)}>
              <option value="Tümü">Tümü</option>
              <option value="3AA">3AA</option>
              <option value="3CC">3CC</option>
              <option value="4AA">4AA</option>
              <option value="Cleavage">Cleavage</option>
            </select>
          </div>
          <div style={{flex: 1}}>
            <label style={{display: 'block', fontSize: '12px', marginBottom: '4px', fontWeight: 600}}>Güven Eşiği</label>
            <select className="input-field" value={confidenceFilter} onChange={(e) => setConfidenceFilter(e.target.value)}>
              <option value="Tümü">Tümü</option>
              <option value="<70">&lt; %70 (Uyarılı)</option>
              <option value=">=70">&ge; %70 (Güvenilir)</option>
            </select>
          </div>
          <div>
            <button
              className="btn-secondary"
              onClick={applyFilters}
              style={{display: 'flex', alignItems: 'center', gap: '8px'}}
            >
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
            {filteredRows.map((r, idx) => (
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
            {filteredRows.length === 0 && (
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
