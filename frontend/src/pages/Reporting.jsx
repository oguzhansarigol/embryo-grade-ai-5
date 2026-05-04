import React from 'react';
import { Download, FileText, Printer } from 'lucide-react';

export default function Reporting() {
  const handleExportPDF = () => {
    window.open('http://localhost:5000/api/export.pdf', '_blank');
  };

  const handleExportCSV = () => {
    window.open('http://localhost:5000/api/export.csv', '_blank');
  };

  return (
    <div>
      <h1 className="page-title">Raporlama</h1>
      <p className="page-subtitle">Sistemdeki tüm kayıtları dışa aktarın ve raporlayın.</p>

      <div className="grid-2">
        <div className="card" style={{display: 'flex', flexDirection: 'column', gap: '16px'}}>
          <h3 style={{marginBottom: '8px'}}>Dışa Aktarım Seçenekleri</h3>
          <p>Seçilen filtrelere veya tüm veritabanına ait sonuçları PDF veya CSV formatında indirebilirsiniz. Raporlar klinik kullanıma uygun olarak oluşturulmaktadır.</p>
          
          <button className="btn-primary" onClick={handleExportPDF} style={{display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px'}}>
            <FileText size={18} /> Klinik PDF Raporu İndir
          </button>
          
          <button className="btn-secondary" onClick={handleExportCSV} style={{display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px'}}>
            <Download size={18} /> Veritabanı CSV İndir
          </button>
        </div>

        <div className="card" style={{backgroundColor: '#E8ECEF'}}>
          <div style={{backgroundColor: '#FFF', padding: '32px', border: '1px solid #D8DDE3', boxShadow: '0 4px 12px rgba(0,0,0,0.05)', height: '400px', position: 'relative'}}>
            <div className="flex-between" style={{borderBottom: '2px solid var(--accent-blue)', paddingBottom: '16px', marginBottom: '24px'}}>
              <h2 style={{fontSize: '20px'}}>DeepEmbryo</h2>
              <div style={{textAlign: 'right', fontSize: '12px', color: 'var(--text-secondary)'}}>
                Tarih: {new Date().toLocaleDateString()}<br/>
                Klinik: Tüp Bebek Merkezi
              </div>
            </div>
            
            <h4 style={{marginBottom: '16px'}}>Toplu Analiz Özeti</h4>
            <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', fontSize: '14px', marginBottom: '32px'}}>
              <div>Toplam İncelenen: <strong>142</strong></div>
              <div>Yüksek Kalite (AA/AB): <strong>84</strong></div>
              <div>Manuel Kontrol Gereken: <strong>12</strong></div>
            </div>

            <div style={{position: 'absolute', bottom: '32px', right: '32px'}}>
              <button className="btn-secondary" style={{padding: '8px 12px'}}>
                <Printer size={16} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
