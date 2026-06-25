import React, { useState } from 'react';
import { WeatherForecastPage } from '@/pages/WeatherForecastPage';
import { ForecastPage } from '@/pages/ForecastPage';
import { SettingsPage } from '@/pages/SettingsPage';
import { Layout } from '@/components/layout/Layout';
import type { PageId } from '@/components/layout/Layout';
import { ParticleBackground } from '@/components/ParticleBackground';
import { TaskProvider } from '@/context/TaskContext';

const AppContent: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<PageId>('forecast');

  return (
    <>
      <ParticleBackground />
      <div style={{ display: currentPage === 'weather' ? 'block' : 'none' }}>
        <Layout currentPage={currentPage} onNavigate={setCurrentPage}>
          <WeatherForecastPage />
        </Layout>
      </div>
      <div style={{ display: currentPage === 'forecast' ? 'block' : 'none' }}>
        <Layout currentPage={currentPage} onNavigate={setCurrentPage}>
          <ForecastPage />
        </Layout>
      </div>
      <div style={{ display: currentPage === 'settings' ? 'block' : 'none' }}>
        <Layout currentPage={currentPage} onNavigate={setCurrentPage}>
          <SettingsPage />
        </Layout>
      </div>
    </>
  );
};

export const App: React.FC = () => {
  return (
    <TaskProvider>
      <AppContent />
    </TaskProvider>
  );
};
