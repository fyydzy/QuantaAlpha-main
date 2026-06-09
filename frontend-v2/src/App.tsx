import React, { useState, useEffect } from 'react';
// import { HomePage } from '@/pages/HomePage';
// import { MiningDashboardPage } from '@/pages/MiningDashboardPage';
// import { FactorLibraryPage } from '@/pages/FactorLibraryPage';
// import { BacktestPage } from '@/pages/BacktestPage';
import { WeatherForecastPage } from '@/pages/WeatherForecastPage';
import { ForecastPage } from '@/pages/ForecastPage';
import { ForecastSearchPage } from '@/pages/ForecastSearchPage';
import { SettingsPage } from '@/pages/SettingsPage';
import { Layout } from '@/components/layout/Layout';
import type { PageId } from '@/components/layout/Layout';
import { ParticleBackground } from '@/components/ParticleBackground';
import { TaskProvider, useTaskContext } from '@/context/TaskContext';

// Inner component to access context
const AppContent: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<PageId>('forecast');
  const { miningTask } = useTaskContext();

  // Auto-switch to dashboard when task starts
  useEffect(() => {
    if (miningTask && miningTask.status === 'running' && currentPage === 'home') {
       // Only auto-redirect if we are on home and a new task starts
       // But wait, user requirement says: "Don't disconnect when going back to home"
       // So we should redirect to dashboard ONLY when a NEW task is created via ChatInput
       // The ChatInput in HomePage calls startMining.
       // We can detect this change.
       setCurrentPage('mining_dashboard');
    }
  }, [miningTask?.taskId]); // Only trigger on new task ID

  return (
    <>
      <ParticleBackground />
      {/*
        Use display:none to hide non-current pages instead of conditional unmounting.
        This ensures that components are not unmounted when switching pages, so WebSocket/task state is not lost.
      */}
      {/* 因子挖掘 / 因子库 / 回测 暂时隐藏
      <div style={{ display: currentPage === 'home' ? 'block' : 'none' }}>
        <HomePage onNavigate={setCurrentPage} />
      </div>
      <div style={{ display: currentPage === 'mining_dashboard' ? 'block' : 'none' }}>
        <MiningDashboardPage onNavigate={setCurrentPage} />
      </div>
      <div style={{ display: currentPage === 'library' ? 'block' : 'none' }}>
        <Layout currentPage={currentPage} onNavigate={setCurrentPage}>
          <FactorLibraryPage />
        </Layout>
      </div>
      <div style={{ display: currentPage === 'backtest' ? 'block' : 'none' }}>
        <Layout currentPage={currentPage} onNavigate={setCurrentPage}>
          <BacktestPage />
        </Layout>
      </div>
      */}
      <div style={{ display: currentPage === 'weather' ? 'block' : 'none' }}>
        <Layout currentPage={currentPage} onNavigate={setCurrentPage}>
          <WeatherForecastPage />
        </Layout>
      </div>
      <div style={{ display: currentPage === 'forecast_search' ? 'block' : 'none' }}>
        <Layout currentPage={currentPage} onNavigate={setCurrentPage}>
          <ForecastSearchPage />
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
