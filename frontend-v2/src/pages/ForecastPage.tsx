import React, { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import {
  Play,
  Square,
  Loader2,
  TrendingUp,
  AlertCircle,
  Flame,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { useTaskContext } from '@/context/TaskContext';
import { formatNumber } from '@/utils';

const MODEL_OPTIONS = [
  { value: 'auto', label: '自动比选 (auto)' },
  { value: 'lasso', label: 'Lasso' },
  { value: 'ridge', label: 'Ridge' },
  { value: 'elasticnet', label: 'ElasticNet' },
  { value: 'lstm', label: 'LSTM' },
  { value: 'xgboost', label: 'XGBoost' },
  { value: 'lightgbm', label: 'LightGBM' },
  { value: 'catboost', label: 'CatBoost' },
  { value: 'random_forest', label: 'Random Forest' },
  { value: 'sarimax', label: 'SARIMAX' },
  { value: 'timesfm', label: 'TimesFM' },
];

export const ForecastPage: React.FC = () => {
  const {
    backendAvailable,
    forecastTask: task,
    forecastLogs: logs,
    startForecastTask,
    stopForecastTask,
  } = useTaskContext();

  const [province, setProvince] = useState('河北');
  const [model, setModel] = useState('auto');
  const [asOfMonth, setAsOfMonth] = useState('2025-06-21');
  const [testStart, setTestStart] = useState('2025-11-01');
  const [testEnd, setTestEnd] = useState('2026-03-21');
  const [contextLen, setContextLen] = useState(270);
  const [isStarting, setIsStarting] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const handleStart = async () => {
    setIsStarting(true);
    try {
      await startForecastTask({
        model,
        province,
        asOfMonth,
        testStart,
        testEnd,
        contextLen,
        outputDir: `forecast_agent_output/${province}`,
      });
    } catch (err) {
      console.error('Failed to start forecast:', err);
    } finally {
      setIsStarting(false);
    }
  };

  const isRunning = task?.status === 'running';
  const metrics = task?.metrics || {};
  const testMetrics = metrics.test_metrics || {};
  const curve = metrics.forecast_curve || [];

  return (
    <div className="space-y-6 animate-fade-in-up">
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <Flame className="h-8 w-8 text-orange-500" />
          燃气旬度预测
        </h1>
        <p className="text-muted-foreground mt-1">
          基于 processed_data 旬度 Excel，预测 bridge + test 各旬销量
        </p>
      </div>

      {backendAvailable === false && (
        <Card className="glass border-destructive/50">
          <CardContent className="p-4 flex items-center gap-3">
            <AlertCircle className="h-5 w-5 text-destructive" />
            <span className="text-sm text-destructive">
              后端未连接，请先运行 frontend-v2/start.sh
            </span>
          </CardContent>
        </Card>
      )}

      <Card className="glass">
        <CardHeader>
          <CardTitle>任务配置</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <label className="text-sm">
            <span className="text-muted-foreground">省份</span>
            <input
              className="mt-1 w-full rounded-lg bg-secondary/50 border border-border px-3 py-2"
              value={province}
              onChange={(e) => setProvince(e.target.value)}
              disabled={isRunning}
            />
          </label>
          <label className="text-sm">
            <span className="text-muted-foreground">模型</span>
            <select
              className="mt-1 w-full rounded-lg bg-secondary/50 border border-border px-3 py-2"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={isRunning}
            >
              {MODEL_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
          <label className="text-sm">
            <span className="text-muted-foreground">context_len（旬）</span>
            <input
              type="number"
              className="mt-1 w-full rounded-lg bg-secondary/50 border border-border px-3 py-2"
              value={contextLen}
              onChange={(e) => setContextLen(Number(e.target.value))}
              disabled={isRunning}
            />
          </label>
          <label className="text-sm">
            <span className="text-muted-foreground">训练截止日</span>
            <input
              className="mt-1 w-full rounded-lg bg-secondary/50 border border-border px-3 py-2"
              value={asOfMonth}
              onChange={(e) => setAsOfMonth(e.target.value)}
              disabled={isRunning}
            />
          </label>
          <label className="text-sm">
            <span className="text-muted-foreground">测试起始日</span>
            <input
              className="mt-1 w-full rounded-lg bg-secondary/50 border border-border px-3 py-2"
              value={testStart}
              onChange={(e) => setTestStart(e.target.value)}
              disabled={isRunning}
            />
          </label>
          <label className="text-sm">
            <span className="text-muted-foreground">测试结束日</span>
            <input
              className="mt-1 w-full rounded-lg bg-secondary/50 border border-border px-3 py-2"
              value={testEnd}
              onChange={(e) => setTestEnd(e.target.value)}
              disabled={isRunning}
            />
          </label>
        </CardContent>
      </Card>

      <div className="flex gap-3">
        {!isRunning ? (
          <Button onClick={handleStart} disabled={isStarting || backendAvailable === false}>
            {isStarting ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Play className="h-4 w-4 mr-2" />}
            开始预测
          </Button>
        ) : (
          <Button variant="destructive" onClick={stopForecastTask}>
            <Square className="h-4 w-4 mr-2" />
            取消
          </Button>
        )}
        {task?.status && (
          <Badge variant={task.status === 'completed' ? 'default' : 'secondary'}>{task.status}</Badge>
        )}
        {metrics.best_model && <Badge variant="outline">最优: {metrics.best_model}</Badge>}
      </div>

      {(testMetrics.MAPE != null || testMetrics.RMSE != null) && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="glass p-4 text-center">
            <div className="text-xs text-muted-foreground">MAPE</div>
            <div className="text-lg font-bold">{formatNumber(testMetrics.MAPE, 2)}%</div>
          </Card>
          <Card className="glass p-4 text-center">
            <div className="text-xs text-muted-foreground">RMSE</div>
            <div className="text-lg font-bold">{formatNumber(testMetrics.RMSE, 0)}</div>
          </Card>
          <Card className="glass p-4 text-center">
            <div className="text-xs text-muted-foreground">MAE</div>
            <div className="text-lg font-bold">{formatNumber(testMetrics.MAE, 0)}</div>
          </Card>
          <Card className="glass p-4 text-center">
            <div className="text-xs text-muted-foreground">R²</div>
            <div className="text-lg font-bold">{formatNumber(testMetrics.R2, 4)}</div>
          </Card>
        </div>
      )}

      {curve.length > 0 && (
        <Card className="glass">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              预测曲线
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[280px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={curve}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis dataKey="ds" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="yhat" stroke="#f97316" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      <Card className="glass">
        <CardHeader>
          <CardTitle>运行日志</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64 overflow-y-auto font-mono text-xs bg-black/30 rounded-lg p-3 space-y-1">
            {logs.length === 0 && <p className="text-muted-foreground">暂无日志</p>}
            {logs.map((log) => (
              <div
                key={log.id}
                className={
                  log.level === 'error'
                    ? 'text-red-400'
                    : log.level === 'success'
                      ? 'text-green-400'
                      : 'text-foreground/80'
                }
              >
                [{log.timestamp}] {log.message}
              </div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
