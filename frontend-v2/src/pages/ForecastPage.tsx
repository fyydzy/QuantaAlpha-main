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
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useTaskContext } from '@/context/TaskContext';
import { formatNumber } from '@/utils';
import { askForecastQa } from '@/services/api';

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
];

const TABULAR_FEATURE_OPTIONS = [
  'avg_temp',
  'max_temp',
  'min_temp',
  'HDD',
  'extreme_cold_days',
  'temp_range',
  'time_index',
  'month_sin',
  'month_cos',
  'ten_sin',
  'ten_cos',
  'is_heating_season',
  'spring_rework_peak',
  'Lag_36',
  'HDD_squared',
  'HDD_cross_Lag_36',
  'HDD_cross_HeatingSeason',
  'HDD_cross_spring_rework_peak',
  'ColdDays_cross_Lag_36',
];

const FEATURE_OPTIONS_BY_MODEL: Record<string, string[]> = {
  lasso: TABULAR_FEATURE_OPTIONS,
  ridge: TABULAR_FEATURE_OPTIONS,
  elasticnet: TABULAR_FEATURE_OPTIONS,
  xgboost: TABULAR_FEATURE_OPTIONS,
  lightgbm: TABULAR_FEATURE_OPTIONS,
  catboost: TABULAR_FEATURE_OPTIONS,
  random_forest: TABULAR_FEATURE_OPTIONS,
  sarimax: ['avg_temp', 'max_temp', 'min_temp', 'HDD', 'extreme_cold_days', 'temp_range'],
  lstm: ['Lag_36', 'HDD', 'is_heating_season'],
};

function renderInlineMarkdown(text: string, keyPrefix: string): React.ReactNode {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).filter(Boolean);
  return parts.map((part, idx) => {
    if (part.startsWith('`') && part.endsWith('`')) {
      return (
        <code
          key={`${keyPrefix}-code-${idx}`}
          className="px-1.5 py-0.5 rounded bg-black/30 text-amber-300 text-xs"
        >
          {part.slice(1, -1)}
        </code>
      );
    }
    if (part.startsWith('**') && part.endsWith('**')) {
      return (
        <strong key={`${keyPrefix}-strong-${idx}`} className="font-semibold text-foreground">
          {part.slice(2, -2)}
        </strong>
      );
    }
    return <React.Fragment key={`${keyPrefix}-text-${idx}`}>{part}</React.Fragment>;
  });
}

function renderMarkdownLike(answer: string): React.ReactNode {
  const nodes: React.ReactNode[] = [];
  const chunks = answer.split(/```/);
  let listBuffer: { type: 'ul' | 'ol'; items: string[] } | null = null;

  const flushList = (keyBase: string) => {
    if (!listBuffer || listBuffer.items.length === 0) return;
    if (listBuffer.type === 'ul') {
      nodes.push(
        <ul key={`${keyBase}-ul`} className="list-disc pl-5 space-y-1">
          {listBuffer.items.map((it, i) => (
            <li key={`${keyBase}-uli-${i}`} className="text-sm leading-6">
              {renderInlineMarkdown(it, `${keyBase}-uli-${i}`)}
            </li>
          ))}
        </ul>,
      );
    } else {
      nodes.push(
        <ol key={`${keyBase}-ol`} className="list-decimal pl-5 space-y-1">
          {listBuffer.items.map((it, i) => (
            <li key={`${keyBase}-oli-${i}`} className="text-sm leading-6">
              {renderInlineMarkdown(it, `${keyBase}-oli-${i}`)}
            </li>
          ))}
        </ol>,
      );
    }
    listBuffer = null;
  };

  chunks.forEach((chunk, chunkIdx) => {
    const keyBase = `md-${chunkIdx}`;
    // odd chunk index = code block content
    if (chunkIdx % 2 === 1) {
      flushList(`${keyBase}-before-code`);
      const codeText = chunk.replace(/^\s*\n/, '').replace(/\n\s*$/, '');
      nodes.push(
        <pre
          key={`${keyBase}-code`}
          className="overflow-x-auto rounded-lg bg-black/40 border border-border p-3 text-xs text-slate-100 leading-6"
        >
          <code>{codeText}</code>
        </pre>,
      );
      return;
    }

    const lines = chunk.split('\n');
    lines.forEach((rawLine, lineIdx) => {
      const line = rawLine.trim();
      const lineKey = `${keyBase}-line-${lineIdx}`;
      if (!line) {
        // Keep current list across blank lines; many LLM outputs place an empty line
        // between numbered items, and flushing here would restart numbering at 1.
        if (!listBuffer) {
          flushList(`${lineKey}-blank`);
        }
        return;
      }

      const ulMatch = line.match(/^[-*]\s+(.+)$/);
      if (ulMatch) {
        if (!listBuffer || listBuffer.type !== 'ul') {
          flushList(`${lineKey}-switch-ul`);
          listBuffer = { type: 'ul', items: [] };
        }
        listBuffer.items.push(ulMatch[1]);
        return;
      }

      const olMatch = line.match(/^\d+\.\s+(.+)$/);
      if (olMatch) {
        if (!listBuffer || listBuffer.type !== 'ol') {
          flushList(`${lineKey}-switch-ol`);
          listBuffer = { type: 'ol', items: [] };
        }
        listBuffer.items.push(olMatch[1]);
        return;
      }

      flushList(`${lineKey}-before-text`);

      const heading = line.match(/^(#{1,3})\s+(.+)$/);
      if (heading) {
        const level = heading[1].length;
        const title = heading[2];
        const cls =
          level === 1
            ? 'text-base font-semibold'
            : level === 2
              ? 'text-sm font-semibold'
              : 'text-sm font-medium';
        nodes.push(
          <div key={`${lineKey}-h`} className={`${cls} text-foreground mt-1`}>
            {renderInlineMarkdown(title, `${lineKey}-h`)}
          </div>,
        );
        return;
      }

      nodes.push(
        <p key={`${lineKey}-p`} className="text-sm leading-6 text-foreground/90">
          {renderInlineMarkdown(line, `${lineKey}-p`)}
        </p>,
      );
    });
  });

  flushList('md-end');
  return <div className="space-y-2">{nodes}</div>;
}

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
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [isStarting, setIsStarting] = useState(false);
  const [qaQuery, setQaQuery] = useState('');
  const [qaAnswer, setQaAnswer] = useState('');
  const [qaError, setQaError] = useState('');
  const [qaLoading, setQaLoading] = useState(false);
  const [qaMeta, setQaMeta] = useState<{ modelUsed: string; rowsUsed: number; dataMode: string } | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const modelFeatureOptions = FEATURE_OPTIONS_BY_MODEL[model] || [];

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  useEffect(() => {
    setSelectedFeatures((prev) => prev.filter((f) => modelFeatureOptions.includes(f)));
  }, [model]);

  useEffect(() => {
    setQaAnswer('');
    setQaError('');
    setQaMeta(null);
  }, [task?.taskId, task?.status]);

  const toggleFeature = (feature: string) => {
    setSelectedFeatures((prev) => (
      prev.includes(feature)
        ? prev.filter((f) => f !== feature)
        : [...prev, feature]
    ));
  };

  const handleStart = async () => {
    setIsStarting(true);
    try {
      const modelFeatures = model !== 'auto' && modelFeatureOptions.length > 0 && selectedFeatures.length > 0
        ? { [model]: selectedFeatures }
        : undefined;
      await startForecastTask({
        model,
        province,
        asOfMonth,
        testStart,
        testEnd,
        contextLen,
        outputDir: `forecast_agent_output/${province}`,
        modelFeatures,
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
  const hasActual = curve.some((p: { y?: number }) => p.y != null && !Number.isNaN(p.y));
  const qaModel = String(metrics.display_model || metrics.best_model || model || '').trim();
  const qaOutputDir = String(task?.config?.outputDir || `forecast_agent_output/${province}`);

  const handleAskQa = async () => {
    const query = qaQuery.trim();
    if (!query) return;
    if (!qaModel || qaModel === 'auto' || qaModel === 'auto_select') {
      setQaError('请先完成预测并确定具体模型，再进行问答。');
      return;
    }
    setQaLoading(true);
    setQaError('');
    try {
      const resp = await askForecastQa({
        outputDir: qaOutputDir,
        model: qaModel,
        query,
        selectedFeatures,
      });
      if (!resp.success || !resp.data) {
        throw new Error(resp.error || '问答请求失败');
      }
      setQaAnswer(resp.data.answer || '');
      setQaMeta({
        modelUsed: resp.data.modelUsed || '',
        rowsUsed: Number(resp.data.rowsUsed || 0),
        dataMode: resp.data.dataMode || '',
      });
    } catch (err: any) {
      setQaError(String(err?.message || err || '问答失败'));
    } finally {
      setQaLoading(false);
    }
  };

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
          <div className="text-sm md:col-span-2 lg:col-span-3">
            <span className="text-muted-foreground">特征组合选择（当前模型）</span>
            <div className="mt-2 rounded-lg bg-secondary/30 border border-border p-3">
              {model === 'auto' && (
                <p className="text-xs text-muted-foreground">
                  auto 比选暂不支持在页面里统一指定特征；请先选择具体模型后再勾选特征。
                </p>
              )}
              {model === 'timesfm' && (
                <p className="text-xs text-muted-foreground">
                  TimesFM 不使用 tabular 特征列，无需选择。
                </p>
              )}
              {model !== 'auto' && model !== 'timesfm' && (
                <>
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                    {modelFeatureOptions.map((feature) => (
                      <label
                        key={feature}
                        className="flex items-center gap-2 text-xs text-foreground/90"
                      >
                        <input
                          type="checkbox"
                          checked={selectedFeatures.includes(feature)}
                          onChange={() => toggleFeature(feature)}
                          disabled={isRunning}
                        />
                        <span>{feature}</span>
                      </label>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground mt-2">
                    不勾选任何特征时，仍按配置默认行为（通常为该模型默认全集）。
                  </p>
                </>
              )}
            </div>
          </div>
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
          <Badge variant={task.status === 'completed' ? 'default' : 'outline'}>{task.status}</Badge>
        )}
        {(metrics.display_model || metrics.best_model) && (
          <Badge variant="outline">
            展示: {metrics.display_model || metrics.best_model}
          </Badge>
        )}
        {Array.isArray(metrics.feature_cols_used) && metrics.feature_cols_used.length > 0 && (
          <Badge variant="outline" title={metrics.feature_cols_used.join(', ')}>
            特征 {metrics.feature_cols_used.length} 列
          </Badge>
        )}
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
              预测 vs 真实
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground mb-3">
              橙色为模型预测（含 bridge 与 test）；蓝色为测试区间真实销量（有实测值的旬）。
              {!hasActual && ' 当前结果中暂无真实值，请重新跑预测后刷新。'}
            </p>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={curve}
                  margin={{ top: 8, right: 16, left: 8, bottom: 28 }}
                >
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis
                    dataKey="ds"
                    tick={{ fontSize: 10 }}
                    interval={0}
                    minTickGap={0}
                    angle={-28}
                    textAnchor="end"
                    height={56}
                    tickFormatter={(v: string) => (v && v.length >= 10 ? v.slice(5) : v)}
                  />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip
                    formatter={(value: number, name: string) => [
                      formatNumber(value, 0),
                      name === 'yhat' ? '预测' : '真实',
                    ]}
                  />
                  <Legend
                    formatter={(value) => (value === 'yhat' ? '预测' : '真实')}
                  />
                  <Line
                    type="monotone"
                    dataKey="yhat"
                    name="yhat"
                    stroke="#f97316"
                    strokeWidth={2}
                    dot={false}
                  />
                  {hasActual && (
                    <Line
                      type="monotone"
                      dataKey="y"
                      name="y"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={{ r: 3 }}
                      connectNulls={false}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      <Card className="glass">
        <CardHeader>
          <CardTitle>LLM 结果问答</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-xs text-muted-foreground">
            基于当前模型结果表（真实值/预测值）和特征列回答问题。
          </p>
          <textarea
            className="w-full min-h-[84px] rounded-lg bg-secondary/50 border border-border px-3 py-2 text-sm"
            placeholder="例如：为什么 2025-12 到 2026-01 的误差更大？"
            value={qaQuery}
            onChange={(e) => setQaQuery(e.target.value)}
            disabled={qaLoading}
          />
          <div className="flex items-center gap-3">
            <Button onClick={handleAskQa} disabled={qaLoading || !qaQuery.trim()}>
              {qaLoading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
              发送提问
            </Button>
            <span className="text-xs text-muted-foreground">
              结果模型: {qaModel || '-'} | 输出目录: {qaOutputDir}
            </span>
          </div>
          {qaError && <div className="text-sm text-red-400">{qaError}</div>}
          {qaAnswer && (
            <div className="rounded-lg border border-border bg-secondary/30 p-3 space-y-2">
              {qaMeta && (
                <div className="text-xs text-muted-foreground">
                  LLM: {qaMeta.modelUsed || '-'} | rows: {qaMeta.rowsUsed} | mode: {qaMeta.dataMode}
                </div>
              )}
              <div>{renderMarkdownLike(qaAnswer)}</div>
            </div>
          )}
        </CardContent>
      </Card>

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
