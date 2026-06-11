import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { AlertCircle, Flame, Loader2, Play, Square } from 'lucide-react';
import { useTaskContext } from '@/context/TaskContext';

function renderInlineMarkdown(text: string, keyPrefix: string): React.ReactNode {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).filter(Boolean);
  return parts.map((part, idx) => {
    if (part.startsWith('`') && part.endsWith('`')) {
      return (
        <code key={`${keyPrefix}-code-${idx}`} className="px-1.5 py-0.5 rounded bg-black/30 text-amber-300 text-xs">
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
    return <React.Fragment key={`${keyPrefix}-txt-${idx}`}>{part}</React.Fragment>;
  });
}

function renderMarkdownLike(answer: string): React.ReactNode {
  const lines = String(answer || '').split('\n');
  return (
    <div className="space-y-2">
      {lines.map((line, i) => {
        const l = line.trim();
        if (!l) return <div key={`empty-${i}`} className="h-1" />;
        const ul = l.match(/^[-*]\s+(.+)$/);
        if (ul) {
          return (
            <div key={`ul-${i}`} className="text-sm leading-6 pl-4">
              • {renderInlineMarkdown(ul[1], `ul-${i}`)}
            </div>
          );
        }
        const ol = l.match(/^\d+\.\s+(.+)$/);
        if (ol) {
          return (
            <div key={`ol-${i}`} className="text-sm leading-6 pl-4">
              {renderInlineMarkdown(ol[1], `ol-${i}`)}
            </div>
          );
        }
        return (
          <p key={`p-${i}`} className="text-sm leading-6 text-foreground/90">
            {renderInlineMarkdown(l, `p-${i}`)}
          </p>
        );
      })}
    </div>
  );
}

export const ForecastPage: React.FC = () => {
  const {
    backendAvailable,
    forecastAgentTask,
    forecastAgentLogs,
    startForecastAgentTask,
    stopForecastAgentTask,
  } = useTaskContext();

  const [query, setQuery] = useState('预测2026年4月河北天然气销量');
  const [province, setProvince] = useState('河北');
  const [qaQuery, setQaQuery] = useState('');
  const [contextLen, setContextLen] = useState(270);
  const [starting, setStarting] = useState(false);
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [forecastAgentLogs]);

  const running = forecastAgentTask?.status === 'running';
  const metrics = forecastAgentTask?.metrics || {};
  const stageMessage = metrics.agent_message as any;
  const compare = (metrics.compare || {}) as any;
  const leaderboard = (compare.leaderboard || []) as any[];
  const monthlyRollup = (compare.monthly_rollup || []) as any[];
  const qa = metrics.qa as any;

  const best = useMemo(() => leaderboard.find((r) => r.rank === 1) || leaderboard[0], [leaderboard]);

  const handleStart = async () => {
    const q = query.trim();
    if (!q) return;
    setStarting(true);
    try {
      await startForecastAgentTask({
        query: q,
        province,
        contextLen,
        candidateModels: 'catboost,elasticnet,lasso,lightgbm,lstm,random_forest,ridge,sarimax,xgboost',
        qaQuery: qaQuery.trim() || undefined,
      });
    } catch (err) {
      console.error('Failed to start forecast agent task:', err);
    } finally {
      setStarting(false);
    }
  };

  return (
    <div className="space-y-6 animate-fade-in-up">
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <Flame className="h-8 w-8 text-orange-500" />
          燃气预测智能体
        </h1>
        <p className="text-muted-foreground mt-1">
          对话输入一次请求，自动完成：意图解析 → 重要性诊断 → 特征推荐 → 比选预测 → 月度汇总 → 问答。
        </p>
      </div>

      {backendAvailable === false && (
        <Card className="glass border-destructive/50">
          <CardContent className="p-4 flex items-center gap-3">
            <AlertCircle className="h-5 w-5 text-destructive" />
            <span className="text-sm text-destructive">后端未连接，请先启动 frontend-v2/backend/app.py</span>
          </CardContent>
        </Card>
      )}

      <Card className="glass border-primary/30">
        <CardHeader>
          <CardTitle>对话输入</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <textarea
            className="w-full min-h-[76px] rounded-lg bg-secondary/50 border border-border px-3 py-2 text-sm"
            placeholder="例如：预测2026年4月河北天然气销量"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={starting || running}
          />
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <input
              className="rounded-lg bg-secondary/50 border border-border px-3 py-2 text-sm"
              placeholder="省份"
              value={province}
              onChange={(e) => setProvince(e.target.value)}
              disabled={starting || running}
            />
            <input
              type="number"
              className="rounded-lg bg-secondary/50 border border-border px-3 py-2 text-sm"
              placeholder="context_len"
              value={contextLen}
              onChange={(e) => setContextLen(Number(e.target.value))}
              disabled={starting || running}
            />
            <input
              className="rounded-lg bg-secondary/50 border border-border px-3 py-2 text-sm"
              placeholder="可选：流程结束后自动问答"
              value={qaQuery}
              onChange={(e) => setQaQuery(e.target.value)}
              disabled={starting || running}
            />
          </div>

          <div className="flex items-center gap-3">
            {running ? (
              <Button variant="destructive" onClick={stopForecastAgentTask}>
                <Square className="h-4 w-4 mr-2" />
                取消任务
              </Button>
            ) : (
              <Button onClick={handleStart} disabled={starting || backendAvailable === false || !query.trim()}>
                {starting ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : <Play className="h-4 w-4 mr-2" />}
                运行智能体
              </Button>
            )}
            {forecastAgentTask?.status && (
              <Badge variant={forecastAgentTask.status === 'completed' ? 'default' : 'outline'}>
                {forecastAgentTask.status}
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>

      {stageMessage && (
        <Card className="glass">
          <CardHeader>
            <CardTitle>阶段消息</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-xs text-muted-foreground">stage: {stageMessage.stage}</div>
            <div className="text-sm font-medium">{stageMessage.text}</div>
            {stageMessage.payload && (
              <pre className="text-xs overflow-x-auto bg-black/30 rounded p-3 border border-border">
                {JSON.stringify(stageMessage.payload, null, 2)}
              </pre>
            )}
          </CardContent>
        </Card>
      )}

      {best && (
        <Card className="glass">
          <CardHeader>
            <CardTitle>比选结果</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="text-sm">
              最优模型：<span className="font-semibold">{best.model}</span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="rounded-lg border border-border p-3 text-center">
                <div className="text-xs text-muted-foreground">MAPE</div>
                <div className="text-base font-semibold">{best.MAPE?.toFixed?.(3) ?? '-'}</div>
              </div>
              <div className="rounded-lg border border-border p-3 text-center">
                <div className="text-xs text-muted-foreground">RMSE</div>
                <div className="text-base font-semibold">{best.RMSE?.toFixed?.(1) ?? '-'}</div>
              </div>
              <div className="rounded-lg border border-border p-3 text-center">
                <div className="text-xs text-muted-foreground">MAE</div>
                <div className="text-base font-semibold">{best.MAE?.toFixed?.(1) ?? '-'}</div>
              </div>
              <div className="rounded-lg border border-border p-3 text-center">
                <div className="text-xs text-muted-foreground">R²</div>
                <div className="text-base font-semibold">{best.R2?.toFixed?.(4) ?? '-'}</div>
              </div>
            </div>

            {monthlyRollup.length > 0 && (
              <div className="space-y-2">
                <div className="text-sm font-medium">月度汇总</div>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-xs border border-border rounded">
                    <thead className="bg-secondary/30">
                      <tr>
                        <th className="px-2 py-2 text-left">month</th>
                        <th className="px-2 py-2 text-right">predicted</th>
                        <th className="px-2 py-2 text-right">actual</th>
                        <th className="px-2 py-2 text-right">mape_pct</th>
                      </tr>
                    </thead>
                    <tbody>
                      {monthlyRollup.map((r, i) => (
                        <tr key={`m-${i}`} className="border-t border-border/50">
                          <td className="px-2 py-2">{r.month}</td>
                          <td className="px-2 py-2 text-right">{Number(r.predicted_gas_sales ?? 0).toFixed(2)}</td>
                          <td className="px-2 py-2 text-right">
                            {r.actual_gas_sales == null ? '-' : Number(r.actual_gas_sales).toFixed(2)}
                          </td>
                          <td className="px-2 py-2 text-right">
                            {r.mape_pct == null ? '-' : Number(r.mape_pct).toFixed(3)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {qa?.answer && (
        <Card className="glass">
          <CardHeader>
            <CardTitle>自动问答结果</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="text-xs text-muted-foreground">
              LLM: {qa.model_used || '-'} | rows: {qa.rows_used ?? 0} | mode: {qa.data_mode || '-'}
            </div>
            <div>{renderMarkdownLike(String(qa.answer || ''))}</div>
          </CardContent>
        </Card>
      )}

      <Card className="glass">
        <CardHeader>
          <CardTitle>运行日志</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-64 overflow-y-auto font-mono text-xs bg-black/30 rounded-lg p-3 space-y-1">
            {forecastAgentLogs.length === 0 && <p className="text-muted-foreground">暂无日志</p>}
            {forecastAgentLogs.map((log) => (
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

