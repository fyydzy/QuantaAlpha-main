import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Badge } from '@/components/ui/Badge';
import { Progress } from '@/components/ui/Progress';
import {
  AlertCircle,
  Bot,
  ChevronDown,
  ChevronUp,
  Flame,
  Loader2,
  Send,
  Settings2,
  Square,
} from 'lucide-react';
import { useTaskContext } from '@/context/TaskContext';
import type { ForecastAgentMessage } from '@/context/TaskContext';
import { askForecastQa } from '@/services/api';

function getForecastQaContext(metrics: Record<string, any>) {
  const compare = metrics.compare || {};
  const recommend = metrics.recommend || {};
  return {
    outputDir: String(metrics.output_dir || metrics.outputDir || '').trim(),
    model: String(
      metrics.selected_model || compare.best_model || compare.bestModel || '',
    ).trim(),
    selectedFeatures: (recommend.feature_superset || recommend.featureSuperset || []) as string[],
  };
}

const QA_EXAMPLE_PROMPTS = [
  '哪几个旬误差最大？',
  '预测整体偏高还是偏低？',
  '目标月各旬实际和预测分别是多少？',
];

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

const EXAMPLE_PROMPTS = [
  '预测2026年4月河北天然气销量',
  '预测2025年12月山东天然气销量',
  '预测2026年3月北京天然气销量，重点关注采暖季',
];

const QUICK_REPLIES: Record<string, string[]> = {
  confirm_intent: ['好的，继续', '改成山东', '目标月改为2026年5月'],
  confirm_features: ['好的，继续', '去掉 HDD_squared', '加上 max_temp'],
};

type ForecastCurvePoint = { ds: string; yhat?: number; y?: number; period?: string };

function ForecastPredActualChart({ data }: { data: ForecastCurvePoint[] }) {
  if (!Array.isArray(data) || data.length === 0) return null;

  const chartData = data.map((row) => ({
    ds: row.ds,
    label: row.period ? `${row.ds} (${row.period})` : row.ds,
    predicted: row.yhat,
    actual: row.y,
  }));
  const hasActual = chartData.some((row) => row.actual != null && Number.isFinite(row.actual));

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={chartData} margin={{ top: 8, right: 12, left: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.5)" />
        <XAxis
          dataKey="ds"
          tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 10 }}
          interval="preserveStartEnd"
          minTickGap={28}
        />
        <YAxis
          tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 10 }}
          tickFormatter={(v: number) => (v >= 10000 ? `${(v / 10000).toFixed(1)}万` : String(v))}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: 'hsl(var(--card))',
            border: '1px solid hsl(var(--border))',
            borderRadius: '0.5rem',
            fontSize: 12,
          }}
          labelFormatter={(_, payload) => {
            const row = payload?.[0]?.payload as { label?: string; ds?: string } | undefined;
            return row?.label || row?.ds || '';
          }}
          formatter={(value: number, name: string) => [
            Number.isFinite(value) ? value.toLocaleString('zh-CN', { maximumFractionDigits: 0 }) : '-',
            name,
          ]}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line
          type="monotone"
          dataKey="predicted"
          name="预测值"
          stroke="#f97316"
          strokeWidth={2}
          dot={{ r: 2 }}
          connectNulls
        />
        {hasActual && (
          <Line
            type="monotone"
            dataKey="actual"
            name="真实值"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={{ r: 2 }}
            connectNulls
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}

function StagePayload({ stage, payload }: { stage: string; payload: any }) {
  if (!payload || typeof payload !== 'object') return null;

  const panel = (children: React.ReactNode) => (
    <div className="mt-3 rounded-lg border border-border/50 bg-secondary/20 p-3 text-sm">{children}</div>
  );

  if (stage === 'parse_intent' || stage === 'intent_applied' || stage === 'confirm_intent') {
    const data = payload.intent || payload;
    return panel(
      <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
        <div><span className="text-muted-foreground">省份</span> {data.province || '-'}</div>
        <div><span className="text-muted-foreground">目标月</span> {data.target_month || '-'}</div>
        <div><span className="text-muted-foreground">测试起</span> {data.test_start || '-'}</div>
        <div><span className="text-muted-foreground">测试止</span> {data.test_end || '-'}</div>
        <div className="col-span-2"><span className="text-muted-foreground">截止旬</span> {data.as_of_month || '-'}</div>
      </div>,
    );
  }

  if (stage === 'data_loading') {
    return panel(
      <div className="text-xs text-muted-foreground font-mono break-all">
        {payload.dataPath || '-'}
      </div>,
    );
  }

  if (stage === 'diagnose') {
    const top = Array.isArray(payload.topFeatures) ? payload.topFeatures.slice(0, 8) : [];
    return panel(
      <div className="space-y-2">
        {top.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {top.map((x: any, i: number) => (
              <Badge key={`f-${i}`} variant="outline" className="text-xs font-normal">
                {x.name} ({Number(x.score).toFixed(3)})
              </Badge>
            ))}
          </div>
        )}
      </div>,
    );
  }

  if (stage === 'recommend' || stage === 'features_applied' || stage === 'confirm_features') {
    const fs = Array.isArray(payload.featureSuperset) ? payload.featureSuperset : [];
    return panel(
      <div className="space-y-2">
        <div className="flex flex-wrap gap-1.5">
          {fs.map((f: string) => (
            <Badge key={f} variant="outline" className="text-xs font-normal">
              {f}
            </Badge>
          ))}
        </div>
        {payload.reason && <p className="text-xs text-muted-foreground leading-relaxed">{payload.reason}</p>}
      </div>,
    );
  }

  if (stage === 'compare') {
    const rows = Array.isArray(payload.leaderboard) ? payload.leaderboard.slice(0, 8) : [];
    const rollup = Array.isArray(payload.monthlyRollup) ? payload.monthlyRollup : [];
    const outputDir = String(payload.outputDir || payload.output_dir_display || '').trim();
    const curve: ForecastCurvePoint[] = Array.isArray(payload.forecastCurve)
      ? payload.forecastCurve
      : Array.isArray(payload.forecast_curve)
        ? payload.forecast_curve
        : [];
    return panel(
      <div className="space-y-3">
        {outputDir && (
          <p className="text-xs text-muted-foreground">
            结果存入 <span className="font-mono text-foreground/80 break-all">{outputDir}</span>
          </p>
        )}
        {payload.bestModel && (
          <div className="text-sm">
            最优模型：<span className="font-semibold text-orange-400">{payload.bestModel}</span>
          </div>
        )}
        {rows.length > 0 && (
          <div className="overflow-x-auto">
            <table className="min-w-full text-xs">
              <thead>
                <tr className="text-muted-foreground border-b border-border/40">
                  <th className="py-1.5 text-left font-medium">模型</th>
                  <th className="py-1.5 text-right font-medium">MAPE%</th>
                  <th className="py-1.5 text-right font-medium">RMSE</th>
                  <th className="py-1.5 text-right font-medium">R²</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r: any, i: number) => (
                  <tr key={`cmp-${i}`} className="border-b border-border/20">
                    <td className="py-1.5">{r.model}</td>
                    <td className="py-1.5 text-right">{r.MAPE?.toFixed?.(3) ?? '-'}</td>
                    <td className="py-1.5 text-right">{r.RMSE?.toFixed?.(1) ?? '-'}</td>
                    <td className="py-1.5 text-right">{r.R2?.toFixed?.(4) ?? '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {rollup.length > 0 && (
          <div>
            <div className="text-xs text-muted-foreground mb-1">月度汇总</div>
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead>
                  <tr className="text-muted-foreground border-b border-border/40">
                    <th className="py-1 text-left">月份</th>
                    <th className="py-1 text-right">预测</th>
                    <th className="py-1 text-right">实际</th>
                    <th className="py-1 text-right">MAPE%</th>
                  </tr>
                </thead>
                <tbody>
                  {rollup.map((r: any, i: number) => (
                    <tr key={`m-${i}`} className="border-b border-border/20">
                      <td className="py-1">{r.month}</td>
                      <td className="py-1 text-right">{Number(r.predicted_gas_sales ?? 0).toFixed(0)}</td>
                      <td className="py-1 text-right">
                        {r.actual_gas_sales == null ? '-' : Number(r.actual_gas_sales).toFixed(0)}
                      </td>
                      <td className="py-1 text-right">
                        {r.mape_pct == null ? '-' : Number(r.mape_pct).toFixed(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        {curve.length > 0 && (
          <div>
            <div className="text-xs text-muted-foreground mb-2">预测值 vs 真实值（测试集旬度）</div>
            <ForecastPredActualChart data={curve} />
          </div>
        )}
      </div>,
    );
  }

  if (stage === 'qa' && payload.answer) {
    return panel(renderMarkdownLike(String(payload.answer)));
  }

  return panel(
    <pre className="text-xs overflow-x-auto">{JSON.stringify(payload, null, 2)}</pre>,
  );
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end mb-5">
      <div className="max-w-[min(85%,32rem)] rounded-2xl rounded-tr-md bg-primary text-primary-foreground px-4 py-2.5 text-sm leading-relaxed shadow-sm">
        {text}
      </div>
    </div>
  );
}

function AssistantBlock({
  msg,
  renderPayload,
}: {
  msg: ForecastAgentMessage;
  renderPayload: (stage: string, payload: any) => React.ReactNode;
}) {
  return (
    <div className="flex gap-3 mb-6 group">
      <div className="shrink-0 w-8 h-8 rounded-full bg-orange-500/15 border border-orange-500/25 flex items-center justify-center mt-0.5">
        <Bot className="h-4 w-4 text-orange-500" />
      </div>
      <div className="flex-1 min-w-0 pt-0.5">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-medium text-muted-foreground">今冬明春预测智能体</span>
          {msg.needConfirm && (
            <Badge variant="outline" className="text-[10px] h-5 px-1.5">
              待你确认
            </Badge>
          )}
        </div>
        <p className="text-sm leading-relaxed text-foreground/95">{msg.text}</p>
        {renderPayload(msg.stage, msg.payload)}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex gap-3 mb-4">
      <div className="shrink-0 w-8 h-8 rounded-full bg-orange-500/15 border border-orange-500/25 flex items-center justify-center">
        <Bot className="h-4 w-4 text-orange-500" />
      </div>
      <div className="flex items-center gap-1 pt-2">
        <span className="w-2 h-2 rounded-full bg-muted-foreground/50 animate-bounce [animation-delay:0ms]" />
        <span className="w-2 h-2 rounded-full bg-muted-foreground/50 animate-bounce [animation-delay:150ms]" />
        <span className="w-2 h-2 rounded-full bg-muted-foreground/50 animate-bounce [animation-delay:300ms]" />
      </div>
    </div>
  );
}

export const ForecastPage: React.FC = () => {
  const {
    backendAvailable,
    forecastAgentTask,
    forecastAgentLogs,
    forecastAgentMessages,
    startForecastAgentTask,
    continueForecastAgentTask,
    stopForecastAgentTask,
  } = useTaskContext();

  const [input, setInput] = useState('');
  const [province, setProvince] = useState('河北');
  const [qaQuery, setQaQuery] = useState('');
  const [contextLen, setContextLen] = useState(270);
  const [showSettings, setShowSettings] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [followUpMessages, setFollowUpMessages] = useState<ForecastAgentMessage[]>([]);
  const [qaMode, setQaMode] = useState(true);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const running = forecastAgentTask?.status === 'running';
  const completed = forecastAgentTask?.status === 'completed';
  const metrics = forecastAgentTask?.metrics || {};
  const qaContext = useMemo(() => getForecastQaContext(metrics), [metrics]);
  const canAskQa = completed && qaMode && !!qaContext.outputDir && !!qaContext.model;

  useEffect(() => {
    if (completed) setQaMode(true);
  }, [completed, forecastAgentTask?.taskId]);

  useEffect(() => {
    for (let i = forecastAgentMessages.length - 1; i >= 0; i -= 1) {
      const msg = forecastAgentMessages[i];
      if (msg.stage !== 'parse_intent' && msg.stage !== 'intent_applied') continue;
      const p = String((msg.payload as { province?: string } | undefined)?.province || '').trim();
      if (p) {
        setProvince(p);
        break;
      }
    }
  }, [forecastAgentMessages]);
  const awaiting = (metrics.awaiting_confirmation || null) as any;
  const taskQuery = String((forecastAgentTask?.config as any)?.query || '').trim();
  const progressValue = Number(forecastAgentTask?.progress?.progress || 0);
  const progressMessage = String(forecastAgentTask?.progress?.message || '');

  const awaitingCheckpoint = awaiting?.checkpoint as string | undefined;
  const canReply = running && !!awaitingCheckpoint;
  const isBusy = submitting || (running && !canReply);

  const quickReplies = awaitingCheckpoint ? QUICK_REPLIES[awaitingCheckpoint] || [] : [];

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [forecastAgentMessages, followUpMessages, awaitingCheckpoint, running, taskQuery]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  }, [input]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || submitting) return;

    if (canAskQa) {
      const now = new Date().toISOString();
      const userMsg: ForecastAgentMessage = {
        role: 'user',
        stage: 'qa_followup',
        messageType: 'user_reply',
        text,
        timestamp: now,
      };
      setFollowUpMessages((prev) => [...prev, userMsg]);
      setInput('');
      setSubmitting(true);
      try {
        const resp = await askForecastQa({
          outputDir: qaContext.outputDir,
          model: qaContext.model,
          query: text,
          selectedFeatures: qaContext.selectedFeatures,
        });
        if (!resp.success || !resp.data) {
          throw new Error(resp.error || '问答失败');
        }
        const assistantMsg: ForecastAgentMessage = {
          role: 'assistant',
          stage: 'qa',
          messageType: 'qa_answer',
          text: '根据预测结果表，我的分析如下：',
          payload: resp.data,
          timestamp: new Date().toISOString(),
        };
        setFollowUpMessages((prev) => [...prev, assistantMsg]);
      } catch (err) {
        console.error('Failed to ask forecast QA:', err);
        const errMsg: ForecastAgentMessage = {
          role: 'assistant',
          stage: 'qa',
          messageType: 'qa_error',
          text: `问答失败：${err instanceof Error ? err.message : String(err)}`,
          timestamp: new Date().toISOString(),
        };
        setFollowUpMessages((prev) => [...prev, errMsg]);
      } finally {
        setSubmitting(false);
      }
      return;
    }

    if (!forecastAgentTask || forecastAgentTask.status !== 'running') {
      setSubmitting(true);
      try {
        setFollowUpMessages([]);
        setQaMode(true);
        await startForecastAgentTask({
          query: text,
          province,
          contextLen,
          candidateModels: 'catboost,elasticnet,lasso,lightgbm,lstm,random_forest,ridge,sarimax,xgboost',
          qaQuery: qaQuery.trim() || undefined,
        });
        setInput('');
      } catch (err) {
        console.error('Failed to start forecast agent:', err);
      } finally {
        setSubmitting(false);
      }
      return;
    }

    if (!canReply) return;

    setSubmitting(true);
    try {
      await continueForecastAgentTask({
        checkpoint: awaitingCheckpoint,
        message: text,
      });
      setInput('');
    } catch (err) {
      console.error('Failed to continue forecast agent:', err);
    } finally {
      setSubmitting(false);
    }
  }, [
    input,
    submitting,
    forecastAgentTask,
    canAskQa,
    qaContext,
    canReply,
    awaitingCheckpoint,
    province,
    contextLen,
    qaQuery,
    startForecastAgentTask,
    continueForecastAgentTask,
  ]);

  const handleNewForecast = useCallback(() => {
    setFollowUpMessages([]);
    setInput('');
    setQaMode(false);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const showWelcome = !forecastAgentTask && forecastAgentMessages.length === 0;

  const renderPayload = useCallback((stage: string, payload: any) => {
    return <StagePayload stage={stage} payload={payload} />;
  }, []);

  const placeholder = useMemo(() => {
    if (canReply) {
      return '回复助手：可直接说「继续」，或说明要改的省份、月份、特征…';
    }
    if (running) {
      return '智能体正在处理，请稍候…';
    }
    if (canAskQa) {
      return '继续追问预测结果，例如：哪几个旬误差最大？（Enter 发送）';
    }
    return '描述预测需求，例如：预测2026年4月河北天然气销量（Enter 发送，Shift+Enter 换行）';
  }, [canReply, running, canAskQa]);

  return (
    <div className="flex flex-col h-[calc(100vh-5rem)] max-h-[900px] animate-fade-in-up">
      {/* Header */}
      <div className="shrink-0 pb-3 border-b border-border/40">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h1 className="text-xl font-bold flex items-center gap-2">
              <Flame className="h-6 w-6 text-orange-500" />
              今冬明春预测智能体
            </h1>
            <p className="text-xs text-muted-foreground mt-0.5">
              自然语言对话 · 意图解析 · 特征推荐 · 多模型比选 · 结果问答
            </p>
          </div>
          <div className="flex items-center gap-2">
            {canAskQa && (
              <button
                type="button"
                onClick={handleNewForecast}
                className="text-xs px-3 py-1.5 rounded-lg border border-border/60 hover:bg-secondary/50 transition-colors"
              >
                新预测
              </button>
            )}
            {forecastAgentTask?.status && (
              <Badge variant={forecastAgentTask.status === 'completed' ? 'default' : 'outline'}>
                {forecastAgentTask.status}
              </Badge>
            )}
          </div>
        </div>

        {backendAvailable === false && (
          <div className="mt-2 flex items-center gap-2 text-xs text-destructive">
            <AlertCircle className="h-4 w-4 shrink-0" />
            后端未连接，请先启动 frontend-v2/backend/app.py
          </div>
        )}

        {forecastAgentTask && running && (
          <div className="mt-3 space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{progressMessage || '处理中…'}</span>
              <span>{progressValue}%</span>
            </div>
            <Progress value={progressValue} />
          </div>
        )}
      </div>

      {/* Chat area */}
      <div className="flex-1 overflow-y-auto py-4 pr-1 scrollbar-thin">
        {showWelcome && (
          <div className="flex gap-3 mb-6">
            <div className="shrink-0 w-8 h-8 rounded-full bg-orange-500/15 border border-orange-500/25 flex items-center justify-center">
              <Bot className="h-4 w-4 text-orange-500" />
            </div>
            <div className="flex-1 pt-0.5">
              <p className="text-sm leading-relaxed text-foreground/90">
                你好，我是今冬明春预测智能体。请直接告诉我你想预测哪个省份、哪个月份的销量，我会引导你确认参数和特征，然后自动完成多模型比选。
              </p>
              <div className="flex flex-wrap gap-2 mt-3">
                {EXAMPLE_PROMPTS.map((p) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => setInput(p)}
                    className="text-xs px-3 py-1.5 rounded-full border border-border/60 bg-secondary/30 hover:bg-secondary/60 transition-colors text-muted-foreground hover:text-foreground"
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        {taskQuery && <UserBubble text={taskQuery} />}

        {forecastAgentMessages.map((msg, i) =>
          msg.role === 'user' ? (
            <UserBubble key={`${msg.timestamp}-u-${i}`} text={msg.text} />
          ) : (
            <AssistantBlock key={`${msg.timestamp}-a-${i}`} msg={msg} renderPayload={renderPayload} />
          ),
        )}

        {followUpMessages.map((msg, i) =>
          msg.role === 'user' ? (
            <UserBubble key={`follow-${msg.timestamp}-u-${i}`} text={msg.text} />
          ) : (
            <AssistantBlock key={`follow-${msg.timestamp}-a-${i}`} msg={msg} renderPayload={renderPayload} />
          ),
        )}

        {(running && !awaitingCheckpoint) || (submitting && canAskQa) ? <TypingIndicator /> : null}

        <div ref={chatEndRef} />
      </div>

      {/* QA quick replies after forecast completes */}
      {canAskQa && !submitting && (
        <div className="shrink-0 flex flex-wrap gap-2 pb-2">
          {QA_EXAMPLE_PROMPTS.map((q) => (
            <button
              key={q}
              type="button"
              disabled={submitting}
              onClick={() => setInput(q)}
              className="text-xs px-3 py-1.5 rounded-full border border-primary/30 bg-primary/10 hover:bg-primary/20 text-foreground transition-colors disabled:opacity-50"
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {/* Quick replies */}
      {canReply && quickReplies.length > 0 && (
        <div className="shrink-0 flex flex-wrap gap-2 pb-2">
          {quickReplies.map((q) => (
            <button
              key={q}
              type="button"
              disabled={submitting}
              onClick={() => {
                setInput(q);
                void (async () => {
                  setSubmitting(true);
                  try {
                    await continueForecastAgentTask({
                      checkpoint: awaitingCheckpoint,
                      message: q,
                    });
                    setInput('');
                  } finally {
                    setSubmitting(false);
                  }
                })();
              }}
              className="text-xs px-3 py-1.5 rounded-full border border-primary/30 bg-primary/10 hover:bg-primary/20 text-foreground transition-colors disabled:opacity-50"
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {/* Composer */}
      <div className="shrink-0 pt-2 border-t border-border/40">
        <div className="glass rounded-xl border border-border/50 p-3">
          <div className="flex items-end gap-2">
            <button
              type="button"
              onClick={() => setShowSettings((v) => !v)}
              className={`shrink-0 p-2 rounded-lg transition-colors ${
                showSettings ? 'bg-secondary text-foreground' : 'text-muted-foreground hover:bg-secondary/50'
              }`}
              title="高级设置"
            >
              <Settings2 className="h-4 w-4" />
            </button>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={isBusy && !canReply}
              rows={1}
              className="flex-1 bg-transparent text-sm placeholder:text-muted-foreground focus:outline-none resize-none leading-relaxed min-h-[40px] max-h-[140px] py-2"
            />
            <div className="flex shrink-0 items-center gap-2">
              {canReply && (
                <button
                  type="button"
                  onClick={() => void handleSend()}
                  disabled={submitting || backendAvailable === false}
                  className="p-2.5 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-40 transition-colors"
                  title="发送 (Enter)"
                >
                  {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                </button>
              )}
              {running ? (
                <button
                  type="button"
                  onClick={stopForecastAgentTask}
                  className="p-2.5 rounded-lg bg-destructive text-destructive-foreground hover:bg-destructive/90 transition-colors"
                  title="取消任务"
                >
                  <Square className="h-4 w-4" />
                </button>
              ) : (
                !canReply && (
                  <button
                    type="button"
                    onClick={() => void handleSend()}
                    disabled={!input.trim() || submitting || backendAvailable === false}
                    className="p-2.5 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-40 transition-colors"
                    title="发送 (Enter)"
                  >
                    {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
                  </button>
                )
              )}
            </div>
          </div>

          {showSettings && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 mt-3 pt-3 border-t border-border/40">
              <input
                className="rounded-lg bg-secondary/50 border border-border px-3 py-1.5 text-xs"
                placeholder="兜底省份（对话未提及时）"
                value={province}
                onChange={(e) => setProvince(e.target.value)}
                disabled={running}
              />
              <input
                type="number"
                className="rounded-lg bg-secondary/50 border border-border px-3 py-1.5 text-xs"
                placeholder="context_len"
                value={contextLen}
                onChange={(e) => setContextLen(Number(e.target.value))}
                disabled={running}
              />
              <input
                className="rounded-lg bg-secondary/50 border border-border px-3 py-1.5 text-xs"
                placeholder="可选：结束后自动问答"
                value={qaQuery}
                onChange={(e) => setQaQuery(e.target.value)}
                disabled={running}
              />
            </div>
          )}
        </div>

        <button
          type="button"
          onClick={() => setShowLogs((v) => !v)}
          className="mt-2 flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {showLogs ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          运行日志 {forecastAgentLogs.length > 0 ? `(${forecastAgentLogs.length})` : ''}
        </button>

        {showLogs && (
          <div className="mt-2 max-h-36 overflow-y-auto font-mono text-[11px] bg-black/30 rounded-lg p-2 space-y-0.5 border border-border/40">
            {forecastAgentLogs.length === 0 && <p className="text-muted-foreground">暂无日志</p>}
            {forecastAgentLogs.map((log) => (
              <div
                key={log.id}
                className={
                  log.level === 'error'
                    ? 'text-red-400'
                    : log.level === 'success'
                      ? 'text-green-400'
                      : 'text-foreground/70'
                }
              >
                [{log.timestamp}] {log.message}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
