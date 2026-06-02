import React, { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import {
  Play,
  Square,
  Loader2,
  AlertCircle,
  ChevronDown,
  ChevronRight,
  Flame,
  TrendingUp,
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

type ForecastFeedbackPayload = {
  best_solution_id?: string;
  best_solution_name?: string;
  conclusion?: string;
  effective_features?: string[] | string;
  business_interpretation?: string;
  next_step?: string;
};

type SolutionLibraryPayload = {
  goal?: string;
  fixed_model?: string;
  solutions?: Array<{
    solution_id?: string;
    name?: string;
    hypothesis?: string;
    feature_set?: string[];
  }>;
};

type SolutionTestPoint = {
  ds: string;
  period?: string;
  yhat?: number;
  y?: number;
  error_pct?: number;
};

type SolutionTestResult = {
  solution_id?: string;
  name?: string;
  model?: string;
  mape?: number;
  rmse?: number;
  r2?: number;
  curve?: SolutionTestPoint[];
  table?: SolutionTestPoint[];
};

function normalizeFeatures(raw: string[] | string | undefined): string[] {
  if (Array.isArray(raw)) return raw.map((x) => String(x)).filter((x) => x.trim().length > 0);
  if (typeof raw === 'string') {
    return raw
      .split(',')
      .map((x) => x.trim())
      .filter((x) => x.length > 0);
  }
  return [];
}

export const ForecastSearchPage: React.FC = () => {
  const {
    backendAvailable,
    forecastSearchTask,
    forecastSearchLogs,
    startForecastSearchTask,
    stopForecastSearchTask,
  } = useTaskContext();

  const [province, setProvince] = useState('河北');
  const [asOfMonth, setAsOfMonth] = useState('2025-06-21');
  const [testStart, setTestStart] = useState('2025-11-01');
  const [testEnd, setTestEnd] = useState('2026-03-21');
  const [contextLen, setContextLen] = useState(270);

  const [goal, setGoal] = useState('降低测试集 MAPE，给出更可解释的特征组合方案');
  const [isStartingSearch, setIsStartingSearch] = useState(false);
  const searchLogsEndRef = useRef<HTMLDivElement>(null);

  const [expandedSolutionId, setExpandedSolutionId] = useState<string | null>(null);
  const [numberOfSolutions, setNumberOfSolutions] = useState(4);
  const [maxFeatureCount, setMaxFeatureCount] = useState(10);
  const [selectedTestSolutionId, setSelectedTestSolutionId] = useState('');

  useEffect(() => {
    searchLogsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [forecastSearchLogs]);

  const isSearchRunning = forecastSearchTask?.status === 'running';
  const searchMetrics = forecastSearchTask?.metrics || {};
  const leaderboard = (searchMetrics.leaderboard || []) as any[];
  const solutionLibrary = (searchMetrics.solution_library || null) as SolutionLibraryPayload | null;
  const feedback = (searchMetrics.forecast_feedback || null) as ForecastFeedbackPayload | null;
  const effectiveFeatures = normalizeFeatures(feedback?.effective_features);
  const proposedSolutions = solutionLibrary?.solutions || [];
  const solutionTestResults = (searchMetrics.solution_test_results || {}) as Record<
    string,
    SolutionTestResult
  >;
  const testSolutionIds = Object.keys(solutionTestResults);

  useEffect(() => {
    if (testSolutionIds.length === 0) return;
    const bestId = feedback?.best_solution_id;
    if (bestId && solutionTestResults[bestId]) {
      setSelectedTestSolutionId(bestId);
      return;
    }
    const rank1 = leaderboard[0]?.solution_id;
    if (rank1 && solutionTestResults[rank1]) {
      setSelectedTestSolutionId(rank1);
      return;
    }
    if (!selectedTestSolutionId || !solutionTestResults[selectedTestSolutionId]) {
      setSelectedTestSolutionId(testSolutionIds[0]);
    }
  }, [testSolutionIds.join(','), feedback?.best_solution_id, leaderboard[0]?.solution_id]);

  const selectedTestResult = selectedTestSolutionId
    ? solutionTestResults[selectedTestSolutionId]
    : null;
  const testCurve = selectedTestResult?.curve || [];
  const testTable = selectedTestResult?.table || [];
  const hasTestActual = testCurve.some((p) => p.y != null && !Number.isNaN(p.y));

  const handleStartSearch = async () => {
    setIsStartingSearch(true);
    try {
      await startForecastSearchTask({
        goal,
        province,
        asOfMonth,
        testStart,
        testEnd,
        contextLen,
        outputDir: `forecast_agent_output/${province}`,
        numberOfSolutions,
        maxFeatureCount,
      });
    } catch (err) {
      console.error('Failed to start forecast_search:', err);
    } finally {
      setIsStartingSearch(false);
    }
  };

  const handleGenerateFeedbackOnly = async () => {
    setIsStartingSearch(true);
    try {
      await startForecastSearchTask({
        goal,
        province,
        asOfMonth,
        testStart,
        testEnd,
        contextLen,
        outputDir: `forecast_agent_output/${province}`,
        onlyFeedback: true,
      });
    } catch (err) {
      console.error('Failed to generate feedback only:', err);
    } finally {
      setIsStartingSearch(false);
    }
  };

  return (
    <div className="space-y-6 animate-fade-in-up">
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <Flame className="h-8 w-8 text-orange-500" />
          特征方案搜索（燃气预测）
        </h1>
        <p className="text-muted-foreground mt-1">
          LLM 生成特征组合 → 固定模型跑评测 → 排行榜 → 自动生成总结
        </p>
      </div>

      {backendAvailable === false && (
        <Card className="glass border-destructive/50">
          <CardContent className="p-4 flex items-center gap-3">
            <AlertCircle className="h-5 w-5 text-destructive" />
            <span className="text-sm text-destructive">后端未连接，请先启动后端服务</span>
          </CardContent>
        </Card>
      )}

      <Card className="glass">
        <CardHeader>
          <CardTitle>搜索配置</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <label className="text-sm">
            <span className="text-muted-foreground">省份</span>
            <input
              className="mt-1 w-full rounded-lg bg-secondary/50 border border-border px-3 py-2"
              value={province}
              onChange={(e) => setProvince(e.target.value)}
              disabled={isSearchRunning}
            />
          </label>
          <label className="text-sm">
            <span className="text-muted-foreground">context_len（旬）</span>
            <input
              type="number"
              className="mt-1 w-full rounded-lg bg-secondary/50 border border-border px-3 py-2"
              value={contextLen}
              onChange={(e) => setContextLen(Number(e.target.value))}
              disabled={isSearchRunning}
            />
          </label>
          <label className="text-sm">
            <span className="text-muted-foreground">训练截止日</span>
            <input
              className="mt-1 w-full rounded-lg bg-secondary/50 border border-border px-3 py-2"
              value={asOfMonth}
              onChange={(e) => setAsOfMonth(e.target.value)}
              disabled={isSearchRunning}
            />
          </label>
          <label className="text-sm">
            <span className="text-muted-foreground">测试起始日</span>
            <input
              className="mt-1 w-full rounded-lg bg-secondary/50 border border-border px-3 py-2"
              value={testStart}
              onChange={(e) => setTestStart(e.target.value)}
              disabled={isSearchRunning}
            />
          </label>
          <label className="text-sm">
            <span className="text-muted-foreground">测试结束日</span>
            <input
              className="mt-1 w-full rounded-lg bg-secondary/50 border border-border px-3 py-2"
              value={testEnd}
              onChange={(e) => setTestEnd(e.target.value)}
              disabled={isSearchRunning}
            />
          </label>
        </CardContent>
      </Card>

      <Card className="glass">
        <CardHeader>
          <CardTitle>LLM 目标与运行</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <label className="text-sm block">
            <span className="text-muted-foreground">研究目标（goal）</span>
            <textarea
              className="mt-1 w-full rounded-lg bg-secondary/50 border border-border px-3 py-2 min-h-[88px]"
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              disabled={isSearchRunning}
            />
          </label>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="text-sm">
              <div className="text-muted-foreground">number_of_solutions</div>
              <div className="mt-1 flex items-center gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setNumberOfSolutions((v) => Math.max(1, v - 1))}
                  disabled={isSearchRunning}
                >
                  -
                </Button>
                <input
                  type="number"
                  className="w-24 rounded-lg bg-secondary/50 border border-border px-3 py-2"
                  value={numberOfSolutions}
                  onChange={(e) => setNumberOfSolutions(Math.max(1, Number(e.target.value) || 1))}
                  disabled={isSearchRunning}
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setNumberOfSolutions((v) => Math.min(20, v + 1))}
                  disabled={isSearchRunning}
                >
                  +
                </Button>
                <span className="text-xs text-muted-foreground">（1-20）</span>
              </div>
            </div>
            <div className="text-sm">
              <div className="text-muted-foreground">max_feature_count</div>
              <div className="mt-1 flex items-center gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setMaxFeatureCount((v) => Math.max(1, v - 1))}
                  disabled={isSearchRunning}
                >
                  -
                </Button>
                <input
                  type="number"
                  className="w-24 rounded-lg bg-secondary/50 border border-border px-3 py-2"
                  value={maxFeatureCount}
                  onChange={(e) => setMaxFeatureCount(Math.max(1, Number(e.target.value) || 1))}
                  disabled={isSearchRunning}
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setMaxFeatureCount((v) => Math.min(50, v + 1))}
                  disabled={isSearchRunning}
                >
                  +
                </Button>
                <span className="text-xs text-muted-foreground">（1-50）</span>
              </div>
            </div>
          </div>

          <div className="flex gap-3 items-center">
            {!isSearchRunning ? (
              <>
                <Button onClick={handleStartSearch} disabled={isStartingSearch || backendAvailable === false}>
                  {isStartingSearch ? (
                    <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  ) : (
                    <Play className="h-4 w-4 mr-2" />
                  )}
                  开始方案搜索
                </Button>
                <Button
                  variant="secondary"
                  onClick={handleGenerateFeedbackOnly}
                  disabled={isStartingSearch || backendAvailable === false}
                >
                  只生成总结
                </Button>
              </>
            ) : (
              <Button variant="destructive" onClick={stopForecastSearchTask}>
                <Square className="h-4 w-4 mr-2" />
                取消
              </Button>
            )}
            {forecastSearchTask?.status && (
              <Badge variant={forecastSearchTask.status === 'completed' ? 'default' : 'outline'}>
                {forecastSearchTask.status}
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>

      {leaderboard.length > 0 && (
        <Card className="glass p-4">
          <div className="text-sm font-semibold mb-2">排行榜（按 MAPE 排序）</div>
          <div className="text-xs overflow-x-auto">
            <table className="min-w-[700px] w-full">
              <thead className="text-muted-foreground">
                <tr>
                  <th className="text-left py-1 pr-3"></th>
                  <th className="text-left py-1 pr-3">rank</th>
                  <th className="text-left py-1 pr-3">solution_id</th>
                  <th className="text-left py-1 pr-3">name</th>
                  <th className="text-left py-1 pr-3">MAPE</th>
                  <th className="text-left py-1 pr-3">RMSE</th>
                  <th className="text-left py-1 pr-3">R2</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.slice(0, 10).map((r, i) => (
                  <React.Fragment key={i}>
                    <tr className="border-t border-border/50">
                      <td className="py-1 pr-3">
                        <button
                          className="text-muted-foreground hover:text-foreground"
                          onClick={() =>
                            setExpandedSolutionId((prev) => (prev === r.solution_id ? null : r.solution_id))
                          }
                          title="展开/收起"
                          type="button"
                        >
                          {expandedSolutionId === r.solution_id ? (
                            <ChevronDown className="h-4 w-4" />
                          ) : (
                            <ChevronRight className="h-4 w-4" />
                          )}
                        </button>
                      </td>
                      <td className="py-1 pr-3">{r.rank ?? i + 1}</td>
                      <td className="py-1 pr-3">{r.solution_id}</td>
                      <td className="py-1 pr-3">{r.name}</td>
                      <td className="py-1 pr-3">{formatNumber(r.MAPE, 2)}%</td>
                      <td className="py-1 pr-3">{formatNumber(r.RMSE, 0)}</td>
                      <td className="py-1 pr-3">{formatNumber(r.R2, 4)}</td>
                    </tr>
                    {expandedSolutionId === r.solution_id && (
                      <tr className="border-t border-border/50">
                        <td colSpan={7} className="py-2 pr-3">
                          <div className="space-y-2">
                            <div>
                              <div className="text-muted-foreground mb-1">hypothesis</div>
                              <div className="whitespace-pre-wrap">{r.hypothesis || ''}</div>
                            </div>
                            <div>
                              <div className="text-muted-foreground mb-1">feature_set</div>
                              <div className="whitespace-pre-wrap break-words">
                                {(r.features || '')
                                  .split(',')
                                  .filter((x: string) => x.trim().length > 0)
                                  .join(', ')}
                              </div>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {testSolutionIds.length > 0 && (
        <Card className="glass">
          <CardHeader>
            <CardTitle className="flex flex-wrap items-center gap-2">
              <TrendingUp className="h-5 w-5 text-orange-500" />
              测试集预测结果
            </CardTitle>
            <div className="flex flex-wrap items-center gap-3 mt-2">
              <label className="text-sm flex items-center gap-2">
                <span className="text-muted-foreground">查看方案</span>
                <select
                  className="rounded-lg bg-secondary/50 border border-border px-3 py-1.5 text-sm"
                  value={selectedTestSolutionId}
                  onChange={(e) => setSelectedTestSolutionId(e.target.value)}
                >
                  {testSolutionIds.map((sid) => {
                    const item = solutionTestResults[sid];
                    return (
                      <option key={sid} value={sid}>
                        {sid} — {item?.name || sid}
                        {item?.mape != null ? ` (MAPE ${formatNumber(item.mape, 2)}%)` : ''}
                      </option>
                    );
                  })}
                </select>
              </label>
              {selectedTestResult?.mape != null && (
                <Badge variant="outline">MAPE {formatNumber(selectedTestResult.mape, 2)}%</Badge>
              )}
              {selectedTestResult?.rmse != null && (
                <Badge variant="outline">RMSE {formatNumber(selectedTestResult.rmse, 0)}</Badge>
              )}
            </div>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
              <div className="rounded-xl border border-border/60 bg-secondary/10 p-3">
                <div className="text-sm font-medium mb-2">预测 vs 真实（测试集）</div>
                {testCurve.length > 0 ? (
                  <div className="h-[320px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={testCurve}
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
                        <Legend formatter={(value) => (value === 'yhat' ? '预测' : '真实')} />
                        <Line
                          type="monotone"
                          dataKey="yhat"
                          name="yhat"
                          stroke="#f97316"
                          strokeWidth={2}
                          dot={{ r: 3 }}
                        />
                        {hasTestActual && (
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
                ) : (
                  <p className="text-sm text-muted-foreground py-8 text-center">暂无曲线数据</p>
                )}
              </div>

              <div className="rounded-xl border border-border/60 bg-secondary/10 p-3">
                <div className="text-sm font-medium mb-2">测试集明细</div>
                <div className="max-h-[300px] overflow-auto text-xs">
                  <table className="w-full min-w-[320px]">
                    <thead className="text-muted-foreground sticky top-0 bg-secondary/80">
                      <tr>
                        <th className="text-left py-2 pr-2">日期</th>
                        <th className="text-left py-2 pr-2">旬</th>
                        <th className="text-right py-2 pr-2">预测</th>
                        <th className="text-right py-2 pr-2">真实</th>
                        <th className="text-right py-2">误差%</th>
                      </tr>
                    </thead>
                    <tbody>
                      {testTable.length === 0 && (
                        <tr>
                          <td colSpan={5} className="py-4 text-center text-muted-foreground">
                            暂无表格数据
                          </td>
                        </tr>
                      )}
                      {testTable.map((row, idx) => (
                        <tr key={`${row.ds}-${idx}`} className="border-t border-border/40">
                          <td className="py-1.5 pr-2">{row.ds}</td>
                          <td className="py-1.5 pr-2 text-muted-foreground">{row.period || '-'}</td>
                          <td className="py-1.5 pr-2 text-right">
                            {row.yhat != null ? formatNumber(row.yhat, 0) : '-'}
                          </td>
                          <td className="py-1.5 pr-2 text-right">
                            {row.y != null ? formatNumber(row.y, 0) : '-'}
                          </td>
                          <td className="py-1.5 text-right">
                            {row.error_pct != null ? `${formatNumber(row.error_pct, 2)}%` : '-'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {proposedSolutions.length > 0 && (
        <Card className="glass">
          <CardHeader>
            <CardTitle className="flex flex-wrap items-center gap-2">
              LLM 生成的特征方案
              {solutionLibrary?.fixed_model && (
                <Badge variant="outline">固定模型: {solutionLibrary.fixed_model}</Badge>
              )}
            </CardTitle>
            {solutionLibrary?.goal && (
              <p className="text-sm text-muted-foreground mt-1">{solutionLibrary.goal}</p>
            )}
          </CardHeader>
          <CardContent className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {proposedSolutions.map((sol, idx) => {
              const features = normalizeFeatures(sol.feature_set);
              return (
                <div
                  key={sol.solution_id || idx}
                  className="rounded-xl border border-border/60 bg-secondary/20 p-4 space-y-3"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <div className="text-xs text-muted-foreground">{sol.solution_id || `方案 ${idx + 1}`}</div>
                      <div className="text-base font-semibold mt-0.5">{sol.name || '未命名方案'}</div>
                    </div>
                    <Badge variant="outline">{features.length} 个特征</Badge>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">假设（hypothesis）</div>
                    <p className="text-sm leading-7 text-foreground/90">{sol.hypothesis || '-'}</p>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground mb-2">特征组合</div>
                    <div className="flex flex-wrap gap-1.5">
                      {features.map((f) => (
                        <Badge key={f} variant="outline" className="text-xs font-normal">
                          {f}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>
      )}

      {feedback && (
        <Card className="glass">
          <CardHeader>
            <CardTitle>LLM 总结</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="rounded-lg border border-border/60 bg-secondary/30 p-3">
                <div className="text-xs text-muted-foreground">最优方案 ID</div>
                <div className="text-sm font-medium mt-1">{feedback.best_solution_id || '-'}</div>
              </div>
              <div className="rounded-lg border border-border/60 bg-secondary/30 p-3">
                <div className="text-xs text-muted-foreground">方案名称</div>
                <div className="text-sm font-medium mt-1">{feedback.best_solution_name || '-'}</div>
              </div>
            </div>

            <div className="rounded-lg border border-border/60 bg-secondary/20 p-4 space-y-3">
              <div>
                <div className="text-xs text-muted-foreground mb-1">结论</div>
                <p className="text-sm leading-7">{feedback.conclusion || '-'}</p>
              </div>
              <div>
                <div className="text-xs text-muted-foreground mb-2">关键特征</div>
                <div className="flex flex-wrap gap-2">
                  {effectiveFeatures.length > 0 ? (
                    effectiveFeatures.map((f) => (
                      <Badge key={f} variant="outline">
                        {f}
                      </Badge>
                    ))
                  ) : (
                    <span className="text-sm text-muted-foreground">-</span>
                  )}
                </div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground mb-1">业务解释</div>
                <p className="text-sm leading-7">{feedback.business_interpretation || '-'}</p>
              </div>
              <div>
                <div className="text-xs text-muted-foreground mb-1">下一步建议</div>
                <p className="text-sm leading-7">{feedback.next_step || '-'}</p>
              </div>
            </div>

          </CardContent>
        </Card>
      )}

      <Card className="glass p-4">
        <div className="text-sm font-semibold mb-2">方案搜索日志</div>
        <div className="h-48 overflow-y-auto font-mono text-xs bg-black/30 rounded-lg p-3 space-y-1">
          {forecastSearchLogs.length === 0 && <p className="text-muted-foreground">暂无日志</p>}
          {forecastSearchLogs.map((log) => (
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
          <div ref={searchLogsEndRef} />
        </div>
      </Card>
    </div>
  );
};

