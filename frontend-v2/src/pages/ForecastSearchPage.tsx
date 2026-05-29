import React, { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Play, Square, Loader2, AlertCircle, ChevronDown, ChevronRight, Flame } from 'lucide-react';
import { useTaskContext } from '@/context/TaskContext';
import { formatNumber } from '@/utils';

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

  useEffect(() => {
    searchLogsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [forecastSearchLogs]);

  const isSearchRunning = forecastSearchTask?.status === 'running';
  const searchMetrics = forecastSearchTask?.metrics || {};
  const leaderboard = (searchMetrics.leaderboard || []) as any[];
  const solutionLibrary = searchMetrics.solution_library || null;
  const feedback = searchMetrics.forecast_feedback || null;
  const feedbackMd = searchMetrics.forecast_feedback_md_text || '';

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

      {solutionLibrary && (
        <Card className="glass p-4">
          <div className="text-sm font-semibold mb-2">LLM 方案输出（solution_library）</div>
          <pre className="text-xs bg-black/30 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap">
            {JSON.stringify(solutionLibrary, null, 2)}
          </pre>
        </Card>
      )}

      {feedback && (
        <Card className="glass p-4">
          <div className="text-sm font-semibold mb-2">LLM 总结（forecast_feedback）</div>
          <pre className="text-xs bg-black/30 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap">
            {JSON.stringify(feedback, null, 2)}
          </pre>
          {feedbackMd && (
            <>
              <div className="text-xs text-muted-foreground mt-3 mb-1">Markdown</div>
              <pre className="text-xs bg-black/30 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap">
                {feedbackMd}
              </pre>
            </>
          )}
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

