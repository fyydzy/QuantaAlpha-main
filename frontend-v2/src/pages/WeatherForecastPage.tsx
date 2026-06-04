import React, { useCallback, useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { CloudSun, AlertCircle, Thermometer } from 'lucide-react';
import {
  getWeatherDaily,
  getWeatherFiles,
  getWeatherPreviewMeta,
  getWeatherPreviewValue,
  type WeatherDataSource,
  type WeatherDailyRow,
  type WeatherPreviewValue,
} from '@/services/api';

const PREVIEW_METHOD = `参考点 (38.04°N, 114.51°E) 取最近网格，保留全部集合成员。
集合成员说明：每个时刻有 51 套预报（number=0..50）。
展示值：在所选北京时间（日期 + 整点）筛选 valid_time_bj，计算 temp_mean_c/temp_p10_c/temp_p50_c/temp_p90_c。`;

const DAILY_METHOD = `日度计算：先按成员在每个北京日历日内对 6 小时 t2m_c 取均值，再计算 temp_mean_c/temp_p10_c/temp_p50_c/temp_p90_c。`;

export const WeatherForecastPage: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dataSource, setDataSource] = useState<WeatherDataSource | null>(null);

  const [previewFiles, setPreviewFiles] = useState<string[]>([]);
  const [dailyFiles, setDailyFiles] = useState<string[]>([]);
  const [previewFile, setPreviewFile] = useState('');
  const [dailyFile, setDailyFile] = useState('');

  const [previewDates, setPreviewDates] = useState<string[]>([]);
  const [hoursByDate, setHoursByDate] = useState<Record<string, number[]>>({});
  const [previewDate, setPreviewDate] = useState('');
  const [previewHour, setPreviewHour] = useState(0);
  const [gridLat, setGridLat] = useState<number | null>(null);
  const [gridLon, setGridLon] = useState<number | null>(null);

  const [previewValue, setPreviewValue] = useState<WeatherPreviewValue | null>(null);
  const [previewValueError, setPreviewValueError] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  const [dailyRows, setDailyRows] = useState<WeatherDailyRow[]>([]);
  const [dailyDate, setDailyDate] = useState('');
  const [dailyLoading, setDailyLoading] = useState(false);

  useEffect(() => {
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await getWeatherFiles();
        if (!res.success || !res.data) {
          setError(res.error || '加载文件列表失败');
          return;
        }
        setDataSource(res.data.dataSource);
        setPreviewFiles(res.data.previewFiles);
        setDailyFiles(res.data.dailyFiles);
        if (res.data.previewFiles.length) {
          setPreviewFile(res.data.previewFiles[res.data.previewFiles.length - 1]);
        }
        if (res.data.dailyFiles.length) {
          setDailyFile(res.data.dailyFiles[res.data.dailyFiles.length - 1]);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : '请求失败');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (!previewFile) return;
    (async () => {
      setPreviewLoading(true);
      setPreviewValueError(null);
      try {
        const res = await getWeatherPreviewMeta(previewFile);
        if (!res.success || !res.data) {
          setPreviewValueError(res.error || '加载预览元数据失败');
          return;
        }
        setPreviewDates(res.data.dates);
        setHoursByDate(res.data.hoursByDate);
        setGridLat(res.data.gridLatitude ?? null);
        setGridLon(res.data.gridLongitude ?? null);
        const firstDate = res.data.dates[0] ?? '';
        setPreviewDate(firstDate);
        const hours = res.data.hoursByDate[firstDate] ?? [];
        setPreviewHour(hours[0] ?? 0);
      } catch (e) {
        setPreviewValueError(e instanceof Error ? e.message : '请求失败');
      } finally {
        setPreviewLoading(false);
      }
    })();
  }, [previewFile]);

  const loadPreviewValue = useCallback(async () => {
    if (!previewFile || !previewDate) return;
    setPreviewLoading(true);
    setPreviewValueError(null);
    try {
      const res = await getWeatherPreviewValue(previewFile, previewDate, previewHour);
      if (!res.success || !res.data) {
        setPreviewValue(null);
        setPreviewValueError(res.error || '无数据');
        return;
      }
      setPreviewValue(res.data);
    } catch (e) {
      setPreviewValueError(e instanceof Error ? e.message : '请求失败');
    } finally {
      setPreviewLoading(false);
    }
  }, [previewFile, previewDate, previewHour]);

  useEffect(() => {
    loadPreviewValue();
  }, [loadPreviewValue]);

  useEffect(() => {
    if (!dailyFile) return;
    (async () => {
      setDailyLoading(true);
      setError(null);
      try {
        const res = await getWeatherDaily(dailyFile);
        if (!res.success || !res.data) {
          setError(res.error || '加载日度 CSV 失败');
          return;
        }
        setDailyRows(res.data.rows);
        if (res.data.rows.length) {
          setDailyDate(res.data.rows[0].dateBj);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : '请求失败');
      } finally {
        setDailyLoading(false);
      }
    })();
  }, [dailyFile]);

  const previewHours = hoursByDate[previewDate] ?? [];
  const selectedDaily = dailyRows.find((r) => r.dateBj === dailyDate);

  return (
    <div className="space-y-6 animate-fade-in-up">
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-3">
          <CloudSun className="h-8 w-8 text-sky-500" />
          天气预测
        </h1>
        <p className="text-muted-foreground mt-1">查看 ECMWF 季节预报导出的 6 小时与日度气温 CSV</p>
      </div>

      {dataSource && (
        <Card className="glass border-sky-500/30">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">数据源说明</CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground space-y-2">
            <p>
              <strong>数据集</strong>：{' '}
              <a
                href={dataSource.url}
                target="_blank"
                rel="noreferrer"
                className="text-sky-400 hover:underline"
              >
                {dataSource.title}
              </a>
              （<code className="text-xs">{dataSource.dataset}</code>）
            </p>
            <p>
              <strong>变量与时效</strong>：{dataSource.variable}，{dataSource.step} 步长，{dataSource.horizon}
            </p>
            <p>
              <strong>集合成员</strong>：每个时刻提供 51 套预报（number=0..50）
            </p>
            <p>
              <strong>起报日</strong>：{dataSource.initDate}；<strong>区域</strong>：{dataSource.area}
            </p>
            <p>{dataSource.gridNote}</p>
          </CardContent>
        </Card>
      )}

      <Card className="glass border-sky-500/20">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">温度指标说明</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground space-y-1">
          <p><strong>temp_mean_c</strong>：集合平均温度。51 套预报温度的平均值，代表“平均预测结果”。</p>
          <p><strong>temp_p10_c</strong>：10% 分位温度。偏冷情景，表示大约有 10% 的成员比它更低。</p>
          <p><strong>temp_p50_c</strong>：50% 分位温度。中位数情景，也就是比较中性的预测。</p>
          <p><strong>temp_p90_c</strong>：90% 分位温度。偏暖情景，表示大约有 90% 的成员比它更低。</p>
        </CardContent>
      </Card>

      {error && (
        <Card className="glass border-destructive/50">
          <CardContent className="p-4 flex items-center gap-3">
            <AlertCircle className="h-5 w-5 text-destructive shrink-0" />
            <span className="text-sm text-destructive">{error}</span>
          </CardContent>
        </Card>
      )}

      {loading ? (
        <p className="text-muted-foreground">加载中…</p>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* 6 小时气温预测 */}
          <Card className="glass">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Thermometer className="h-5 w-5 text-orange-400" />
                6 小时气温预测
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <label className="text-sm block">
                <span className="text-muted-foreground">CSV 文件</span>
                <select
                  className="mt-1 w-full rounded-lg bg-secondary/50 border border-border px-3 py-2"
                  value={previewFile}
                  onChange={(e) => setPreviewFile(e.target.value)}
                  disabled={!previewFiles.length}
                >
                  {previewFiles.length === 0 && <option value="">暂无 *_preview.csv</option>}
                  {previewFiles.map((f) => (
                    <option key={f} value={f}>
                      {f}
                    </option>
                  ))}
                </select>
              </label>

              <div className="grid grid-cols-2 gap-3">
                <label className="text-sm">
                  <span className="text-muted-foreground">北京时间 · 日期</span>
                  <select
                    className="mt-1 w-full rounded-lg bg-secondary/50 border border-border px-3 py-2"
                    value={previewDate}
                    onChange={(e) => {
                      const d = e.target.value;
                      setPreviewDate(d);
                      const h = hoursByDate[d]?.[0] ?? 0;
                      setPreviewHour(h);
                    }}
                    disabled={!previewDates.length}
                  >
                    {previewDates.map((d) => (
                      <option key={d} value={d}>
                        {d}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="text-sm">
                  <span className="text-muted-foreground">北京时间 · 整点</span>
                  <select
                    className="mt-1 w-full rounded-lg bg-secondary/50 border border-border px-3 py-2"
                    value={previewHour}
                    onChange={(e) => setPreviewHour(Number(e.target.value))}
                    disabled={!previewHours.length}
                  >
                    {previewHours.map((h) => (
                      <option key={h} value={h}>
                        {String(h).padStart(2, '0')}:00
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              {gridLat != null && gridLon != null && (
                <p className="text-xs text-muted-foreground">
                  最近网格：{gridLat}°N, {gridLon}°E
                </p>
              )}

              <div className="rounded-xl bg-secondary/40 p-6 text-center">
                {previewLoading ? (
                  <span className="text-muted-foreground">查询中…</span>
                ) : previewValueError ? (
                  <span className="text-destructive text-sm">{previewValueError}</span>
                ) : previewValue ? (
                  <>
                    <div className="grid grid-cols-2 gap-3 text-left">
                      <div className="rounded-lg bg-background/50 p-3">
                        <p className="text-xs text-muted-foreground">temp_mean_c</p>
                        <p className="text-xl font-semibold text-orange-400">{previewValue.temperatureC} °C</p>
                      </div>
                      <div className="rounded-lg bg-background/50 p-3">
                        <p className="text-xs text-muted-foreground">temp_p10_c</p>
                        <p className="text-xl font-semibold">{previewValue.tempP10C ?? previewValue.tempMinC} °C</p>
                      </div>
                      <div className="rounded-lg bg-background/50 p-3">
                        <p className="text-xs text-muted-foreground">temp_p50_c</p>
                        <p className="text-xl font-semibold">{previewValue.tempP50C ?? previewValue.temperatureC} °C</p>
                      </div>
                      <div className="rounded-lg bg-background/50 p-3">
                        <p className="text-xs text-muted-foreground">temp_p90_c</p>
                        <p className="text-xl font-semibold">{previewValue.tempP90C ?? previewValue.tempMaxC} °C</p>
                      </div>
                    </div>
                  </>
                ) : (
                  <span className="text-muted-foreground">—</span>
                )}
              </div>

              <details className="text-xs text-muted-foreground">
                <summary className="cursor-pointer text-foreground/80">获取与计算方法</summary>
                <p className="mt-2 whitespace-pre-line leading-relaxed">{PREVIEW_METHOD}</p>
              </details>
            </CardContent>
          </Card>

          {/* 日度 */}
          <Card className="glass">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Thermometer className="h-5 w-5 text-sky-400" />
                日度气温
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <label className="text-sm block">
                <span className="text-muted-foreground">CSV 文件</span>
                <select
                  className="mt-1 w-full rounded-lg bg-secondary/50 border border-border px-3 py-2"
                  value={dailyFile}
                  onChange={(e) => setDailyFile(e.target.value)}
                  disabled={!dailyFiles.length}
                >
                  {dailyFiles.length === 0 && (
                    <option value="">暂无 hebei_ecmwf_s5_daily_temperature_*.csv</option>
                  )}
                  {dailyFiles.map((f) => (
                    <option key={f} value={f}>
                      {f}
                    </option>
                  ))}
                </select>
              </label>

              <label className="text-sm block">
                <span className="text-muted-foreground">北京时间 · 日期</span>
                <select
                  className="mt-1 w-full rounded-lg bg-secondary/50 border border-border px-3 py-2"
                  value={dailyDate}
                  onChange={(e) => setDailyDate(e.target.value)}
                  disabled={!dailyRows.length || dailyLoading}
                >
                  {dailyRows.map((r) => (
                    <option key={r.dateBj} value={r.dateBj}>
                      {r.dateBj}
                    </option>
                  ))}
                </select>
              </label>

              <div className="rounded-xl bg-secondary/40 p-6 text-center">
                {dailyLoading ? (
                  <span className="text-muted-foreground">加载中…</span>
                ) : selectedDaily ? (
                  <>
                    <div className="grid grid-cols-2 gap-3 text-left">
                      <div className="rounded-lg bg-background/50 p-3">
                        <p className="text-xs text-muted-foreground">temp_mean_c</p>
                        <p className="text-xl font-semibold text-sky-400">{selectedDaily.tempMeanC} °C</p>
                      </div>
                      <div className="rounded-lg bg-background/50 p-3">
                        <p className="text-xs text-muted-foreground">temp_p10_c</p>
                        <p className="text-xl font-semibold">{selectedDaily.tempP10C ?? selectedDaily.tempMeanC} °C</p>
                      </div>
                      <div className="rounded-lg bg-background/50 p-3">
                        <p className="text-xs text-muted-foreground">temp_p50_c</p>
                        <p className="text-xl font-semibold">{selectedDaily.tempP50C ?? selectedDaily.tempMeanC} °C</p>
                      </div>
                      <div className="rounded-lg bg-background/50 p-3">
                        <p className="text-xs text-muted-foreground">temp_p90_c</p>
                        <p className="text-xl font-semibold">{selectedDaily.tempP90C ?? selectedDaily.tempMeanC} °C</p>
                      </div>
                    </div>
                  </>
                ) : (
                  <span className="text-muted-foreground">—</span>
                )}
              </div>

              <details className="text-xs text-muted-foreground">
                <summary className="cursor-pointer text-foreground/80">计算方法</summary>
                <p className="mt-2 whitespace-pre-line leading-relaxed">{DAILY_METHOD}</p>
              </details>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
};
