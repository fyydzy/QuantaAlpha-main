/**
 * API client — 预测智能体、天气预测与系统配置。
 */

import type { ApiResponse, ForecastAgentTask, WsMessage } from '@/types';

const BASE = '';

async function request<T = any>(path: string, options: RequestInit = {}): Promise<ApiResponse<T>> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers as any) },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API Error ${res.status}: ${text}`);
  }
  return res.json();
}

// ========================== Weather ==========================

export interface WeatherDataSource {
  title: string;
  dataset: string;
  url: string;
  variable: string;
  step: string;
  horizon: string;
  initDate: string;
  area: string;
  gridNote: string;
}

export interface WeatherFilesResponse {
  previewFiles: string[];
  dailyFiles: string[];
  dataSource: WeatherDataSource;
  weatherDir: string;
}

export interface WeatherPreviewMeta {
  file: string;
  dates: string[];
  hoursByDate: Record<string, number[]>;
  gridLatitude: number | null;
  gridLongitude: number | null;
  rowCount: number;
}

export interface WeatherPreviewValue {
  dateBj: string;
  hourBj: number;
  temperatureC: number;
  aggMode: 'mean' | 'member';
  member: number | null;
  memberCount: number;
  sampleCount: number;
  tempMinC: number;
  tempMaxC: number;
  tempP10C?: number;
  tempP50C?: number;
  tempP90C?: number;
  validTimeUtc?: string | null;
  validTimeBj?: string | null;
}

export interface WeatherDailyRow {
  dateBj: string;
  tempMeanC: number;
  tempP10C?: number;
  tempP50C?: number;
  tempP90C?: number;
}

export async function getWeatherFiles() {
  return request<WeatherFilesResponse>('/api/v1/weather/files');
}

export async function getWeatherPreviewMeta(file: string) {
  return request<WeatherPreviewMeta>(`/api/v1/weather/preview/meta?file=${encodeURIComponent(file)}`);
}

export async function getWeatherPreviewValue(file: string, dateBj: string, hourBj: number) {
  const q = new URLSearchParams({ file, date_bj: dateBj, hour_bj: String(hourBj) });
  return request<WeatherPreviewValue>(`/api/v1/weather/preview/value?${q}`);
}

export async function getWeatherDaily(file: string) {
  return request<{ file: string; rows: WeatherDailyRow[]; tempColumn: string }>(
    `/api/v1/weather/daily?file=${encodeURIComponent(file)}`
  );
}

// ========================== Forecast ==========================

export interface ForecastQaParams {
  outputDir: string;
  model: string;
  query: string;
  selectedFeatures?: string[];
}

export interface ForecastQaResponse {
  answer: string;
  modelUsed: string;
  rowsUsed: number;
  dataMode: 'test_only' | 'pred_only';
  featureColsUsed: string[];
}

export interface ForecastAgentStartParams {
  query: string;
  province?: string;
  candidateModels?: string;
  outputDir?: string;
  contextLen?: number;
  maxFeatureCount?: number;
  importanceTopK?: number;
  requiredFeatures?: string[];
  qaQuery?: string;
}

export interface ForecastAgentContinueParams {
  checkpoint?: string;
  approved?: boolean;
  overrides?: Record<string, any>;
  message?: string;
}

export async function askForecastQa(params: ForecastQaParams) {
  return request<ForecastQaResponse>('/api/v1/forecast/qa', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function startForecastAgent(params: ForecastAgentStartParams) {
  return request<{ taskId: string; task: ForecastAgentTask }>('/api/v1/forecast/agent/start', {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

export async function getForecastAgentStatus(taskId: string) {
  return request<{ task: ForecastAgentTask }>(`/api/v1/forecast/agent/${taskId}`);
}

export async function cancelForecastAgent(taskId: string) {
  return request(`/api/v1/forecast/agent/${taskId}`, { method: 'DELETE' });
}

export async function continueForecastAgent(taskId: string, params: ForecastAgentContinueParams) {
  return request<{ task: ForecastAgentTask }>(`/api/v1/forecast/agent/${taskId}/continue`, {
    method: 'POST',
    body: JSON.stringify(params),
  });
}

// ========================== System ==========================

export async function getSystemConfig() {
  return request<{ env: Record<string, string>; forecastConfig: string }>('/api/v1/system/config');
}

export async function updateSystemConfig(update: Record<string, string>) {
  return request('/api/v1/system/config', {
    method: 'PUT',
    body: JSON.stringify(update),
  });
}

export async function healthCheck() {
  return request<{ status: string; timestamp: string }>('/api/health');
}

// ========================== WebSocket ==========================

export type WsCallback = (msg: WsMessage) => void;

export function connectTaskWs(
  taskId: string,
  onMessage: WsCallback,
  onClose?: () => void,
  onError?: (e: Event) => void
): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/mining/${taskId}`;
  const ws = new WebSocket(wsUrl);

  ws.onmessage = (event) => {
    try {
      onMessage(JSON.parse(event.data) as WsMessage);
    } catch {
      console.warn('[WS] Failed to parse message:', event.data);
    }
  };

  ws.onclose = () => onClose?.();
  ws.onerror = (e) => onError?.(e);

  const heartbeat = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send('ping');
    } else {
      clearInterval(heartbeat);
    }
  }, 30000);

  return ws;
}

/** @deprecated 保留旧名以兼容历史调用 */
export const connectMiningWs = connectTaskWs;
