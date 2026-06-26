/**
 * TaskContext — 预测智能体任务状态（WebSocket + 轮询）。
 */

import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';
import type { LogEntry, WsMessage } from '@/types';
import { generateId } from '@/utils';
import {
  startForecastAgent as apiStartForecastAgent,
  getForecastAgentStatus,
  cancelForecastAgent as apiCancelForecastAgent,
  continueForecastAgent as apiContinueForecastAgent,
  connectTaskWs,
  healthCheck,
} from '@/services/api';
import type {
  ForecastAgentContinueParams,
  ForecastAgentStartParams,
} from '@/services/api';

export interface ForecastAgentMessage {
  role: string;
  stage: string;
  messageType: string;
  text: string;
  payload?: Record<string, any>;
  needConfirm?: boolean;
  timestamp: string;
}

export interface ForecastAgentTaskState {
  taskId: string;
  status: string;
  progress: {
    phase: string;
    progress: number;
    message: string;
    timestamp: string;
    currentRound?: number;
    totalRounds?: number;
  };
  logs: LogEntry[];
  metrics: Record<string, any>;
  config: Record<string, any>;
  createdAt: string;
  updatedAt: string;
  lastCheckpoint?: Record<string, any>;
  checkpointPathDisplay?: string;
  auditPathDisplay?: string;
}

interface TaskContextValue {
  backendAvailable: boolean | null;
  forecastAgentTask: ForecastAgentTaskState | null;
  forecastAgentLogs: LogEntry[];
  forecastAgentMessages: ForecastAgentMessage[];
  startForecastAgentTask: (params: ForecastAgentStartParams) => Promise<void>;
  continueForecastAgentTask: (params: ForecastAgentContinueParams) => Promise<void>;
  stopForecastAgentTask: () => void;
}

const TaskContext = createContext<TaskContextValue | null>(null);

export const TaskProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(null);
  const [forecastAgentTask, setForecastAgentTask] = useState<ForecastAgentTaskState | null>(null);
  const [forecastAgentLogs, setForecastAgentLogs] = useState<LogEntry[]>([]);
  const [forecastAgentMessages, setForecastAgentMessages] = useState<ForecastAgentMessage[]>([]);
  const forecastAgentWsRef = useRef<WebSocket | null>(null);
  const forecastAgentPollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    healthCheck()
      .then(() => setBackendAvailable(true))
      .catch(() => setBackendAvailable(false));
  }, []);

  const handleForecastAgentWsMessage = useCallback((msg: WsMessage) => {
    switch (msg.type) {
      case 'progress':
        setForecastAgentTask((prev) => {
          if (!prev) return prev;
          return { ...prev, progress: msg.data, updatedAt: new Date().toISOString() };
        });
        break;
      case 'log':
        setForecastAgentLogs((l) => [...l.slice(-499), msg.data as LogEntry]);
        break;
      case 'metrics':
        if (msg.data?.agent_message) {
          const nextMsg = msg.data.agent_message as ForecastAgentMessage;
          setForecastAgentMessages((prev) => {
            const last = prev[prev.length - 1];
            if (
              last &&
              last.timestamp === nextMsg.timestamp &&
              last.stage === nextMsg.stage &&
              last.text === nextMsg.text
            ) {
              return prev;
            }
            return [...prev, nextMsg];
          });
        }
        setForecastAgentTask((prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            metrics: { ...(prev.metrics || {}), ...(msg.data || {}) },
            updatedAt: new Date().toISOString(),
          };
        });
        break;
      case 'result':
        setForecastAgentTask((prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            status: msg.data.status === 'completed' ? 'completed' : msg.data.status || 'failed',
            metrics: msg.data.metrics || prev.metrics,
            updatedAt: new Date().toISOString(),
          };
        });
        break;
      case 'error':
        setForecastAgentTask((prev) => (prev ? { ...prev, status: 'failed', updatedAt: new Date().toISOString() } : prev));
        setForecastAgentLogs((l) => [
          ...l.slice(-498),
          {
            id: generateId(),
            timestamp: new Date().toISOString(),
            level: 'error',
            message: msg.data?.error || 'Agent task failed',
          },
        ]);
        break;
    }
  }, []);

  const attachPolling = useCallback(
    (taskId: string) => {
      if (forecastAgentPollingRef.current) {
        clearInterval(forecastAgentPollingRef.current);
      }
      forecastAgentPollingRef.current = setInterval(async () => {
        try {
          const r = await getForecastAgentStatus(taskId);
          if (!r.data?.task) return;
          const t = r.data.task as unknown as ForecastAgentTaskState;
          setForecastAgentTask((prev) => {
            if (!prev) return t;
            return {
              ...prev,
              status: t.status,
              progress: t.progress || prev.progress,
              metrics: t.metrics && Object.keys(t.metrics).length > 0 ? t.metrics : prev.metrics,
              lastCheckpoint: t.lastCheckpoint ?? prev.lastCheckpoint,
              checkpointPathDisplay: t.checkpointPathDisplay ?? prev.checkpointPathDisplay,
              auditPathDisplay: t.auditPathDisplay ?? prev.auditPathDisplay,
              updatedAt: t.updatedAt,
            };
          });
          if (t.status === 'completed' || t.status === 'failed' || t.status === 'cancelled') {
            setForecastAgentTask(t);
            if (t.logs?.length) setForecastAgentLogs(t.logs.slice(-500));
            const finalMessages = (t.metrics?.agent_messages as ForecastAgentMessage[] | undefined) || [];
            if (finalMessages.length > 0) setForecastAgentMessages(finalMessages);
            clearInterval(forecastAgentPollingRef.current!);
            forecastAgentPollingRef.current = null;
          }
        } catch {
          // ignore polling errors
        }
      }, 5000);
    },
    []
  );

  const connectWs = useCallback(
    (taskId: string) => {
      forecastAgentWsRef.current?.close();
      forecastAgentWsRef.current = connectTaskWs(taskId, handleForecastAgentWsMessage, () => {
        getForecastAgentStatus(taskId).then((r) => {
          if (r.data?.task) setForecastAgentTask(r.data.task as unknown as ForecastAgentTaskState);
        });
      });
    },
    [handleForecastAgentWsMessage]
  );

  const startForecastAgentTask = useCallback(
    async (params: ForecastAgentStartParams) => {
      if (!backendAvailable) {
        throw new Error('后端未连接，请先启动 FastAPI 服务');
      }
      setForecastAgentLogs([]);
      setForecastAgentMessages([]);
      const resp = await apiStartForecastAgent(params);
      if (!resp.success || !resp.data) throw new Error(resp.error || '启动失败');

      const taskData = resp.data.task as unknown as ForecastAgentTaskState;
      setForecastAgentTask(taskData);
      const initialMessages = (taskData.metrics?.agent_messages as ForecastAgentMessage[] | undefined) || [];
      if (initialMessages.length > 0) setForecastAgentMessages(initialMessages);

      connectWs(resp.data.taskId);
      attachPolling(resp.data.taskId);
    },
    [backendAvailable, attachPolling, connectWs]
  );

  const stopForecastAgentTask = useCallback(async () => {
    if (!forecastAgentTask) return;
    forecastAgentWsRef.current?.close();
    forecastAgentWsRef.current = null;
    if (forecastAgentPollingRef.current) {
      clearInterval(forecastAgentPollingRef.current);
      forecastAgentPollingRef.current = null;
    }
    try {
      await apiCancelForecastAgent(forecastAgentTask.taskId);
    } catch {
      // ignore
    }
    setForecastAgentTask((prev) => (prev ? { ...prev, status: 'cancelled' } : prev));
  }, [forecastAgentTask]);

  const continueForecastAgentTask = useCallback(
    async (params: ForecastAgentContinueParams) => {
      if (!forecastAgentTask) return;
      const resp = await apiContinueForecastAgent(forecastAgentTask.taskId, params);
      if (!resp.success) throw new Error(resp.error || '继续失败');
    },
    [forecastAgentTask]
  );

  const value: TaskContextValue = {
    backendAvailable,
    forecastAgentTask,
    forecastAgentLogs,
    forecastAgentMessages,
    startForecastAgentTask,
    continueForecastAgentTask,
    stopForecastAgentTask,
  };

  return <TaskContext.Provider value={value}>{children}</TaskContext.Provider>;
};

export function useTaskContext(): TaskContextValue {
  const ctx = useContext(TaskContext);
  if (!ctx) throw new Error('useTaskContext must be used inside <TaskProvider>');
  return ctx;
}
