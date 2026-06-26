export type TaskStatus = 'idle' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface LogEntry {
  id: string;
  timestamp: string;
  level: 'info' | 'warning' | 'error' | 'success';
  message: string;
}

export interface ExecutionProgress {
  phase: string;
  currentRound: number;
  totalRounds: number;
  progress: number;
  message: string;
  timestamp: string;
}

export interface ForecastAgentTask {
  taskId: string;
  status: TaskStatus;
  progress: ExecutionProgress;
  metrics: Record<string, any>;
  config: Record<string, any>;
  logs: LogEntry[];
  createdAt: string;
  updatedAt: string;
  flowState?: Record<string, any>;
  lastCheckpoint?: Record<string, any>;
  checkpointPath?: string;
  checkpointPathDisplay?: string;
  auditPath?: string;
  auditPathDisplay?: string;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export type WsMessageType = 'progress' | 'metrics' | 'log' | 'result' | 'error' | 'heartbeat';

export interface WsMessage {
  type: WsMessageType;
  taskId: string;
  data: any;
  timestamp: string;
}
