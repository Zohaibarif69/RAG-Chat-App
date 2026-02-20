import { useState, useCallback } from 'react';

export interface StreamMetadata {
  sources: Array<{ source: string; chunk: number; relevance: number }>;
  confidence: number;
  hallucination_risk: 'low' | 'medium' | 'high';
  verified: boolean;
}

export interface StreamingResponse {
  metadata: StreamMetadata | null;
  answer: string;
  isComplete: boolean;
  totalLatency: number | null;
  error: string | null;
}

const useStreamingQuery = () => {
  const [streaming, setStreaming] = useState(false);

  const query = useCallback(
    async (
      question: string,
      sessionId: string,
      topK: number = 5,
      onUpdate: (response: StreamingResponse) => void
    ) => {
      setStreaming(true);
      const API_BASE = 'http://127.0.0.1:8000';

      try {
        const response = await fetch(`${API_BASE}/query/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question,
            session_id: sessionId,
            top_k: topK,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error('Response body not readable');
        }

        const decoder = new TextDecoder();
        let buffer = '';
        let metadata: StreamMetadata | null = null;
        let answer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // Keep incomplete line in buffer

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));

                if (data.type === 'metadata') {
                  metadata = {
                    sources: data.sources || [],
                    confidence: data.confidence || 0,
                    hallucination_risk: data.hallucination_risk || 'low',
                    verified: data.verified || false,
                  };
                  onUpdate({
                    metadata,
                    answer: '',
                    isComplete: false,
                    totalLatency: null,
                    error: null,
                  });
                } else if (data.type === 'token') {
                  answer += data.content || '';
                  onUpdate({
                    metadata,
                    answer,
                    isComplete: false,
                    totalLatency: null,
                    error: null,
                  });
                } else if (data.type === 'complete') {
                  onUpdate({
                    metadata,
                    answer,
                    isComplete: true,
                    totalLatency: data.total_latency_ms || null,
                    error: null,
                  });
                }
              } catch (e) {
                console.error('Failed to parse SSE data:', e, line);
              }
            }
          }
        }

        // Process any remaining buffer
        if (buffer.startsWith('data: ')) {
          try {
            const data = JSON.parse(buffer.slice(6));
            if (data.type === 'complete') {
              onUpdate({
                metadata,
                answer,
                isComplete: true,
                totalLatency: data.total_latency_ms || null,
                error: null,
              });
            }
          } catch (e) {
            console.error('Failed to parse final SSE data:', e);
          }
        }
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : 'Unknown error';
        onUpdate({
          metadata: null,
          answer: '',
          isComplete: true,
          totalLatency: null,
          error: errorMessage,
        });
      } finally {
        setStreaming(false);
      }
    },
    []
  );

  return { query, streaming };
};

export default useStreamingQuery;
