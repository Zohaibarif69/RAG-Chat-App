'use client';

import React, { useState } from 'react';
import { StreamMetadata } from '../hooks/useStreamingQuery';
import SourcesPanel from './SourcesPanel';
import ChunksPreview from './ChunksPreview';

interface ChatMessageProps {
  type: 'user' | 'assistant';
  content: string;
  metadata?: StreamMetadata;
  isStreaming?: boolean;
  retrievedChunks?: Array<{ source: string; content: string; relevance: number }>;
}

const ChatMessage: React.FC<ChatMessageProps> = ({
  type,
  content,
  metadata,
  isStreaming = false,
  retrievedChunks = [],
}) => {
  const [showSources, setShowSources] = useState(false);
  const [showChunks, setShowChunks] = useState(false);

  const getConfidenceBadgeColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-100 text-green-800';
    if (confidence >= 0.6) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const getRiskColor = (risk: string) => {
    if (risk === 'low') return 'bg-green-50 border-green-200 text-green-700';
    if (risk === 'medium') return 'bg-yellow-50 border-yellow-200 text-yellow-700';
    return 'bg-red-50 border-red-200 text-red-700';
  };

  return (
    <div
      className={`flex ${type === 'user' ? 'justify-end' : 'justify-start'}`}
    >
      <div
        className={`max-w-2xl px-4 py-3 rounded-lg ${
          type === 'user'
            ? 'bg-blue-600 text-white'
            : 'bg-gray-100 text-gray-800'
        }`}
      >
        {/* Message Content */}
        <p className="text-sm whitespace-pre-wrap wrap-break-word">
          {content}
          {isStreaming && <span className="animate-pulse">▌</span>}
        </p>

        {/* Metadata Section (for assistant messages) */}
        {type === 'assistant' && metadata && (
          <div className="mt-4 pt-3 border-t border-gray-300 space-y-3">
            {/* Confidence Badge & Quality Indicators Row */}
            <div className="flex flex-wrap gap-2 items-center">
              {/* Confidence Score */}
              <div
                className={`px-3 py-1 rounded-full text-xs font-semibold ${getConfidenceBadgeColor(
                  metadata.confidence
                )}`}
              >
                Confidence: {(metadata.confidence * 100).toFixed(0)}%
              </div>

              {/* Verification Status */}
              {metadata.verified && (
                <div className="px-3 py-1 rounded-full text-xs font-semibold bg-blue-100 text-blue-800 flex items-center gap-1">
                  ✓ Verified
                </div>
              )}

              {/* Hallucination Risk */}
              <div
                className={`px-3 py-1 rounded-full text-xs font-semibold border ${getRiskColor(
                  metadata.hallucination_risk
                )}`}
              >
                Risk: {metadata.hallucination_risk}
              </div>
            </div>

            {/* Sources Section */}
            {metadata.sources && metadata.sources.length > 0 && (
              <SourcesPanel
                sources={metadata.sources}
                isExpanded={showSources}
                onToggle={() => setShowSources(!showSources)}
              />
            )}

            {/* Retrieved Chunks Preview */}
            {retrievedChunks.length > 0 && (
              <ChunksPreview
                chunks={retrievedChunks}
                isExpanded={showChunks}
                onToggle={() => setShowChunks(!showChunks)}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
