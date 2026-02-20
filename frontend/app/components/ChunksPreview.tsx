'use client';

import React from 'react';

interface Chunk {
  source: string;
  content: string;
  relevance: number;
}

interface ChunksPreviewProps {
  chunks: Chunk[];
  isExpanded: boolean;
  onToggle: () => void;
}

const ChunksPreview: React.FC<ChunksPreviewProps> = ({
  chunks,
  isExpanded,
  onToggle,
}) => {
  const truncateContent = (content: string, maxLength: number = 200) => {
    return content.length > maxLength
      ? content.substring(0, maxLength) + '...'
      : content;
  };

  return (
    <div className="space-y-2">
      <button
        onClick={onToggle}
        className="flex items-center gap-2 text-xs font-semibold text-gray-700 hover:text-gray-900 transition"
      >
        <span
          className={`transform transition-transform ${
            isExpanded ? 'rotate-90' : ''
          }`}
        >
          ▶
        </span>
        <span>📄 Retrieved Chunks ({chunks.length})</span>
      </button>

      {isExpanded && (
        <div className="pl-4 space-y-2 mt-2 max-h-72 overflow-y-auto">
          {chunks.map((chunk, idx) => (
            <div
              key={idx}
              className="p-3 bg-white rounded border border-gray-300 hover:border-blue-300 transition"
            >
              <div className="flex items-start justify-between gap-2 mb-2">
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-gray-800 text-xs">
                    {chunk.source}
                  </p>
                </div>
                <div className="shrink-0">
                  <span className="px-2 py-0.5 bg-green-50 rounded text-xs font-semibold text-green-700">
                    {(chunk.relevance * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <p className="text-xs text-gray-600 leading-relaxed font-mono bg-gray-50 p-2 rounded border border-gray-200">
                {truncateContent(chunk.content)}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ChunksPreview;
