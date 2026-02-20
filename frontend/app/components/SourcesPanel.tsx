'use client';

import React from 'react';

interface Source {
  source: string;
  chunk: number;
  relevance: number;
}

interface SourcesPanelProps {
  sources: Source[];
  isExpanded: boolean;
  onToggle: () => void;
}

const SourcesPanel: React.FC<SourcesPanelProps> = ({
  sources,
  isExpanded,
  onToggle,
}) => {
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
        <span>📚 Sources ({sources.length})</span>
      </button>

      {isExpanded && (
        <div className="pl-4 space-y-2 mt-2 max-h-64 overflow-y-auto">
          {sources.map((source, idx) => (
            <div
              key={idx}
              className="p-2.5 bg-white rounded border border-gray-300 hover:border-gray-400 transition"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-gray-800 text-xs truncate">
                    {source.source}
                  </p>
                  <p className="text-gray-600 text-xs mt-1">
                    Chunk #{source.chunk}
                  </p>
                </div>
                <div className="shrink-0 text-right">
                  <div className="px-2 py-1 bg-blue-50 rounded text-xs font-semibold text-blue-700">
                    {(source.relevance * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default SourcesPanel;
