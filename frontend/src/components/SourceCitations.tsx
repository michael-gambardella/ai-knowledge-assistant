/**
 * SourceCitations — renders the retrieved chunks that grounded an answer.
 *
 * Each source shows:
 *   - Filename and chunk index (which part of the document)
 *   - Relevance score as a coloured badge (green ≥ 0.7, yellow ≥ 0.4, red below)
 *   - The chunk text, collapsed by default and expandable on click
 *
 * WHY SHOW THE CHUNKS:
 *   Transparency is a core principle of RAG. Users should be able to verify
 *   that the answer is grounded in the actual document text, not hallucinated.
 *   Collapsing by default keeps the UI clean while still making verification easy.
 */

import { useState } from 'react'
import { type SourceChunk } from '../api'

interface Props {
  sources: SourceChunk[]
}

function relevanceBadgeClass(score: number): string {
  if (score >= 0.7) return 'bg-green-100 text-green-700'
  if (score >= 0.4) return 'bg-yellow-100 text-yellow-700'
  return 'bg-red-100 text-red-600'
}

function SourceItem({ source, index }: { source: SourceChunk; index: number }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-2 bg-white hover:bg-gray-50 transition-colors text-left"
      >
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-xs font-semibold text-gray-500 flex-shrink-0">
            [{index + 1}]
          </span>
          <span className="text-xs font-medium text-gray-700 truncate">{source.filename}</span>
          <span className="text-xs text-gray-400 flex-shrink-0">chunk {source.chunk_index}</span>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0 ml-2">
          <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${relevanceBadgeClass(source.relevance_score)}`}>
            {(source.relevance_score * 100).toFixed(0)}%
          </span>
          <span className="text-gray-400 text-xs">{expanded ? '▲' : '▼'}</span>
        </div>
      </button>

      {expanded && (
        <div className="px-3 py-2 bg-gray-50 border-t border-gray-200">
          <p className="text-xs text-gray-600 whitespace-pre-wrap leading-relaxed">
            {source.content}
          </p>
        </div>
      )}
    </div>
  )
}

export function SourceCitations({ sources }: Props) {
  if (sources.length === 0) return null

  return (
    <div className="mt-3">
      <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-2">
        Sources ({sources.length})
      </p>
      <div className="flex flex-col gap-1.5">
        {sources.map((source, i) => (
          <SourceItem key={`${source.document_id}-${source.chunk_index}`} source={source} index={i} />
        ))}
      </div>
    </div>
  )
}
