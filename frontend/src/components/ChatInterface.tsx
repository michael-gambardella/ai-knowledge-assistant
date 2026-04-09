/**
 * ChatInterface — question input and answer display with latency metrics.
 *
 * Keeps a history of (question, response) pairs in local state so the user
 * can scroll back through previous answers. Each entry shows:
 *   - The question
 *   - The answer text
 *   - Latency badges: retrieval_ms, generation_ms, total_ms
 *   - Mean relevance score badge
 *   - Expandable source citations
 */

import { useState, useRef, useEffect } from 'react'
import { queryKnowledgeBase, type QueryResponse } from '../api'
import { SourceCitations } from './SourceCitations'

interface HistoryEntry {
  question: string
  response: QueryResponse
}

interface Props {
  hasDocuments: boolean
}

function LatencyBadge({ label, value }: { label: string; value: number }) {
  return (
    <span className="inline-flex items-center gap-1 text-xs bg-gray-100 text-gray-500 rounded px-2 py-0.5">
      <span className="font-medium text-gray-600">{label}</span>
      {value.toFixed(0)}ms
    </span>
  )
}

function RelevanceBadge({ score }: { score: number }) {
  const pct = (score * 100).toFixed(0)
  const cls = score >= 0.7
    ? 'bg-green-100 text-green-700'
    : score >= 0.4
    ? 'bg-yellow-100 text-yellow-700'
    : 'bg-red-100 text-red-600'
  return (
    <span className={`inline-flex items-center gap-1 text-xs rounded px-2 py-0.5 ${cls}`}>
      <span className="font-medium">relevance</span> {pct}%
    </span>
  )
}

export function ChatInterface({ hasDocuments }: Props) {
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to latest answer
  useEffect(() => {
    if (history.length > 0) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [history])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const q = question.trim()
    if (!q || loading) return

    setLoading(true)
    setError(null)
    setQuestion('')

    try {
      const response = await queryKnowledgeBase(q)
      setHistory((prev) => [...prev, { question: q, response }])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Query failed')
      setQuestion(q) // restore so user can retry
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Answer history */}
      <div className="flex-1 overflow-y-auto flex flex-col gap-6 pb-4">
        {history.length === 0 && !loading && (
          <div className="flex-1 flex items-center justify-center text-center py-16">
            {hasDocuments ? (
              <div>
                <p className="text-4xl mb-3">💬</p>
                <p className="text-gray-500 text-sm">Ask a question about your documents.</p>
              </div>
            ) : (
              <div>
                <p className="text-4xl mb-3">📭</p>
                <p className="text-gray-500 text-sm">Upload a document on the left to get started.</p>
              </div>
            )}
          </div>
        )}

        {history.map((entry, i) => (
          <div key={i} className="flex flex-col gap-3">
            {/* Question bubble */}
            <div className="self-end max-w-[80%] bg-indigo-600 text-white rounded-2xl rounded-tr-sm px-4 py-2.5">
              <p className="text-sm">{entry.question}</p>
            </div>

            {/* Answer card */}
            <div className="self-start max-w-full bg-white border border-gray-200 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
              <p className="text-sm text-gray-800 whitespace-pre-wrap leading-relaxed">
                {entry.response.answer}
              </p>

              {/* Metrics row */}
              <div className="flex flex-wrap gap-1.5 mt-3">
                <LatencyBadge label="retrieval" value={entry.response.retrieval_ms} />
                <LatencyBadge label="generation" value={entry.response.generation_ms} />
                <LatencyBadge label="total" value={entry.response.total_ms} />
                <RelevanceBadge score={entry.response.mean_relevance_score} />
              </div>

              <SourceCitations sources={entry.response.sources} />
            </div>
          </div>
        ))}

        {/* Loading indicator */}
        {loading && (
          <div className="self-start bg-white border border-gray-200 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
            <div className="flex gap-1 items-center h-5">
              <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
              <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
              <span className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Error */}
      {error && (
        <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded px-3 py-2 mb-3">
          {error}
        </p>
      )}

      {/* Input */}
      <form onSubmit={handleSubmit} className="flex gap-2">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={hasDocuments ? 'Ask a question…' : 'Upload a document first…'}
          disabled={loading || !hasDocuments}
          className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-sm
                     focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent
                     disabled:bg-gray-50 disabled:text-gray-400"
        />
        <button
          type="submit"
          disabled={loading || !question.trim() || !hasDocuments}
          className="bg-indigo-600 text-white rounded-lg px-4 py-2 text-sm font-medium
                     hover:bg-indigo-700 transition-colors
                     disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Ask
        </button>
      </form>
    </div>
  )
}
