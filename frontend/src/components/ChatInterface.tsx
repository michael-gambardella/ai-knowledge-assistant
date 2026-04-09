/**
 * ChatInterface — question input and streaming answer display.
 *
 * Uses the /query/stream SSE endpoint so the answer renders token-by-token
 * instead of appearing all at once after a 1-2s wait. The source citations
 * appear immediately after retrieval (before generation starts) because the
 * backend sends a "sources" event before the first "token" event.
 *
 * History entry lifecycle:
 *   1. User submits → entry added with empty answer, streaming=true
 *   2. "sources" event → sources + retrieval_ms populated
 *   3. "token" events → answer text builds up character by character
 *   4. "done" event → generation_ms, total_ms, mean_relevance_score set; streaming=false
 *   5. "error" event → error message shown, streaming=false
 */

import { useState, useRef, useEffect } from 'react'
import { streamQuery, type SourceChunk } from '../api'
import { SourceCitations } from './SourceCitations'
import ReactMarkdown from 'react-markdown'

interface HistoryEntry {
  id: string
  question: string
  answer: string
  sources: SourceChunk[]
  retrieval_ms: number
  generation_ms: number | null  // null until "done" event
  total_ms: number | null
  mean_relevance_score: number | null
  streaming: boolean
  error: string | null
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
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [history])

  function updateEntry(id: string, patch: Partial<HistoryEntry>) {
    setHistory((prev) => prev.map((e) => (e.id === id ? { ...e, ...patch } : e)))
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const q = question.trim()
    if (!q || loading) return

    const id = crypto.randomUUID()
    const entry: HistoryEntry = {
      id,
      question: q,
      answer: '',
      sources: [],
      retrieval_ms: 0,
      generation_ms: null,
      total_ms: null,
      mean_relevance_score: null,
      streaming: true,
      error: null,
    }

    setLoading(true)
    setQuestion('')
    setHistory((prev) => [...prev, entry])

    try {
      for await (const event of streamQuery(q)) {
        if (event.type === 'sources') {
          updateEntry(id, { sources: event.sources, retrieval_ms: event.retrieval_ms })
        } else if (event.type === 'token') {
          setHistory((prev) =>
            prev.map((e) => (e.id === id ? { ...e, answer: e.answer + event.text } : e))
          )
        } else if (event.type === 'done') {
          updateEntry(id, {
            generation_ms: event.generation_ms,
            total_ms: event.total_ms,
            mean_relevance_score: event.mean_relevance_score,
            streaming: false,
          })
        } else if (event.type === 'error') {
          updateEntry(id, { error: event.message, streaming: false })
        }
      }
    } catch (err) {
      updateEntry(id, {
        error: err instanceof Error ? err.message : 'Query failed',
        streaming: false,
      })
      setQuestion(q)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Answer history */}
      <div className="flex-1 overflow-y-auto flex flex-col gap-6 pb-4">
        {history.length === 0 && (
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

        {history.map((entry) => (
          <div key={entry.id} className="flex flex-col gap-3">
            {/* Question bubble */}
            <div className="self-end max-w-[80%] bg-indigo-600 text-white rounded-2xl rounded-tr-sm px-4 py-2.5">
              <p className="text-sm">{entry.question}</p>
            </div>

            {/* Answer card */}
            <div className="self-start max-w-full bg-white border border-gray-200 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
              {entry.error ? (
                <p className="text-sm text-red-600">{entry.error}</p>
              ) : (
                <>
                  <div className="text-sm text-gray-800 leading-relaxed prose prose-sm max-w-none">
                    <ReactMarkdown>{entry.answer}</ReactMarkdown>
                    {/* Blinking cursor while streaming */}
                    {entry.streaming && (
                      <span className="inline-block w-0.5 h-4 bg-indigo-500 ml-0.5 animate-pulse align-middle" />
                    )}
                  </div>

                  {/* Metrics — only shown once "done" event arrives */}
                  {!entry.streaming && entry.total_ms !== null && (
                    <div className="flex flex-wrap gap-1.5 mt-3">
                      <LatencyBadge label="retrieval" value={entry.retrieval_ms} />
                      <LatencyBadge label="generation" value={entry.generation_ms!} />
                      <LatencyBadge label="total" value={entry.total_ms} />
                      <RelevanceBadge score={entry.mean_relevance_score!} />
                    </div>
                  )}

                  {/* Sources — shown as soon as retrieval completes (before generation finishes) */}
                  <SourceCitations sources={entry.sources} />
                </>
              )}
            </div>
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="flex gap-2 mt-2">
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
          {loading ? '…' : 'Ask'}
        </button>
      </form>
    </div>
  )
}
