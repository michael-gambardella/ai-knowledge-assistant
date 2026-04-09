/**
 * API client — thin wrappers around fetch for each backend endpoint.
 *
 * All calls go through the Vite dev proxy (/api → http://localhost:8000),
 * so no hardcoded backend URL lives in the frontend code. In production
 * you'd set VITE_API_BASE to the deployed backend URL via .env.
 */

const BASE = import.meta.env.VITE_API_BASE ?? '/api'

// --- Types that mirror the backend Pydantic schemas ---

export interface Document {
  document_id: string
  filename: string
  chunk_count: number
  uploaded_at: string
}

export interface SourceChunk {
  document_id: string
  filename: string
  content: string
  relevance_score: number
  chunk_index: number
}

export interface QueryResponse {
  answer: string
  sources: SourceChunk[]
  retrieval_ms: number
  generation_ms: number
  total_ms: number
  mean_relevance_score: number
}

// --- Document endpoints ---

export async function uploadDocument(file: File): Promise<Document> {
  const form = new FormData()
  form.append('file', file)

  const res = await fetch(`${BASE}/documents/upload`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? 'Upload failed')
  }
  return res.json()
}

export async function listDocuments(): Promise<Document[]> {
  const res = await fetch(`${BASE}/documents/`)
  if (!res.ok) throw new Error('Failed to load documents')
  const data = await res.json()
  return data.documents
}

export async function deleteDocument(documentId: string): Promise<void> {
  const res = await fetch(`${BASE}/documents/${documentId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Failed to delete document')
}

// --- Query endpoint ---

export async function queryKnowledgeBase(
  question: string,
  topK?: number,
): Promise<QueryResponse> {
  const res = await fetch(`${BASE}/query/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? 'Query failed')
  }
  return res.json()
}
