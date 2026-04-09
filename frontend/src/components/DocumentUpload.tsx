/**
 * DocumentUpload — handles file selection, upload progress, and document list.
 *
 * Two responsibilities in one panel:
 *   1. Upload: drag-and-drop or click-to-browse for PDF/TXT files
 *   2. Library: list all indexed documents with chunk counts and delete buttons
 *
 * Parent manages the document list in state so ChatInterface can show
 * a helpful message when the library is empty.
 */

import { useRef, useState } from 'react'
import { uploadDocument, deleteDocument, type Document } from '../api'

interface Props {
  documents: Document[]
  onDocumentsChange: () => void  // tells parent to refresh the list
}

export function DocumentUpload({ documents, onDocumentsChange }: Props) {
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  async function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return
    const file = files[0]

    const allowed = ['application/pdf', 'text/plain']
    if (!allowed.includes(file.type)) {
      setUploadError('Only PDF and plain text files are supported.')
      return
    }

    setUploading(true)
    setUploadError(null)
    try {
      await uploadDocument(file)
      onDocumentsChange()
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  async function handleDelete(docId: string) {
    setDeletingId(docId)
    try {
      await deleteDocument(docId)
      onDocumentsChange()
    } catch {
      // deletion errors are rare — silently retry-able
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setIsDragging(false)
          handleFiles(e.dataTransfer.files)
        }}
        className={`
          border-2 border-dashed rounded-lg p-6 text-center cursor-pointer
          transition-colors select-none
          ${isDragging
            ? 'border-indigo-500 bg-indigo-50'
            : 'border-gray-300 hover:border-indigo-400 hover:bg-gray-50'
          }
          ${uploading ? 'pointer-events-none opacity-60' : ''}
        `}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".pdf,.txt,text/plain,application/pdf"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        <div className="text-3xl mb-2">📄</div>
        {uploading ? (
          <p className="text-sm text-indigo-600 font-medium">Uploading and indexing…</p>
        ) : (
          <>
            <p className="text-sm font-medium text-gray-700">
              Drop a file here or <span className="text-indigo-600">click to browse</span>
            </p>
            <p className="text-xs text-gray-400 mt-1">PDF or TXT · max 10 MB</p>
          </>
        )}
      </div>

      {uploadError && (
        <p className="text-sm text-red-600 bg-red-50 border border-red-200 rounded px-3 py-2">
          {uploadError}
        </p>
      )}

      {/* Document library */}
      {documents.length === 0 ? (
        <p className="text-sm text-gray-400 text-center py-2">
          No documents uploaded yet.
        </p>
      ) : (
        <ul className="flex flex-col gap-2">
          {documents.map((doc) => (
            <li
              key={doc.document_id}
              className="flex items-center justify-between bg-white border border-gray-200 rounded-lg px-3 py-2 shadow-sm"
            >
              <div className="min-w-0">
                <p className="text-sm font-medium text-gray-800 truncate">{doc.filename}</p>
                <p className="text-xs text-gray-400">{doc.chunk_count} chunks</p>
              </div>
              <button
                onClick={() => handleDelete(doc.document_id)}
                disabled={deletingId === doc.document_id}
                className="ml-3 text-gray-400 hover:text-red-500 transition-colors disabled:opacity-40 flex-shrink-0"
                title="Remove document"
              >
                {deletingId === doc.document_id ? '…' : '✕'}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
