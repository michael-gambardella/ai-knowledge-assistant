import { useEffect, useState } from 'react'
import { listDocuments, type Document } from './api'
import { DocumentUpload } from './components/DocumentUpload'
import { ChatInterface } from './components/ChatInterface'

export default function App() {
  const [documents, setDocuments] = useState<Document[]>([])

  async function refreshDocuments() {
    try {
      const docs = await listDocuments()
      setDocuments(docs)
    } catch {
      // backend not ready yet — silent, will retry on next interaction
    }
  }

  useEffect(() => {
    refreshDocuments()
  }, [])

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center gap-3">
        <span className="text-2xl">🧠</span>
        <div>
          <h1 className="text-lg font-semibold text-gray-900 leading-none">AI Knowledge Assistant</h1>
          <p className="text-xs text-gray-400 mt-0.5">Upload documents · Ask questions · Get cited answers</p>
        </div>
      </header>

      {/* Two-panel layout */}
      <div className="flex-1 flex overflow-hidden">

        {/* Left panel — document library */}
        <aside className="w-72 flex-shrink-0 bg-white border-r border-gray-200 flex flex-col">
          <div className="px-4 py-3 border-b border-gray-100">
            <h2 className="text-sm font-semibold text-gray-700">Documents</h2>
            <p className="text-xs text-gray-400">{documents.length} indexed</p>
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            <DocumentUpload
              documents={documents}
              onDocumentsChange={refreshDocuments}
            />
          </div>
        </aside>

        {/* Right panel — chat */}
        <main className="flex-1 flex flex-col p-6 overflow-hidden">
          <ChatInterface hasDocuments={documents.length > 0} />
        </main>
      </div>
    </div>
  )
}
