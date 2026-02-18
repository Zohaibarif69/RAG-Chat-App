"use client";

import { useState, useRef, useEffect } from "react";
import "./globals.css";

interface Message {
  type: "user" | "assistant";
  content: string;
  sources?: Array<{ source: string; chunk: number; relevance: number }>;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [documentCount, setDocumentCount] = useState(0);
  const [documents, setDocuments] = useState<Array<{ name: string; chunks: number }>>([]);
  const [deletingDoc, setDeletingDoc] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const API_BASE = "http://127.0.0.1:8000";

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Fetch documents on mount
  useEffect(() => {
    fetchDocuments();
  }, []);

  // Fetch documents list
  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE}/documents`);
      const data = await response.json();
      console.log("Fetched documents:", data);
      if (data.success) {
        setDocuments(data.documents || []);
        // Update document count from fetched data
        const totalChunks = (data.documents || []).reduce((sum: number, doc: any) => sum + doc.chunks, 0);
        if (totalChunks > 0) {
          setDocumentCount(totalChunks);
        }
      }
    } catch (error) {
      console.error("Error fetching documents:", error);
    }
  };

  // Handle refresh documents
  const handleRefreshDocuments = async () => {
    setRefreshing(true);
    await fetchDocuments();
    setRefreshing(false);
  };

  // Handle delete individual document
  const handleDeleteDocument = async (docName: string) => {
    if (!window.confirm(`Delete "${docName}"? This cannot be undone.`)) {
      return;
    }

    setDeletingDoc(docName);
    try {
      const response = await fetch(`${API_BASE}/delete-document?source_name=${encodeURIComponent(docName)}`, {
        method: "POST",
      });

      const data = await response.json();

      if (data.success) {
        setMessages((prev) => [
          ...prev,
          {
            type: "assistant",
            content: `✓ Deleted "${docName}" successfully.`,
          },
        ]);
        // Refresh the documents list
        setTimeout(() => fetchDocuments(), 500);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            type: "assistant",
            content: ` Error: ${data.error || "Failed to delete document"}`,
          },
        ]);
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          content: ` Error: ${error instanceof Error ? error.message : "Delete failed"}`,
        },
      ]);
    } finally {
      setDeletingDoc(null);
    }
  };

  // Handle PDF upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setUploading(true);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        // Immediately enable the input by setting document count
        setDocumentCount(data.chunks_count);
        
        // Fetch and update the documents list
        try {
          const docResponse = await fetch(`${API_BASE}/documents`);
          const docData = await docResponse.json();
          if (docData.success) {
            setDocuments(docData.documents);
          }
        } catch (e) {
          console.error("Error fetching documents:", e);
        }
        
        setMessages((prev) => [
          ...prev,
          {
            type: "assistant",
            content: ` ${data.message}`,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            type: "assistant",
            content: ` Error: ${data.error || "Failed to upload PDF"}`,
          },
        ]);
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          content: ` Error: ${error instanceof Error ? error.message : "Upload failed"}`,
        },
      ]);
    } finally {
      setUploading(false);
      setFile(null);
      const input = e.target as HTMLInputElement;
      input.value = "";
    }
  };

  // Handle clear database
  const handleClearDatabase = async () => {
    if (!window.confirm("Are you sure you want to clear the database? This cannot be undone.")) {
      return;
    }

    setClearing(true);
    try {
      const response = await fetch(`${API_BASE}/clear-db`, {
        method: "POST",
      });

      const data = await response.json();

      if (data.success) {
        setDocumentCount(0);
        setMessages([]);
        setDocuments([]);
        setMessages((prev) => [
          ...prev,
          {
            type: "assistant",
            content: "✓ Database cleared successfully. All documents have been removed.",
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            type: "assistant",
            content: ` Error: ${data.error || "Failed to clear database"}`,
          },
        ]);
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          content: ` Error: ${error instanceof Error ? error.message : "Clear failed"}`,
        },
      ]);
    } finally {
      setClearing(false);
    }
  };

  // Handle RAG query
  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    if (documentCount === 0) {
      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          content: " Please upload a PDF first before asking questions.",
        },
      ]);
      return;
    }

    // Add user message
    const userMessage = input;
    setMessages((prev) => [...prev, { type: "user", content: userMessage }]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMessage, top_k: 5 }),
      });

      const data = await response.json();

      if (data.success) {
        setMessages((prev) => [
          ...prev,
          {
            type: "assistant",
            content: data.answer,
            sources: data.sources,
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            type: "assistant",
            content: ` Error: ${data.error || "Failed to get answer"}`,
          },
        ]);
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          content: ` Error: ${error instanceof Error ? error.message : "Query failed"}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <div className="w-64 bg-white border-r border-gray-200 p-6 flex flex-col">
        <h1 className="text-2xl font-bold mb-6 text-gray-800">RAG Chat</h1>

        {/* Upload Section */}
        <div className="mb-8">
          <h2 className="text-lg font-semibold mb-3 text-gray-700">Upload PDF</h2>
          <label className="flex items-center justify-center w-full px-4 py-3 bg-blue-50 border-2 border-blue-300 rounded-lg cursor-pointer hover:bg-blue-100 transition">
            <span className="text-blue-600 font-medium text-sm text-center">
              {file ? file.name : "Choose PDF file"}
            </span>
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileUpload}
              disabled={uploading}
              className="hidden"
            />
          </label>
          {uploading && (
            <p className="text-sm text-gray-500 mt-2">Uploading...</p>
          )}
        </div>

        {/* Status */}
        {documentCount > 0 && (
          <div className="mb-6 p-3 bg-green-50 rounded-lg border border-green-200">
            <p className="text-sm text-green-700">
               {documentCount} chunks loaded
            </p>
            <p className="text-xs text-gray-600 mt-1">Ready for questions</p>
            <button
              onClick={handleClearDatabase}
              disabled={clearing}
              className="mt-3 w-full px-3 py-2 bg-red-100 text-red-600 text-xs font-medium rounded hover:bg-red-200 disabled:bg-gray-200 disabled:text-gray-400 disabled:cursor-not-allowed transition"
            >
              {clearing ? "Clearing..." : "Clear Database"}
            </button>
          </div>
        )}
        {/* Documents List */}
        {documents.length > 0 && (
          <div className="mb-6 p-3 bg-blue-50 rounded-lg border border-blue-200">
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-sm font-semibold text-blue-700">Documents ({documents.length})</h3>
              <button
                onClick={handleRefreshDocuments}
                disabled={refreshing}
                className="text-xs px-2 py-1 bg-blue-200 text-blue-700 rounded hover:bg-blue-300 disabled:opacity-50"
              >
                {refreshing ? "..." : "Refresh"}
              </button>
            </div>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {documents.map((doc) => (
                <div key={doc.name} className="flex items-center justify-between bg-white p-2 rounded text-xs border border-gray-200">
                  <div className="flex-1">
                    <p className="font-medium text-gray-700 truncate">{doc.name}</p>
                    <p className="text-gray-500 text-xs">{doc.chunks} chunks</p>
                  </div>
                  <button
                    onClick={() => handleDeleteDocument(doc.name)}
                    disabled={deletingDoc === doc.name}
                    className="ml-2 px-3 py-1 bg-red-500 text-white rounded hover:bg-red-600 disabled:bg-gray-400 disabled:cursor-not-allowed transition text-xs font-medium"
                  >
                    {deletingDoc === doc.name ? "Deleting..." : "Delete"}
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
        {/* Instructions */}
        <div className="bg-gray-50 p-4 rounded-lg flex-1">
          <h3 className="font-semibold text-gray-700 mb-3 text-sm">How it works:</h3>
          <ol className="text-sm text-gray-600 space-y-2 list-decimal list-inside">
            <li>Upload a PDF file</li>
            <li>Ask a question about the content</li>
            <li>Get answers with sources</li>
          </ol>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col bg-white">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full text-gray-400 text-center">
              <div>
                <p className="text-lg font-semibold mb-2">Welcome to RAG Chat</p>
                <p className="text-sm">Upload a PDF and start asking questions</p>
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.type === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-md px-4 py-3 rounded-lg ${
                    msg.type === "user"
                      ? "bg-blue-600 text-white"
                      : "bg-gray-100 text-gray-800"
                  }`}
                >
                  <p className="text-sm whitespace-pre-wrap wrap-break-word">{msg.content}</p>

                  {/* Sources */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-gray-300">
                      <p className="text-xs font-semibold mb-2 text-gray-600">
                        Sources:
                      </p>
                      <div className="space-y-1">
                        {msg.sources.map((source, idx) => (
                          <div key={idx} className="text-xs bg-white p-2 rounded border border-gray-200">
                            <p className="font-medium text-gray-700">
                              {source.source} (Chunk {source.chunk})
                            </p>
                            <p className="text-gray-500">
                              Relevance: {(source.relevance * 100).toFixed(1)}%
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-gray-200 p-6 bg-gray-50">
          <form onSubmit={handleQuery} className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                documentCount === 0
                  ? "Upload a PDF first..."
                  : "Ask a question about your document..."
              }
              disabled={loading || (documentCount === 0 && documents.length === 0)}
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
            />
            <button
              type="submit"
              disabled={loading || !input.trim() || (documentCount === 0 && documents.length === 0)}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
            >
              {loading ? "..." : "Send"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
