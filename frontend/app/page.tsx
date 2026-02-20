"use client";

import { useState, useRef, useEffect } from "react";
import "./globals.css";
import ChatMessage from "./components/ChatMessage";
import useStreamingQuery, { StreamMetadata } from "./hooks/useStreamingQuery";

interface Message {
  type: "user" | "assistant";
  content: string;
  sources?: Array<{ source: string; chunk: number; relevance: number }>;
  metadata?: StreamMetadata;
  retrievedChunks?: Array<{ source: string; content: string; relevance: number }>;
  created_at?: string;
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
  const [sessionId, setSessionId] = useState<string>("");
  const [streamingMessageIdx, setStreamingMessageIdx] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { query: streamQuery, streaming } = useStreamingQuery();

  const API_BASE = "http://127.0.0.1:8000";

  // Initialize session on mount
  useEffect(() => {
    const initializeSession = async () => {
      try {
        // Try to load session from localStorage
        const savedSessionId = localStorage.getItem("rag_session_id");
        
        if (savedSessionId) {
          setSessionId(savedSessionId);
          // Load conversation history
          await loadChatHistory(savedSessionId);
        } else {
          // Create new session
          const response = await fetch(`${API_BASE}/session`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
          });
          const data = await response.json();
          if (data.session_id) {
            setSessionId(data.session_id);
            localStorage.setItem("rag_session_id", data.session_id);
          }
        }
      } catch (error) {
        console.error("Error initializing session:", error);
        // Fallback: create a UUID locally
        const fallbackSessionId = `session_${Date.now()}`;
        setSessionId(fallbackSessionId);
        localStorage.setItem("rag_session_id", fallbackSessionId);
      }
    };

    initializeSession();
    fetchDocuments();
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load chat history from database
  const loadChatHistory = async (sid: string) => {
    try {
      const response = await fetch(`${API_BASE}/session/${sid}/history`);
      const data = await response.json();
      if (data.success && data.messages) {
        const loadedMessages = data.messages.map((msg: any) => ({
          type: msg.role,
          content: msg.content,
          sources: msg.sources,
          created_at: msg.created_at,
        }));
        setMessages(loadedMessages);
      }
    } catch (error) {
      console.error("Error loading chat history:", error);
    }
  };

  // Fetch documents list
  const fetchDocuments = async () => {
    try {
      console.log("Fetching documents...");
      const response = await fetch(`${API_BASE}/documents`);
      const data = await response.json();
      console.log("Fetched documents response:", data);
      
      if (data.success) {
        const docs = data.documents || [];
        console.log("Documents to display:", docs);
        setDocuments(docs);
        
        // Update document count from fetched data
        const totalChunks = docs.reduce((sum: number, doc: any) => sum + doc.chunks, 0);
        console.log("Total chunks calculated:", totalChunks);
        if (totalChunks > 0) {
          setDocumentCount(totalChunks);
        }
      } else {
        console.error("Error fetching documents:", data.error);
        setDocuments([]);
      }
    } catch (error) {
      console.error("Error fetching documents:", error);
      setDocuments([]);
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
      console.log(`Deleting document: ${docName}`);
      const response = await fetch(`${API_BASE}/delete-document?source_name=${encodeURIComponent(docName)}`, {
        method: "POST",
      });

      const data = await response.json();
      console.log("Delete response:", data);

      if (data.success) {
        setMessages((prev) => [
          ...prev,
          {
            type: "assistant",
            content: `Deleted "${docName}" successfully.`,
          },
        ]);
        // Refresh documents list with small delay
        await new Promise(resolve => setTimeout(resolve, 300));
        await fetchDocuments();
      } else {
        setMessages((prev) => [
          ...prev,
          {
            type: "assistant",
            content: `Error deleting "${docName}": ${data.error || "Unknown error"}`,
          },
        ]);
      }
    } catch (error) {
      console.error("Delete error:", error);
      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          content: `Error: ${error instanceof Error ? error.message : "Delete failed"}`,
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
        
        // Show success message
        setMessages((prev) => [
          ...prev,
          {
            type: "assistant",
            content: `${data.message}`,
          },
        ]);
        
        // Fetch and update the documents list with a small delay
        setTimeout(async () => {
          try {
            console.log("Auto-refresh: Fetching documents after upload...");
            const docResponse = await fetch(`${API_BASE}/documents`);
            const docData = await docResponse.json();
            console.log("Auto-refresh: Documents response:", docData);
            if (docData.success && docData.documents) {
              console.log("Auto-refresh: Setting documents:", docData.documents);
              setDocuments(docData.documents);
            }
          } catch (e) {
            console.error("Error auto-refreshing documents:", e);
          }
        }, 500);
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
    if (!window.confirm("Clear all documents from the database? This cannot be undone.")) {
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
        setDocuments([]);
        setMessages([
          {
            type: "assistant",
            content: "Database cleared successfully. All documents have been removed.",
          },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            type: "assistant",
            content: `Error: ${data.error || "Failed to clear database"}`,
          },
        ]);
      }
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          content: `Error: ${error instanceof Error ? error.message : "Clear failed"}`,
        },
      ]);
    } finally {
      setClearing(false);
    }
  };

  // Handle new chat session
  const handleNewChat = async () => {
    if (messages.length > 0 && !window.confirm("Start a new chat? Current conversation will be saved.")) {
      return;
    }

    try {
      // Create a new session
      const response = await fetch(`${API_BASE}/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const data = await response.json();
      
      if (data.session_id) {
        setSessionId(data.session_id);
        localStorage.setItem("rag_session_id", data.session_id);
        setMessages([]);
      }
    } catch (error) {
      console.error("Error creating new chat:", error);
    }
  };

  // Handle RAG query with conversation context and streaming
  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading || streaming) return;

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

    // Add user message to UI immediately
    const userMessage = input;
    setMessages((prev) => [...prev, { type: "user", content: userMessage }]);
    setInput("");
    setLoading(true);

    // Add streaming assistant message placeholder
    const messageIdx = messages.length + 1;
    setStreamingMessageIdx(messageIdx);
    setMessages((prev) => [
      ...prev,
      {
        type: "assistant",
        content: "",
        metadata: {
          sources: [],
          confidence: 0,
          hallucination_risk: "low",
          verified: false,
        },
        retrievedChunks: [],
      },
    ]);

    await streamQuery(userMessage, sessionId, 5, (response) => {
      setMessages((prev) => {
        const updated = [...prev];
        if (updated[messageIdx]) {
          updated[messageIdx] = {
            type: "assistant",
            content: response.answer,
            metadata: response.metadata || undefined,
            retrievedChunks: [],
          };
        }
        return updated;
      });

      if (response.isComplete) {
        setLoading(false);
        setStreamingMessageIdx(null);
      }
    });
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <div className="w-64 bg-white border-r border-gray-200 p-6 flex flex-col">
        <h1 className="text-2xl font-bold mb-6 text-gray-800">RAG Chat</h1>

        {/* New Chat Button */}
        <button
          onClick={handleNewChat}
          className="w-full mb-6 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 font-medium transition"
        >
          New Chat
        </button>

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
              className="mt-4 w-full px-4 py-2 bg-red-600 text-white text-sm font-semibold rounded hover:bg-red-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
            >
              {clearing ? "Clearing..." : "Clear Database"}
            </button>
          </div>
        )}
        {/* Documents List - Professional UI */}
        {documents && documents.length > 0 ? (
          <div className="mb-6 p-4 bg-white rounded border border-gray-300 shadow-sm">
            <div className="flex justify-between items-center mb-3 pb-2 border-b border-gray-200">
              <h3 className="text-sm font-semibold text-gray-800">
                Documents ({documents.length})
              </h3>
              <button
                onClick={handleRefreshDocuments}
                disabled={refreshing}
                className="text-xs px-3 py-1.5 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 disabled:bg-gray-300 disabled:cursor-not-allowed transition font-medium"
              >
                {refreshing ? "Loading..." : "Refresh"}
              </button>
            </div>
            
            <div className="space-y-1 max-h-72 overflow-y-auto">
              {documents.map((doc, idx) => (
                <div
                  key={doc.name}
                  className="flex items-center justify-between bg-gray-50 p-2.5 rounded border border-gray-200 hover:bg-gray-100 transition"
                >
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-gray-800 text-sm truncate">
                      {doc.name}
                    </p>
                    <p className="text-gray-600 text-xs">
                      {doc.chunks} chunk{doc.chunks !== 1 ? 's' : ''}
                    </p>
                  </div>
                  <button
                    onClick={() => handleDeleteDocument(doc.name)}
                    disabled={deletingDoc === doc.name}
                    className="ml-3 px-3 py-1 bg-red-500 hover:bg-red-600 text-white rounded text-xs font-medium disabled:bg-gray-400 disabled:cursor-not-allowed transition whitespace-nowrap"
                  >
                    {deletingDoc === doc.name ? "Deleting..." : "Delete"}
                  </button>
                </div>
              ))}
            </div>
          </div>
        ) : documentCount > 0 ? (
          <div className="mb-6 p-4 bg-white rounded border border-gray-300 shadow-sm">
            <p className="text-sm font-medium text-gray-800">
              Documents list loading...
            </p>
            <p className="text-xs text-gray-600 mt-1">
              {documentCount} chunks are loaded. Click Refresh to reload the list.
            </p>
            <button
              onClick={handleRefreshDocuments}
              disabled={refreshing}
              className="mt-3 px-3 py-1.5 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 text-xs font-medium transition"
            >
              {refreshing ? "Loading..." : "Refresh"}
            </button>
          </div>
        ) : null}
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
              <ChatMessage
                key={idx}
                type={msg.type}
                content={msg.content}
                metadata={msg.metadata}
                isStreaming={streamingMessageIdx === idx && streaming}
                retrievedChunks={msg.retrievedChunks || []}
              />
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
              disabled={loading || streaming || !input.trim() || (documentCount === 0 && documents.length === 0)}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
            >
              {loading || streaming ? "..." : "Send"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
