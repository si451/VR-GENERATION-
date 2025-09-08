"use client"

import type React from "react"
import { useState, useCallback, useEffect, useRef } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Upload, FileVideo, X, Play, Sparkles, AlertCircle, CheckCircle } from "lucide-react"
import { cn } from "@/lib/utils"
import { useToast } from "@/hooks/use-toast"

interface UploadedFile {
  id: string
  name: string
  size: number
  type: string
  progress: number
  status: "uploading" | "processing" | "completed" | "error"
  vrUrl?: string
  jobId?: string
  errorMessage?: string
  currentStage?: string
  stageMessage?: string
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://213.163.196.177:8000'

// Debug: Log the API URL being used
console.log('API_BASE_URL:', API_BASE_URL)
console.log('NEXT_PUBLIC_API_URL env:', process.env.NEXT_PUBLIC_API_URL)

export function UploadInterface() {
  const router = useRouter()
  const { toast } = useToast()
  const [files, setFiles] = useState<UploadedFile[]>([])
  const [isDragOver, setIsDragOver] = useState(false)
  const intervalsRef = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map())
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const droppedFiles = Array.from(e.dataTransfer.files)
    handleFiles(droppedFiles)
  }, [])

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files)
      handleFiles(selectedFiles)
    }
  }, [])

  const handleFiles = async (fileList: File[]) => {
    const newFiles: UploadedFile[] = fileList.map((file) => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: file.size,
      type: file.type,
      progress: 0,
      status: "uploading",
      vrUrl: undefined,
    }))

    setFiles((prev) => [...prev, ...newFiles])

    // Upload each file to the backend
    for (const file of newFiles) {
      await uploadToBackend(file, fileList.find(f => f.name === file.name)!)
    }
  }

  const uploadToBackend = async (file: UploadedFile, actualFile: File) => {
    try {
      const formData = new FormData()
      formData.append('file', actualFile)

      setFiles((prev) =>
        prev.map((f) =>
          f.id === file.id ? { ...f, status: "uploading", progress: 10 } : f
        )
      )

      toast({
        title: "Upload Started",
        description: `Uploading ${file.name}...`,
        duration: 3000,
      })

      console.log('Uploading to:', `${API_BASE_URL}/upload`)
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
        headers: {
          'Accept': 'application/json',
        },
      })
      console.log('Upload response status:', response.status)
      console.log('Upload response headers:', response.headers)

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Upload failed')
      }

      const result = await response.json()
      const jobId = result.job_id

      setFiles((prev) =>
        prev.map((f) =>
          f.id === file.id 
            ? { ...f, jobId, status: "processing", progress: 20 } 
            : f
        )
      )

      toast({
        title: "Upload Successful",
        description: `${file.name} uploaded successfully. Processing started... (This may take 30-40 minutes for large videos)`,
        duration: 6000,
      })

      // Start polling for status updates
      const interval = pollJobStatus(jobId, file.id)
      intervalsRef.current.set(file.id, interval)
    } catch (error) {
      console.error('Upload error:', error)
      setFiles((prev) =>
        prev.map((f) =>
          f.id === file.id 
            ? { 
                ...f, 
                status: "error", 
                progress: 0,
                errorMessage: error instanceof Error ? error.message : 'Upload failed'
              } 
            : f
        )
      )

      toast({
        title: "Upload Failed",
        description: `Failed to upload ${file.name}: ${error instanceof Error ? error.message : 'Unknown error'}`,
        variant: "destructive",
        duration: 5000,
      })
    }
  }

  const pollJobStatus = useCallback((jobId: string, fileId: string) => {
    let lastStage = "queued"
    let lastProgress = 0
    let pollCount = 0
    let timeoutId: ReturnType<typeof setTimeout>
    let isPolling = true
    
    const poll = async () => {
      if (!isPolling) return
      
      try {
        pollCount++
        
        // Use stage transition endpoint for smart polling
        const response = await fetch(`${API_BASE_URL}/status/${jobId}/stage-transitions`, {
          headers: {
            'Accept': 'application/json',
          },
        })
        
        if (!response.ok) {
          const errorText = await response.text()
          console.error(`Status fetch failed: ${response.status} - ${errorText}`)
          throw new Error(`Failed to fetch status: ${response.status}`)
        }

        const data = await response.json()
        const status = data.status
        const hasTransition = data.has_transition
        
        // Only update UI on stage transitions (start/end of stages)
        if (hasTransition || status.stage !== lastStage) {
          lastStage = status.stage || "unknown"
          lastProgress = status.percent || 0
        
          setFiles((prev) =>
            prev.map((f) => {
              if (f.id === fileId) {
                const progress = status.percent || 0
                let newStatus: UploadedFile["status"] = "processing"

                if (status.status === "done") {
                  newStatus = "completed"
                  isPolling = false
                  if (timeoutId) clearTimeout(timeoutId)
                  intervalsRef.current.delete(fileId)
                  // Show completion notification
                  toast({
                    title: "Processing Complete!",
                    description: `${f.name} has been successfully converted to VR180 format.`,
                    duration: 5000,
                  })
                } else if (status.status === "failed") {
                  newStatus = "error"
                  isPolling = false
                  if (timeoutId) clearTimeout(timeoutId)
                  intervalsRef.current.delete(fileId)
                  // Show error notification
                  toast({
                    title: "Processing Failed",
                    description: `Failed to process ${f.name}: ${status.message || 'Unknown error'}`,
                    variant: "destructive",
                    duration: 5000,
                  })
                }

                return {
                  ...f,
                  progress,
                  status: newStatus,
                  vrUrl: status.output ? `${API_BASE_URL}/download/${jobId}` : undefined,
                  errorMessage: status.message || status.error_message,
                  currentStage: status.stage || status.current_stage,
                  stageMessage: status.message || status.stage_message,
                }
              }
              return f
            })
          )
        }
        
        // Stop polling if job is done or failed
        if (status.status === "done" || status.status === "failed") {
          isPolling = false
          if (timeoutId) clearTimeout(timeoutId)
          intervalsRef.current.delete(fileId)
          return
        }
        
        // Determine next poll interval based on stage transitions
        let nextPollInterval = 10000 // Default 10 seconds (reduced from 5)
        
        if (hasTransition || status.stage !== lastStage) {
          // Poll quickly on stage changes
          nextPollInterval = 3000 // Reduced from 2 seconds
        } else if (status.status === "running") {
          // Poll every 30 seconds during processing (waiting for stage transitions)
          nextPollInterval = 30000 // Increased from 15 seconds
        }
        
        // Schedule next poll
        if (isPolling) {
          timeoutId = setTimeout(poll, nextPollInterval)
        }
        
      } catch (error) {
        console.error('Status polling error:', error)
        isPolling = false
        if (timeoutId) clearTimeout(timeoutId)
        setFiles((prev) =>
          prev.map((f) =>
            f.id === fileId 
              ? { 
                  ...f, 
                  status: "error",
                  errorMessage: `Failed to check processing status: ${error instanceof Error ? error.message : 'Unknown error'}`
                } 
              : f
          )
        )
      }
    }
    
    // Start polling
    poll()
    
    // Return cleanup function
    return () => {
      isPolling = false
      if (timeoutId) clearTimeout(timeoutId)
    }
  }, [])

  const removeFile = useCallback((fileId: string) => {
    // Clear polling if it exists
    const cleanup = intervalsRef.current.get(fileId)
    if (cleanup) {
      cleanup() // Call cleanup function
      intervalsRef.current.delete(fileId)
    }
    setFiles((prev) => prev.filter((file) => file.id !== fileId))
  }, [])

  const clearCompletedFiles = useCallback(() => {
    setFiles((prev) => {
      const remaining = prev.filter(f => f.status !== "completed")
      // Clear polling for completed files
      prev.forEach(f => {
        if (f.status === "completed" && intervalsRef.current.has(f.id)) {
          const cleanup = intervalsRef.current.get(f.id)!
          cleanup() // Call cleanup function
          intervalsRef.current.delete(f.id)
        }
      })
      return remaining
    })
  }, [])

  const clearAllFiles = useCallback(() => {
    setFiles([])
    // Clear all polling
    intervalsRef.current.forEach((cleanup) => cleanup())
    intervalsRef.current.clear()
  }, [])

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      intervalsRef.current.forEach((cleanup) => cleanup())
      intervalsRef.current.clear()
    }
  }, [])

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes"
    const k = 1024
    const sizes = ["Bytes", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
  }

  const getStatusColor = (status: UploadedFile["status"]) => {
    switch (status) {
      case "uploading":
        return "bg-blue-500"
      case "processing":
        return "bg-yellow-500"
      case "completed":
        return "bg-green-500"
      case "error":
        return "bg-red-500"
      default:
        return "bg-gray-500"
    }
  }

  const getStatusText = (file: UploadedFile) => {
    switch (file.status) {
      case "uploading":
        return "Uploading..."
        case "processing":
          const baseMessage = file.stageMessage || file.currentStage || "Converting to VR180..."
          return `${baseMessage}`
      case "completed":
        return "Ready for VR"
      case "error":
        return file.errorMessage || "Error"
      default:
        return "Unknown"
    }
  }

  const getStageDisplayName = (stage: string): string => {
    const stageMap: { [key: string]: string } = {
      'queued': 'Queued',
      'starting': 'Starting',
      'probe_video': 'Analyzing',
      'extract_frames': 'Extracting Frames',
      'depth_estimation': 'Processing Depth',
      'temporal_smoothing': 'Smoothing',
      'ldi_reprojection': 'Creating VR180',
      'encode': 'Finalizing',
      'finished': 'Completed'
    }
    return stageMap[stage] || 'Processing'
  }

  const getStageProgress = (stage: string, percent: number): number => {
    // Map stages to their progress ranges for better visual feedback
    const stageRanges: { [key: string]: { min: number; max: number } } = {
      'queued': { min: 0, max: 5 },
      'starting': { min: 5, max: 10 },
      'probe_video': { min: 10, max: 15 },
      'extract_frames': { min: 15, max: 30 },
      'depth_estimation': { min: 30, max: 55 },
      'temporal_smoothing': { min: 55, max: 70 },
      'ldi_reprojection': { min: 70, max: 90 },
      'encode': { min: 90, max: 100 },
      'finished': { min: 100, max: 100 }
    }
    
    const range = stageRanges[stage] || { min: 0, max: 100 }
    const stageProgress = Math.max(0, Math.min(100, percent))
    
    // Interpolate within the stage range
    return range.min + (stageProgress / 100) * (range.max - range.min)
  }

  const getDetailedStatus = (file: UploadedFile) => {
    if (file.status === "processing") {
      const stage = file.currentStage || "processing"
      
      // Simple stage descriptions without technical details
      const stageMap: { [key: string]: string } = {
        "starting": "🚀 Starting video processing...",
        "probe_video": "📊 Analyzing video...",
        "extract_frames": "🎬 Extracting frames...",
        "depth_estimation": "🧠 Processing depth...",
        "temporal_smoothing": "🔄 Smoothing motion...",
        "ldi_reprojection": "🎭 Creating VR180 views...",
        "encode": "🎬 Finalizing video...",
        "finished": "✅ Processing completed!",
        "frame_extraction": "🎬 Extracting frames...",
        "ldi_creation": "🎭 Creating VR180 views...",
        "inpainting": "🎨 Processing views...",
        "interpolation": "⚡ Finalizing...",
        "encoding": "🎥 Finalizing video...",
        "finalizing": "✨ Completing..."
      }
      
      return stageMap[stage] || "Processing video..."
    }
    return null
  }

  return (
    <div className="space-y-8">
      {/* Upload Area */}
      <Card className={cn(
        "border-2 border-dashed transition-all duration-300 bg-black/20 backdrop-blur-xl shadow-xl hover:shadow-2xl",
        isDragOver 
          ? "border-primary bg-primary/10 scale-[1.02] shadow-primary/20" 
          : "border-white/20 hover:border-primary/60 hover:bg-primary/5"
      )}>
        <CardContent className="p-12">
          <div
            className={cn(
              "flex flex-col items-center justify-center space-y-8 text-center cursor-pointer",
              isDragOver && "scale-105 transition-all duration-200",
            )}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className={cn(
              "w-24 h-24 rounded-full flex items-center justify-center shadow-lg transition-all duration-300",
              isDragOver 
                ? "bg-primary/30 scale-110 animate-pulse" 
                : "bg-gradient-to-br from-primary/20 to-secondary/20 hover:scale-105"
            )}>
              <Upload className={cn(
                "w-12 h-12 transition-colors duration-300",
                isDragOver ? "text-primary" : "text-primary/80"
              )} />
            </div>
            <div className="space-y-4">
              <h3 className="text-3xl font-bold text-white mb-4 flex items-center justify-center gap-3">
                <Sparkles className="w-8 h-8 text-primary" />
                {isDragOver ? "🎬 Drop your videos here!" : "Upload Your 2D Video Files"}
              </h3>
              <p className="text-xl text-gray-300 mb-6 font-medium">
                {isDragOver ? "Release to start uploading" : "Drag and drop your video files here, or click to browse"}
              </p>
              
              {/* Processing Time Disclaimer */}
              <div className="bg-amber-500/20 border border-amber-500/30 rounded-xl p-4 mb-6 backdrop-blur-sm">
                <div className="flex items-center justify-center gap-3 text-amber-200">
                  <AlertCircle className="w-5 h-5 text-amber-400 flex-shrink-0" />
                  <div className="text-center">
                    <p className="font-semibold text-lg mb-1">⏱️ Processing Time Notice</p>
                    <p className="text-sm leading-relaxed">
                      Large videos take approximately <span className="font-bold text-amber-300">30-40 minutes</span> to process. 
                      Please be patient - you will receive a <span className="font-bold text-amber-300">high-quality VR180 output</span> with original audio preserved.
                    </p>
                  </div>
                </div>
              </div>
              
              <div className="flex flex-wrap gap-3 justify-center text-sm">
                <Badge variant="secondary" className="bg-primary/20 text-primary border-primary/30 font-medium px-4 py-2 text-sm">
                  MP4
                </Badge>
                <Badge variant="secondary" className="bg-primary/20 text-primary border-primary/30 font-medium px-4 py-2 text-sm">
                  MOV
                </Badge>
                <Badge variant="secondary" className="bg-primary/20 text-primary border-primary/30 font-medium px-4 py-2 text-sm">
                  AVI
                </Badge>
                <Badge variant="secondary" className="bg-primary/20 text-primary border-primary/30 font-medium px-4 py-2 text-sm">
                  MKV
                </Badge>
              </div>
            </div>
            <div className="flex gap-4">
              <Button className="relative bg-gradient-to-r from-primary via-primary to-secondary hover:from-primary/90 hover:via-primary/90 hover:to-secondary/90 text-white font-semibold shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:scale-105 px-12 py-4 text-lg rounded-xl">
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept="video/*"
                  onChange={handleFileInput}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
                <Upload className="w-5 h-5 mr-2" />
                Choose Files
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* File List */}
      {files.length > 0 && (
        <Card className="bg-black/20 backdrop-blur-xl shadow-xl border-white/10">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-2xl font-bold text-white flex items-center gap-3">
                  <FileVideo className="w-6 h-6 text-primary" />
                  Processing Queue
                </CardTitle>
                <CardDescription className="text-lg text-gray-300">
                  Track the progress of your video conversions
                </CardDescription>
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={clearCompletedFiles}
                  className="border-green-500/30 hover:border-green-500 hover:bg-green-500/10 text-green-400 hover:text-green-300 transition-all duration-300"
                >
                  Clear Completed
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={clearAllFiles}
                  className="border-red-500/30 hover:border-red-500 hover:bg-red-500/10 text-red-400 hover:text-red-300 transition-all duration-300"
                >
                  Clear All
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            {files.map((file) => (
              <div
                key={file.id}
                className="flex items-center gap-6 p-6 border border-white/20 rounded-2xl bg-white/5 backdrop-blur-sm hover:bg-white/10 transition-all duration-300 shadow-lg hover:shadow-xl hover:scale-[1.02]"
              >
                <div className="w-16 h-16 bg-gradient-to-br from-primary/20 to-secondary/20 rounded-2xl flex items-center justify-center shadow-lg">
                  <FileVideo className="w-8 h-8 text-primary" />
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="font-bold text-white truncate text-xl">{file.name}</h4>
                    <div className="flex items-center gap-4">
                      <Badge
                        variant="secondary"
                        className={cn("text-white font-semibold px-4 py-2 text-sm rounded-xl", getStatusColor(file.status))}
                      >
                        {getStatusText(file)}
                      </Badge>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => removeFile(file.id)}
                        className="h-10 w-10 hover:bg-red-500/20 hover:text-red-400 transition-all duration-200 rounded-xl"
                      >
                        <X className="h-5 w-5" />
                      </Button>
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-base text-gray-300 mb-4 font-semibold">
                    <span>{formatFileSize(file.size)}</span>
                    <span className="text-primary">{Math.round(file.progress)}%</span>
                  </div>

                  <Progress value={file.progress} className="h-4 bg-white/10 rounded-full" />
                  
                  {/* Detailed Processing Status */}
                  {file.status === "processing" && getDetailedStatus(file) && (
                    <div className="mt-4 p-4 bg-primary/10 border border-primary/30 rounded-xl">
                      <div className="flex items-center gap-3">
                        <div className="w-3 h-3 bg-primary rounded-full animate-pulse"></div>
                        <p className="text-base text-primary font-semibold">{getDetailedStatus(file)}</p>
                      </div>
                      {file.currentStage && (
                        <div className="mt-2 text-sm text-primary/80 font-medium">
                          Stage: {getStageDisplayName(file.currentStage)}
                        </div>
                      )}
                      <div className="mt-3">
                        <div className="flex justify-between text-xs text-primary/60 mb-1">
                          <span>Progress</span>
                          <span>{Math.round(getStageProgress(file.currentStage || 'queued', file.progress || 0))}%</span>
                        </div>
                        <div className="w-full bg-primary/20 rounded-full h-2">
                          <div 
                            className="bg-primary h-2 rounded-full transition-all duration-500 ease-out"
                            style={{ width: `${getStageProgress(file.currentStage || 'queued', file.progress || 0)}%` }}
                          ></div>
                        </div>
                        <div className="mt-1 text-xs text-primary/50">
                          {getStageDisplayName(file.currentStage || 'queued')} - {file.progress || 0}%
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {file.status === "error" && file.errorMessage && (
                    <div className="mt-4 p-4 bg-red-500/10 border border-red-500/30 rounded-xl flex items-start gap-3">
                      <AlertCircle className="h-5 w-5 text-red-400 mt-0.5 flex-shrink-0" />
                      <p className="text-base text-red-400 font-semibold">{file.errorMessage}</p>
                    </div>
                  )}
                </div>

                {file.status === "completed" && file.vrUrl && (
                  <Button
                    size="lg"
                    className="ml-4 bg-gradient-to-r from-primary to-secondary hover:from-primary/90 hover:to-secondary/90 text-primary-foreground font-semibold shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105 px-6 py-3 rounded-xl"
                    onClick={() => {
                      if (file.vrUrl) {
                        router.push(`/preview?videoUrl=${encodeURIComponent(file.vrUrl)}&title=${encodeURIComponent(file.name)}`)
                      }
                    }}
                  >
                    <Play className="w-5 h-5 mr-2" />
                    Preview VR
                  </Button>
                )}
                
                {file.status === "completed" && !file.vrUrl && (
                  <Button
                    size="lg"
                    className="ml-4 bg-gradient-to-r from-green-500 to-green-600 hover:from-green-600 hover:to-green-700 text-white font-semibold shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105 px-6 py-3 rounded-xl"
                    onClick={() => window.open(`${API_BASE_URL}/download/${file.jobId}`, '_blank')}
                  >
                    <Play className="w-5 h-5 mr-2" />
                    Download VR180
                  </Button>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Quick Tips */}
      <Card className="bg-gradient-to-br from-card/60 to-card/40 backdrop-blur-sm shadow-xl border-border/50">
        <CardHeader>
          <CardTitle className="text-xl font-bold text-card-foreground flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-primary" />
            Tips for Best Results
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-card-foreground">
          <div className="flex items-start gap-4">
            <div className="w-3 h-3 bg-gradient-to-r from-primary to-secondary rounded-full mt-2 flex-shrink-0 shadow-md" />
            <p className="text-base font-medium">Use high-resolution videos (1080p or higher) for better VR quality</p>
          </div>
          <div className="flex items-start gap-4">
            <div className="w-3 h-3 bg-gradient-to-r from-primary to-secondary rounded-full mt-2 flex-shrink-0 shadow-md" />
            <p className="text-base font-medium">Ensure good lighting and stable footage for optimal conversion</p>
          </div>
          <div className="flex items-start gap-4">
            <div className="w-3 h-3 bg-gradient-to-r from-primary to-secondary rounded-full mt-2 flex-shrink-0 shadow-md" />
            <p className="text-base font-medium">Processing time varies based on video length and complexity</p>
          </div>
        </CardContent>
      </Card>

    </div>
  )
}
