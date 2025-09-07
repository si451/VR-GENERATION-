"use client"

import type React from "react"
import { useRef, useState, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Slider } from "@/components/ui/slider"
import { Card, CardContent } from "@/components/ui/card"
import { Play, Pause, Volume2, VolumeX, Maximize, Download, X, RotateCcw, Settings, Eye } from "lucide-react"

interface VRPreviewProps {
  videoUrl: string
  title: string
  onDownload?: () => void
  onClose?: () => void
}

const formatTime = (seconds: number) => {
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = Math.floor(seconds % 60)
  return `${minutes.toString().padStart(2, "0")}:${remainingSeconds.toString().padStart(2, "0")}`
}

export function VRPreview({ videoUrl, title, onDownload, onClose }: VRPreviewProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(0.5)
  const [isMuted, setIsMuted] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showControls, setShowControls] = useState(true)
  const [isLoading, setIsLoading] = useState(true)
  const [playbackRate, setPlaybackRate] = useState(1)
  const [videoError, setVideoError] = useState<string | null>(null)

  // If no video URL provided, show a message
  if (!videoUrl || videoUrl.trim() === '') {
    return (
      <div className="w-full h-full flex items-center justify-center bg-black/50 rounded-xl">
        <div className="text-center text-white p-8">
          <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center mx-auto mb-4">
            <X className="w-8 h-8 text-red-400" />
          </div>
          <h3 className="text-xl font-semibold mb-2">No Video Selected</h3>
          <p className="text-gray-300 mb-4">
            Please go to "My Projects" and select a completed VR video to preview.
          </p>
          {onClose && (
            <Button
              onClick={onClose}
              className="bg-primary hover:bg-primary/80 text-white"
            >
              Go Back
            </Button>
          )}
        </div>
      </div>
    )
  }

  const handlePlayPause = useCallback(() => {
    const video = videoRef.current
    if (video) {
      if (isPlaying) {
        video.pause()
      } else {
        video.play().catch(e => console.error("Error playing video:", e))
      }
      setIsPlaying(!isPlaying)
    }
  }, [isPlaying])

  const handleSeek = useCallback((value: number) => {
    const video = videoRef.current
    if (video) {
      video.currentTime = value
      setCurrentTime(value)
    }
  }, [])

  const handleVolumeChange = useCallback((value: number) => {
    const video = videoRef.current
    if (video) {
      video.volume = value
      setVolume(value)
      if (value > 0) setIsMuted(false)
    }
  }, [])

  const handleToggleMute = useCallback(() => {
    const video = videoRef.current
    if (video) {
      video.muted = !isMuted
      setIsMuted(!isMuted)
      if (!isMuted) setVolume(0)
      else setVolume(0.5)
    }
  }, [isMuted])

  const handleToggleFullscreen = useCallback(() => {
    const container = containerRef.current
    if (container) {
      if (!isFullscreen) {
        if (container.requestFullscreen) {
          container.requestFullscreen()
        } else if ((container as any).webkitRequestFullscreen) {
          (container as any).webkitRequestFullscreen()
        } else if ((container as any).msRequestFullscreen) {
          (container as any).msRequestFullscreen()
        }
      } else {
        if (document.exitFullscreen) {
          document.exitFullscreen()
        } else if ((document as any).webkitExitFullscreen) {
          (document as any).webkitExitFullscreen()
        } else if ((document as any).msExitFullscreen) {
          (document as any).msExitFullscreen()
        }
      }
    }
  }, [isFullscreen])

  const handlePlaybackRateChange = useCallback((rate: number) => {
    const video = videoRef.current
    if (video) {
      video.playbackRate = rate
      setPlaybackRate(rate)
    }
  }, [])

  const handleRestart = useCallback(() => {
    const video = videoRef.current
    if (video) {
      video.currentTime = 0
      setCurrentTime(0)
    }
  }, [])

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    console.log("Setting video source:", videoUrl)
    
    // Clear any existing source
    video.src = ""
    video.load()
    
    // Set new source
    video.src = videoUrl
    video.crossOrigin = "anonymous"
    video.loop = true
    video.muted = false
    video.playsInline = true
    video.preload = "metadata"
    
    // Try to load the video
    video.load()

    const updateProgress = () => setCurrentTime(video.currentTime)
    const updateDuration = () => setDuration(video.duration)
    const handlePlay = () => {
      setIsPlaying(true)
      setIsLoading(false)
    }
    const handlePause = () => setIsPlaying(false)
    const handleVolume = () => {
      setVolume(video.volume)
      setIsMuted(video.muted)
    }
    const handleLoadedData = () => {
      setIsLoading(false)
    }
    const handleCanPlay = () => {
      setIsLoading(false)
    }
    const handleError = () => {
      setIsLoading(false)
      console.error("Video failed to load")
    }

    video.addEventListener("timeupdate", updateProgress)
    video.addEventListener("loadedmetadata", updateDuration)
    video.addEventListener("loadeddata", handleLoadedData)
    video.addEventListener("canplay", handleCanPlay)
    video.addEventListener("play", handlePlay)
    video.addEventListener("pause", handlePause)
    video.addEventListener("volumechange", handleVolume)
    video.addEventListener("error", handleError)

    // Don't try to autoplay - let user control it
    // video.play().catch(error => {
    //   console.warn("Autoplay prevented:", error)
    //   setIsPlaying(false)
    //   setIsLoading(false)
    // })

    return () => {
      video.removeEventListener("timeupdate", updateProgress)
      video.removeEventListener("loadedmetadata", updateDuration)
      video.removeEventListener("loadeddata", handleLoadedData)
      video.removeEventListener("canplay", handleCanPlay)
      video.removeEventListener("play", handlePlay)
      video.removeEventListener("pause", handlePause)
      video.removeEventListener("volumechange", handleVolume)
      video.removeEventListener("error", handleError)
    }
  }, [videoUrl])

  // Handle fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }

    document.addEventListener("fullscreenchange", handleFullscreenChange)
    document.addEventListener("webkitfullscreenchange", handleFullscreenChange)
    document.addEventListener("msfullscreenchange", handleFullscreenChange)

    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange)
      document.removeEventListener("webkitfullscreenchange", handleFullscreenChange)
      document.removeEventListener("msfullscreenchange", handleFullscreenChange)
    }
  }, [])

  // Auto-hide controls
  useEffect(() => {
    let timeout: NodeJS.Timeout
    const resetTimeout = () => {
      clearTimeout(timeout)
      setShowControls(true)
      timeout = setTimeout(() => setShowControls(false), 3000)
    }

    if (isPlaying) {
      resetTimeout()
    }

    return () => clearTimeout(timeout)
  }, [isPlaying])

  return (
    <div 
      ref={containerRef}
      className="relative w-full h-full bg-black rounded-xl overflow-hidden flex flex-col group"
      onMouseMove={() => setShowControls(true)}
      onMouseLeave={() => setShowControls(false)}
    >
      {/* Loading overlay */}
      {isLoading && !videoError && (
        <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-20">
          <div className="flex flex-col items-center gap-4">
            <div className="w-12 h-12 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
            <p className="text-white text-lg font-semibold">Loading VR video...</p>
          </div>
        </div>
      )}

      {/* Error overlay */}
      {videoError && (
        <div className="absolute inset-0 bg-black/80 flex items-center justify-center z-20">
          <div className="flex flex-col items-center gap-4 text-center p-6">
            <div className="w-16 h-16 bg-red-500/20 rounded-full flex items-center justify-center">
              <X className="w-8 h-8 text-red-400" />
            </div>
            <div>
              <p className="text-white text-lg font-semibold mb-2">Video Loading Error</p>
              <p className="text-gray-300 text-sm mb-4">{videoError}</p>
              <p className="text-gray-400 text-xs">Video URL: {videoUrl}</p>
            </div>
            <Button
              onClick={() => {
                setVideoError(null)
                setIsLoading(true)
                const video = videoRef.current
                if (video) {
                  video.load()
                }
              }}
              className="bg-primary hover:bg-primary/80 text-white"
            >
              Retry
            </Button>
          </div>
        </div>
      )}

      {/* Top bar with VR 180 badge, title, download, and close */}
      <div className={`absolute top-0 left-0 right-0 p-4 z-10 flex items-center justify-between bg-gradient-to-b from-black/70 to-transparent transition-opacity duration-300 ${showControls ? 'opacity-100' : 'opacity-0'}`}>
        <div className="flex items-center gap-3">
          <Badge className="bg-primary/90 text-white px-4 py-2 text-sm font-semibold rounded-full">
            <Eye className="w-4 h-4 mr-2" />
            VR 180°
          </Badge>
          <h2 className="text-white font-bold text-xl truncate">{title}</h2>
        </div>
        <div className="flex items-center gap-3">
          <Button
            size="sm"
            variant="ghost"
            onClick={handleRestart}
            className="text-white hover:bg-white/20 rounded-xl"
            title="Restart video"
          >
            <RotateCcw className="h-4 w-4" />
          </Button>
          {onDownload && (
            <Button
              size="sm"
              variant="ghost"
              onClick={onDownload}
              className="text-white hover:bg-white/20 rounded-xl"
              title="Download video"
            >
              <Download className="h-4 w-4" />
            </Button>
          )}
          {onClose && (
            <Button
              size="sm"
              variant="ghost"
              onClick={onClose}
              className="text-white hover:bg-white/20 rounded-xl"
              title="Close preview"
            >
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      {/* Video element */}
      <video
        ref={videoRef}
        className="w-full h-full object-contain"
        onClick={handlePlayPause}
        crossOrigin="anonymous"
        loop
        playsInline
        preload="metadata"
        onError={(e) => {
          console.error("Video error:", e)
          console.error("Video error details:", e.currentTarget.error)
          setIsLoading(false)
          setVideoError("Failed to load video. Please check the console for details.")
        }}
        onLoadStart={() => {
          console.log("Video load started")
          setIsLoading(true)
        }}
        onCanPlay={() => {
          console.log("Video can play")
          setIsLoading(false)
        }}
        onLoadedData={() => {
          console.log("Video data loaded")
          setIsLoading(false)
        }}
        onLoadedMetadata={() => {
          console.log("Video metadata loaded")
        }}
        onProgress={() => {
          console.log("Video loading progress")
        }}
      >
        <source src={videoUrl} type="video/mp4" />
        <source src={videoUrl} type="video/webm" />
        Your browser does not support the video tag.
      </video>

      {/* Play button overlay when video is loaded but not playing */}
      {!isLoading && !videoError && !isPlaying && (
        <div className="absolute inset-0 flex items-center justify-center z-10">
          <Button
            size="lg"
            onClick={handlePlayPause}
            className="bg-primary/90 hover:bg-primary text-white rounded-full w-20 h-20 shadow-2xl hover:shadow-3xl transition-all duration-300 transform hover:scale-110"
          >
            <Play className="h-8 w-8" />
          </Button>
        </div>
      )}

      {/* Bottom controls bar */}
      <div className={`absolute bottom-0 left-0 right-0 p-4 z-10 bg-gradient-to-t from-black/70 to-transparent transition-opacity duration-300 ${showControls ? 'opacity-100' : 'opacity-0'}`}>
        <div className="flex items-center gap-4">
          <Button 
            size="icon" 
            onClick={handlePlayPause} 
            className="bg-primary/90 hover:bg-primary text-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105"
          >
            {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
          </Button>

          <div className="flex-1 flex items-center gap-3">
            <span className="text-sm text-white font-semibold min-w-[40px]">{formatTime(currentTime)}</span>
            <Slider
              value={[currentTime]}
              max={duration}
              step={1}
              onValueChange={([value]) => handleSeek(value)}
              className="flex-1"
            />
            <span className="text-sm text-white font-semibold min-w-[40px]">{formatTime(duration)}</span>
          </div>

          <div className="flex items-center gap-3">
            <Button 
              size="icon" 
              variant="ghost" 
              onClick={handleToggleMute} 
              className="text-white hover:bg-white/20 rounded-xl"
              title={isMuted ? "Unmute" : "Mute"}
            >
              {isMuted || volume === 0 ? <VolumeX className="h-5 w-5" /> : <Volume2 className="h-5 w-5" />}
            </Button>
            <Slider
              value={[isMuted ? 0 : volume]}
              max={1}
              step={0.01}
              onValueChange={([value]) => handleVolumeChange(value)}
              className="w-24"
            />
            
            {/* Playback speed selector */}
            <div className="flex items-center gap-1">
              <Button
                size="sm"
                variant="ghost"
                onClick={() => handlePlaybackRateChange(0.5)}
                className={`text-white hover:bg-white/20 rounded-lg text-xs ${playbackRate === 0.5 ? 'bg-white/20' : ''}`}
                title="0.5x Speed"
              >
                0.5x
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => handlePlaybackRateChange(1)}
                className={`text-white hover:bg-white/20 rounded-lg text-xs ${playbackRate === 1 ? 'bg-white/20' : ''}`}
                title="1x Speed"
              >
                1x
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => handlePlaybackRateChange(1.5)}
                className={`text-white hover:bg-white/20 rounded-lg text-xs ${playbackRate === 1.5 ? 'bg-white/20' : ''}`}
                title="1.5x Speed"
              >
                1.5x
              </Button>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => handlePlaybackRateChange(2)}
                className={`text-white hover:bg-white/20 rounded-lg text-xs ${playbackRate === 2 ? 'bg-white/20' : ''}`}
                title="2x Speed"
              >
                2x
              </Button>
            </div>

            <Button 
              size="icon" 
              variant="ghost" 
              onClick={handleToggleFullscreen} 
              className="text-white hover:bg-white/20 rounded-xl"
              title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
            >
              <Maximize className="h-5 w-5" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

export function VRPreviewCard({
  title,
  thumbnail,
  duration,
  onPreview,
}: {
  title: string
  thumbnail: string
  duration: string
  onPreview: () => void
}) {
  return (
    <Card className="group cursor-pointer hover:shadow-lg transition-all duration-200" onClick={onPreview}>
      <CardContent className="p-0">
        <div className="relative aspect-video overflow-hidden rounded-t-lg">
          <img
            src={thumbnail || "/placeholder.svg"}
            alt={title}
            loading="lazy"
            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-200"
          />
          <div className="absolute inset-0 bg-black/20 group-hover:bg-black/10 transition-colors" />
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-16 h-16 bg-primary/90 rounded-full flex items-center justify-center group-hover:bg-primary transition-colors">
              <Play className="w-6 h-6 text-primary-foreground ml-1" />
            </div>
          </div>
          <Badge className="absolute top-2 left-2 bg-primary text-primary-foreground">VR 180°</Badge>
          <Badge variant="secondary" className="absolute bottom-2 right-2">
            {duration}
          </Badge>
        </div>
        <div className="p-4">
          <h3 className="font-semibold text-foreground group-hover:text-primary transition-colors">{title}</h3>
          <p className="text-sm text-muted-foreground mt-1">Click to preview in VR</p>
        </div>
      </CardContent>
    </Card>
  )
}
