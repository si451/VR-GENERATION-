"use client"

import { useEffect, useRef, useState } from "react"

export function ThreeDBackground() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isLoaded, setIsLoaded] = useState(false)
  const [hasError, setHasError] = useState(false)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    // Start playing immediately
    const playVideo = () => {
      video.play().catch((error) => {
        console.warn("Video autoplay failed:", error)
        setHasError(true)
      })
    }

    // Try to play immediately
    playVideo()

    const handleCanPlay = () => {
      playVideo()
      setIsLoaded(true)
    }

    const handleError = () => {
      console.error("Video failed to load")
      setHasError(true)
    }

    const handleLoadedData = () => {
      setIsLoaded(true)
      playVideo()
    }

    const handlePlay = () => {
      setIsLoaded(true)
    }

    const handleEnded = () => {
      video.currentTime = 0
      video.play().catch(() => {}) // Silent fail
    }

    const handleTimeUpdate = () => {
      // Ensure seamless looping by restarting before the end
      if (video.duration > 0 && video.currentTime >= video.duration - 0.1) {
        video.currentTime = 0
      }
    }

    video.addEventListener("canplay", handleCanPlay)
    video.addEventListener("error", handleError)
    video.addEventListener("loadeddata", handleLoadedData)
    video.addEventListener("play", handlePlay)
    video.addEventListener("ended", handleEnded)
    video.addEventListener("timeupdate", handleTimeUpdate)

    return () => {
      video.removeEventListener("canplay", handleCanPlay)
      video.removeEventListener("error", handleError)
      video.removeEventListener("loadeddata", handleLoadedData)
      video.removeEventListener("play", handlePlay)
      video.removeEventListener("ended", handleEnded)
      video.removeEventListener("timeupdate", handleTimeUpdate)
    }
  }, [])

  return (
    <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
      {/* Saturn Background Video - Always visible */}
      <video
        ref={videoRef}
        className="w-full h-full object-cover"
        autoPlay
        loop
        muted
        playsInline
        crossOrigin="anonymous"
        preload="auto"
        style={{ 
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          objectFit: 'cover'
        }}
        onEnded={(e) => {
          // Ensure seamless looping
          e.currentTarget.currentTime = 0;
          e.currentTarget.play();
        }}
      >
        <source src="/saturn-background.mp4" type="video/mp4" />
        <source src="/placeholder.mp4" type="video/mp4" />
      </video>

      {/* Fallback Background - Only show if video completely fails */}
      {hasError && (
        <div className="absolute inset-0 bg-black">
          <div className="absolute inset-0 bg-[url('/saturn-with-rings.png')] bg-center bg-no-repeat bg-contain opacity-30" />
        </div>
      )}

      {/* Simple Overlay for text readability - minimal */}
      <div className="absolute inset-0 bg-black/20" />
    </div>
  )
}
