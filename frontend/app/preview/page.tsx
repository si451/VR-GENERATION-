"use client"

import { useSearchParams } from "next/navigation"
import { VRPreview } from "@/components/vr-preview"
import { Header } from "@/components/header"
import { ThreeDBackground } from "@/components/3d-background"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft, Play, FolderOpen, Upload } from "lucide-react"
import Link from "next/link"

export default function PreviewPage() {
  const searchParams = useSearchParams()
  const videoUrl = searchParams.get('videoUrl') || ''
  const title = searchParams.get('title') || 'VR Preview'

  const handleDownload = () => {
    if (videoUrl) {
      const link = document.createElement('a')
      link.href = videoUrl
      link.download = title
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
  }

  const handleClose = () => {
    window.history.back()
  }

  // If no video URL provided, show demo and instructions
  if (!videoUrl) {
    return (
      <div className="min-h-screen relative">
        <ThreeDBackground />
        <Header />
        
        <div className="relative z-10 flex items-center justify-center min-h-screen p-4">
          <div className="w-full max-w-4xl">
            <Card className="bg-white/10 backdrop-blur-lg border-white/20 text-white">
              <CardHeader className="text-center">
                <div className="w-20 h-20 bg-primary/20 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Play className="w-10 h-10 text-primary" />
                </div>
                <CardTitle className="text-3xl font-bold text-white mb-2">
                  VR Preview Demo
                </CardTitle>
                <CardDescription className="text-lg text-gray-300">
                  Experience the power of VR 180° video conversion
                </CardDescription>
              </CardHeader>
              
              <CardContent className="space-y-6">
                {/* Demo Video Section */}
                <div className="text-center">
                  <h3 className="text-xl font-semibold text-white mb-4">Try Our Demo Video</h3>
                  <div className="bg-black/20 rounded-xl p-4 mb-4">
                    <VRPreview
                      videoUrl="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
                      title="VR Demo Video"
                      onDownload={() => {}}
                      onClose={() => {}}
                    />
                  </div>
                  <p className="text-sm text-gray-400">
                    This is a sample VR 180° video. Upload your own video to create personalized VR experiences!
                  </p>
                </div>

                {/* Instructions Section */}
                <div className="bg-white/5 rounded-xl p-6">
                  <h3 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                    <FolderOpen className="w-5 h-5" />
                    How to Preview Your Own Videos
                  </h3>
                  <div className="space-y-4">
                    <div className="flex items-start gap-3">
                      <div className="w-6 h-6 bg-primary rounded-full flex items-center justify-center text-white text-sm font-bold flex-shrink-0 mt-0.5">
                        1
                      </div>
                      <div>
                        <p className="text-white font-medium">Upload Your Video</p>
                        <p className="text-gray-300 text-sm">Go to the Upload page and select your 2D video file</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-6 h-6 bg-primary rounded-full flex items-center justify-center text-white text-sm font-bold flex-shrink-0 mt-0.5">
                        2
                      </div>
                      <div>
                        <p className="text-white font-medium">Wait for Processing</p>
                        <p className="text-gray-300 text-sm">Our AI will convert your video to VR 180° format</p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="w-6 h-6 bg-primary rounded-full flex items-center justify-center text-white text-sm font-bold flex-shrink-0 mt-0.5">
                        3
                      </div>
                      <div>
                        <p className="text-white font-medium">Preview in VR</p>
                        <p className="text-gray-300 text-sm">Go to My Projects and click "Preview VR" on completed videos</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex flex-col sm:flex-row gap-4 justify-center">
                  <Link href="/upload">
                    <Button className="bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 text-white font-semibold shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105 px-8 py-3">
                      <Upload className="w-5 h-5 mr-2" />
                      Upload Video
                    </Button>
                  </Link>
                  <Link href="/projects">
                    <Button variant="outline" className="border-white/30 hover:border-white hover:bg-white/10 text-white hover:text-white transition-all duration-300 px-8 py-3">
                      <FolderOpen className="w-5 h-5 mr-2" />
                      My Projects
                    </Button>
                  </Link>
                  <Button 
                    variant="ghost" 
                    onClick={handleClose}
                    className="text-gray-300 hover:text-white hover:bg-white/10 transition-all duration-300 px-8 py-3"
                  >
                    <ArrowLeft className="w-5 h-5 mr-2" />
                    Go Back
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    )
  }

  // If video URL is provided, show the VR preview
  return (
    <div className="min-h-screen relative">
      <ThreeDBackground />
      <Header />
      
      <div className="relative z-10 flex items-center justify-center min-h-screen p-2">
        <div className="w-full h-screen max-w-7xl">
          <VRPreview
            videoUrl={videoUrl}
            title={title}
            onDownload={handleDownload}
            onClose={handleClose}
          />
        </div>
      </div>
    </div>
  )
}