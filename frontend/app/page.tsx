import { UploadInterface } from "@/components/upload-interface"
import { Header } from "@/components/header"
import { Sidebar } from "@/components/sidebar"
import { ThreeDBackground } from "@/components/3d-background"
import { Upload, FolderOpen, Eye, Sparkles } from "lucide-react"
import Link from "next/link"

export default function HomePage() {
  return (
    <div className="min-h-screen relative">
      <ThreeDBackground />

      <div className="relative z-20">
        <Header />
      </div>

      <div className="flex relative z-10">
        <Sidebar />
        <main className="flex-1 p-6">
          <div className="max-w-6xl mx-auto">
            <div className="mb-12">
              <div className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-3xl p-8 md:p-12 shadow-2xl">
                <h1 className="text-4xl md:text-5xl lg:text-6xl font-black mb-6 text-balance leading-tight text-display">
                  <span className="bg-gradient-to-r from-white via-gray-100 to-gray-300 bg-clip-text text-transparent drop-shadow-lg">
                    Welcome to{" "}
                  </span>
                  <span className="bg-gradient-to-r from-primary via-emerald-400 to-cyan-400 bg-clip-text text-transparent drop-shadow-lg animate-pulse">
                    VR Transform
                  </span>
                </h1>
                <p className="text-lg md:text-xl text-pretty leading-relaxed font-light text-body max-w-4xl">
                  <span className="bg-gradient-to-r from-gray-200 via-white to-gray-200 bg-clip-text text-transparent font-medium">
                    Transform your 2D movie clips into stunning{" "}
                  </span>
                  <span className="bg-gradient-to-r from-primary via-emerald-400 to-cyan-400 bg-clip-text text-transparent font-bold drop-shadow-md">
                    VR 180 experiences
                  </span>
                  <span className="bg-gradient-to-r from-gray-200 via-white to-gray-200 bg-clip-text text-transparent font-medium">
                    {" "}with our advanced AI technology.
                  </span>
                  <br />
                  <span className="text-gray-300 font-normal text-base mt-2 block">
                    Get started by uploading your videos or explore your existing projects.
                  </span>
                </p>
              </div>
            </div>

            <div className="grid md:grid-cols-3 gap-6 mb-12">
              <Link href="/upload">
                <div className="bg-black/15 backdrop-blur-xl border border-white/5 rounded-2xl p-6 hover:bg-black/25 hover:border-white/20 transition-all duration-500 group cursor-pointer transform hover:scale-105">
                  <div className="flex items-center gap-4 mb-4">
                    <div className="p-3 bg-primary/20 rounded-xl group-hover:bg-primary/30 transition-all duration-300 group-hover:scale-110">
                      <Upload className="h-6 w-6 text-primary" />
                    </div>
                    <h3 className="text-xl font-bold text-white text-display">Upload Videos</h3>
                  </div>
                  <p className="text-gray-300 text-base font-light text-body leading-relaxed">
                    Upload your 2D movie clips and start the VR conversion process.
                  </p>
                </div>
              </Link>

              <Link href="/projects">
                <div className="bg-black/15 backdrop-blur-xl border border-white/5 rounded-2xl p-6 hover:bg-black/25 hover:border-white/20 transition-all duration-500 group cursor-pointer transform hover:scale-105">
                  <div className="flex items-center gap-4 mb-4">
                    <div className="p-3 bg-primary/20 rounded-xl group-hover:bg-primary/30 transition-all duration-300 group-hover:scale-110">
                      <FolderOpen className="h-6 w-6 text-primary" />
                    </div>
                    <h3 className="text-xl font-bold text-white text-display">My Projects</h3>
                  </div>
                  <p className="text-gray-300 text-base font-light text-body leading-relaxed">
                    Manage and track your VR conversion projects and downloads.
                  </p>
                </div>
              </Link>

              <Link href="/preview">
                <div className="bg-black/15 backdrop-blur-xl border border-white/5 rounded-2xl p-6 hover:bg-black/25 hover:border-white/20 transition-all duration-500 group cursor-pointer transform hover:scale-105">
                  <div className="flex items-center gap-4 mb-4">
                    <div className="p-3 bg-primary/20 rounded-xl group-hover:bg-primary/30 transition-all duration-300 group-hover:scale-110">
                      <Eye className="h-6 w-6 text-primary" />
                    </div>
                    <h3 className="text-xl font-bold text-white text-display">VR Preview</h3>
                  </div>
                  <p className="text-gray-300 text-base font-light text-body leading-relaxed">
                    Experience your converted VR content in immersive preview mode.
                  </p>
                </div>
              </Link>
            </div>

            <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-8 shadow-2xl">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-3 bg-gradient-to-r from-primary/20 to-emerald-400/20 rounded-xl shadow-lg">
                  <Sparkles className="h-6 w-6 text-primary drop-shadow-md" />
                </div>
                <h2 className="text-2xl font-black text-display">
                  <span className="bg-gradient-to-r from-white via-gray-100 to-gray-300 bg-clip-text text-transparent drop-shadow-lg">
                    Platform{" "}
                  </span>
                  <span className="bg-gradient-to-r from-primary via-emerald-400 to-cyan-400 bg-clip-text text-transparent drop-shadow-lg">
                    Features
                  </span>
                </h2>
              </div>
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="w-3 h-3 bg-gradient-to-r from-primary to-emerald-400 rounded-full mt-2 flex-shrink-0 shadow-lg"></div>
                    <div>
                      <h4 className="bg-gradient-to-r from-white to-gray-200 bg-clip-text text-transparent font-bold text-lg text-display mb-1 drop-shadow-md">AI-Powered Conversion</h4>
                      <p className="text-gray-300 text-sm font-light text-body leading-relaxed">
                        Advanced algorithms transform <span className="bg-gradient-to-r from-primary to-emerald-400 bg-clip-text text-transparent font-semibold">2D content</span> into immersive VR
                        180 experiences
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-3 h-3 bg-gradient-to-r from-primary to-emerald-400 rounded-full mt-2 flex-shrink-0 shadow-lg"></div>
                    <div>
                      <h4 className="bg-gradient-to-r from-white to-gray-200 bg-clip-text text-transparent font-bold text-lg text-display mb-1 drop-shadow-md">Multiple Format Support</h4>
                      <p className="text-gray-300 text-sm font-light text-body leading-relaxed">
                        Compatible with <span className="bg-gradient-to-r from-primary to-emerald-400 bg-clip-text text-transparent font-semibold">MP4, MOV, AVI, and MKV</span> video
                        formats
                      </p>
                    </div>
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="w-3 h-3 bg-gradient-to-r from-primary to-emerald-400 rounded-full mt-2 flex-shrink-0 shadow-lg"></div>
                    <div>
                      <h4 className="bg-gradient-to-r from-white to-gray-200 bg-clip-text text-transparent font-bold text-lg text-display mb-1 drop-shadow-md">VR Headset Compatible</h4>
                      <p className="text-gray-300 text-sm font-light text-body leading-relaxed">
                        Works with all major <span className="bg-gradient-to-r from-primary to-emerald-400 bg-clip-text text-transparent font-semibold">VR headsets</span> and platforms
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-3 h-3 bg-gradient-to-r from-primary to-emerald-400 rounded-full mt-2 flex-shrink-0 shadow-lg"></div>
                    <div>
                      <h4 className="bg-gradient-to-r from-white to-gray-200 bg-clip-text text-transparent font-bold text-lg text-display mb-1 drop-shadow-md">Real-time Preview</h4>
                      <p className="text-gray-300 text-sm font-light text-body leading-relaxed">
                        Preview your <span className="bg-gradient-to-r from-primary to-emerald-400 bg-clip-text text-transparent font-semibold">VR content</span> before downloading
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="relative">
              <UploadInterface />
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
