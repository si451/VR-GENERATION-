import { UploadInterface } from "@/components/upload-interface"
import { ThreeDBackground } from "@/components/3d-background"
import { Header } from "@/components/header"

export default function UploadPage() {
  return (
    <div className="relative min-h-screen bg-black overflow-hidden">
      <ThreeDBackground />
      <Header />

      <div className="relative z-10 min-h-screen flex flex-col">
        <main className="flex-1 p-6">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-8">
              <h1 className="text-4xl md:text-6xl font-bold text-white mb-4">Upload Your 2D Videos</h1>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                Transform your 2D movie clips into immersive VR 180 experiences with our advanced AI technology.
              </p>
            </div>

            <UploadInterface />
          </div>
        </main>
      </div>
    </div>
  )
}
