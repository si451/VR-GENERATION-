import { ProcessingDashboard } from "@/components/processing-dashboard"
import { ThreeDBackground } from "@/components/3d-background"
import { Header } from "@/components/header"

export default function ProjectsPage() {
  return (
    <div className="relative min-h-screen bg-black overflow-hidden">
      <ThreeDBackground />
      <Header />

      <div className="relative z-10 min-h-screen flex flex-col">
        <main className="flex-1 p-6">
          <div className="max-w-6xl mx-auto">
            <div className="text-center mb-8">
              <h1 className="text-4xl md:text-6xl font-bold text-white mb-4">My VR Projects</h1>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                Manage and track your 2D to VR conversions. Preview, download, and share your immersive experiences.
              </p>
            </div>

            <ProcessingDashboard />
          </div>
        </main>
      </div>
    </div>
  )
}
