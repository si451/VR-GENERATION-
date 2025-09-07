"use client"

import { usePathname, useRouter } from "next/navigation"
import { ArrowLeft } from "lucide-react"
import { Button } from "@/components/ui/button"

export function Header() {
  const pathname = usePathname()
  const router = useRouter()
  const isHomePage = pathname === "/"

  const handleBack = () => {
    router.back()
  }

  return (
    <header className="relative z-20">
      <div className="flex h-24 items-center justify-between px-8">
        <div className="flex items-center gap-3">
          {!isHomePage && (
            <Button
              onClick={handleBack}
              variant="ghost"
              size="icon"
              className="text-white hover:bg-white/20 mr-2 drop-shadow-lg bg-black/20 backdrop-blur-sm"
            >
              <ArrowLeft className="h-5 w-5" />
            </Button>
          )}
          <div className="w-12 h-12 bg-primary rounded-xl flex items-center justify-center shadow-lg animate-glow">
            <span className="text-white font-black text-lg">VR</span>
          </div>
          <div>
            <h1 className="text-2xl font-black text-white text-display drop-shadow-lg">VR Transform</h1>
            <p className="text-xs text-white/90 font-medium text-caption drop-shadow-md">Next-Gen Video Conversion</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="text-sm font-bold text-white text-display drop-shadow-lg">Transform 2D videos to VR 180°</p>
            
          </div>
        </div>
      </div>
    </header>
  )
}
