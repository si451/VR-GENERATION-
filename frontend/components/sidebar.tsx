"use client"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { Upload, FolderOpen, Eye, Home } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"

const sidebarItems = [
  { icon: Home, label: "Dashboard", href: "/", active: false },
  { icon: Upload, label: "Upload", href: "/upload", active: false },
  { icon: FolderOpen, label: "My Projects", href: "/projects", active: false },
  { icon: Eye, label: "VR Preview", href: "/preview", active: false },
]

export function Sidebar() {
  const pathname = usePathname()

  return (
    <aside className="w-64 relative z-10">
      <div className="p-6">
        <nav className="space-y-2">
          {sidebarItems.map((item) => {
            const isActive = pathname === item.href || (item.href === "/" && pathname === "/")
            return (
              <Link key={item.label} href={item.href}>
                <Button
                  variant={isActive ? "default" : "ghost"}
                  className={cn(
                    "w-full justify-start gap-3 transition-all duration-300 font-medium text-base",
                    isActive
                      ? "bg-primary/95 text-white shadow-2xl backdrop-blur-lg"
                      : "text-white/90 hover:bg-white/20 hover:text-white backdrop-blur-lg bg-black/30",
                  )}
                >
                  <item.icon className="h-5 w-5" />
                  <span className="font-semibold">{item.label}</span>
                </Button>
              </Link>
            )
          })}
        </nav>
      </div>
    </aside>
  )
}
