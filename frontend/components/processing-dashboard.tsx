"use client"

import { useState, useMemo, useCallback, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { VRPreview } from "@/components/vr-preview"
import {
  Play,
  Download,
  Trash2,
  Clock,
  CheckCircle,
  AlertCircle,
  MoreHorizontal,
  Eye,
  Share2,
  Settings,
  X,
} from "lucide-react"
import { cn } from "@/lib/utils"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface Project {
  job_id: string
  status: "processing" | "done" | "failed" | "queued"
  stage: string
  percent: number
  message: string
  output?: string
  output_filename?: string
  file_size: number
  created_at: string
  input_file: string
}

const vrExample = {
  id: "example-1",
  name: "Ocean Waves VR Experience",
  description: "Immersive 180° ocean waves captured at sunset",
  thumbnail: "/beach-sunset.png",
  videoUrl: "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
  duration: "2:34",
  quality: "4K"
}

export function ProcessingDashboard() {
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState("all")
  const [vrPreview, setVrPreview] = useState<{ isOpen: boolean; project: Project | null; example?: any }>({
    isOpen: false,
    project: null,
  })
  const [selectedProject, setSelectedProject] = useState<Project | null>(null)

  // Fetch projects from API
  const fetchProjects = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/jobs`)
      if (response.ok) {
        const data = await response.json()
        setProjects(data.jobs || [])
      }
    } catch (error) {
      console.error('Failed to fetch projects:', error)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchProjects()
    // Refresh every 5 seconds to get updates
    const interval = setInterval(fetchProjects, 5000)
    return () => clearInterval(interval)
  }, [fetchProjects])

  const statusConfig = useMemo(() => ({
    done: { icon: <CheckCircle className="h-4 w-4 text-green-500" />, text: "Ready", color: "bg-green-500" },
    processing: { icon: <Clock className="h-4 w-4 text-blue-500" />, text: "Processing", color: "bg-blue-500" },
    queued: { icon: <Clock className="h-4 w-4 text-yellow-500" />, text: "Queued", color: "bg-yellow-500" },
    failed: { icon: <AlertCircle className="h-4 w-4 text-red-500" />, text: "Failed", color: "bg-red-500" }
  }), [])

  const getStatusIcon = useCallback((status: Project["status"]) => statusConfig[status]?.icon, [statusConfig])
  const getStatusText = useCallback((status: Project["status"]) => statusConfig[status]?.text, [statusConfig])
  const getStatusColor = useCallback((status: Project["status"]) => statusConfig[status]?.color, [statusConfig])

  const filteredProjects = projects.filter((project) => {
    // Only show projects that have completed VR output, not input videos
    if (project.status !== "done" || !project.output) return false
    
    if (activeTab === "all") return true
    if (activeTab === "completed") return project.status === "done"
    return project.status === activeTab
  })

  const openVRPreview = useCallback((project: Project) => {
    if (project.output) {
      // Navigate to VR preview page with the project data
      const params = new URLSearchParams({
        videoUrl: `${API_BASE_URL}/download/${project.job_id}`,
        title: project.output_filename || project.input_file
      })
      window.location.href = `/preview?${params.toString()}`
    }
  }, [])

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const formatDate = (dateString: string) => {
    if (!dateString) return 'Unknown'
    try {
      return new Date(dateString).toLocaleDateString()
    } catch {
      return 'Unknown'
    }
  }

  const openExamplePreview = useCallback(() => {
    setVrPreview({ isOpen: true, project: null, example: vrExample })
  }, [])

  const closeVRPreview = useCallback(() => {
    setVrPreview({ isOpen: false, project: null })
  }, [])

  const showProjectDetails = useCallback((project: Project) => {
    setSelectedProject(project)
  }, [])

  const closeProjectDetails = useCallback(() => {
    setSelectedProject(null)
  }, [])

  const deleteProject = useCallback(async (project: Project) => {
    if (!confirm(`Are you sure you want to delete this project? This action cannot be undone.`)) {
      return
    }

    try {
      // For now, we'll just remove it from the local state
      // In a real app, you'd call a delete API endpoint
      setProjects(prev => prev.filter(p => p.job_id !== project.job_id))
      
      // If there's a delete API endpoint, you would call it here:
      // const response = await fetch(`${API_BASE_URL}/jobs/${project.job_id}`, {
      //   method: 'DELETE'
      // })
      // if (response.ok) {
      //   setProjects(prev => prev.filter(p => p.job_id !== project.job_id))
      // }
    } catch (error) {
      console.error('Failed to delete project:', error)
      alert('Failed to delete project. Please try again.')
    }
  }, [])

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">VR Videos</p>
                <p className="text-2xl font-bold text-foreground">
                  {projects.filter((p) => p.status === "done" && p.output).length}
                </p>
              </div>
              <div className="w-8 h-8 bg-primary/10 rounded-lg flex items-center justify-center">
                <Settings className="h-4 w-4 text-primary" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Completed</p>
                <p className="text-2xl font-bold text-green-600">
                  {projects.filter((p) => p.status === "done" && p.output).length}
                </p>
              </div>
              <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                <CheckCircle className="h-4 w-4 text-green-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Processing</p>
                <p className="text-2xl font-bold text-blue-600">
                  {projects.filter((p) => p.status === "processing" || p.status === "queued").length}
                </p>
              </div>
              <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                <Clock className="h-4 w-4 text-blue-600" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Failed</p>
                <p className="text-2xl font-bold text-red-600">
                  {projects.filter((p) => p.status === "failed").length}
                </p>
              </div>
              <div className="w-8 h-8 bg-red-100 rounded-lg flex items-center justify-center">
                <AlertCircle className="h-4 w-4 text-red-600" />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>VR 180 Example</CardTitle>
          <CardDescription>Experience what your converted videos will look like</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="border border-border rounded-lg p-4 hover:bg-muted/50 transition-colors">
            <div className="flex items-start gap-4">
              <div className="w-24 h-16 bg-muted rounded-lg overflow-hidden flex-shrink-0">
                <img
                  src={vrExample.thumbnail || "/placeholder.svg"}
                  alt={vrExample.name}
                  loading="lazy"
                  className="w-full h-full object-cover"
                />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <h3 className="font-semibold text-foreground">{vrExample.name}</h3>
                    <p className="text-sm text-muted-foreground">{vrExample.description}</p>
                  </div>
                  <Badge variant="secondary" className="bg-green-500 text-white">
                    <CheckCircle className="h-4 w-4" />
                    <span className="ml-1">Example</span>
                  </Badge>
                </div>
                <div className="flex items-center gap-4 text-sm text-muted-foreground mb-3">
                  <span>{vrExample.duration}</span>
                  <span>{vrExample.quality}</span>
                  <span>VR 180°</span>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    onClick={openExamplePreview}
                    className="bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 text-white font-medium shadow-lg hover:shadow-xl transition-all duration-200 transform hover:scale-105"
                  >
                    <Play className="h-4 w-4 mr-2" />
                    Preview VR Example
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Projects List */}
      <Card>
        <CardHeader>
          <CardTitle>Your VR Projects</CardTitle>
          <CardDescription>Manage and track your 2D to VR conversions</CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto mb-4">
                <Settings className="h-8 w-8 text-muted-foreground animate-spin" />
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">Loading Projects...</h3>
              <p className="text-muted-foreground">Fetching your VR projects</p>
            </div>
          ) : projects.length === 0 ? (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto mb-4">
                <Settings className="h-8 w-8 text-muted-foreground" />
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">No VR Videos Yet</h3>
              <p className="text-muted-foreground mb-4">
                Upload your first 2D video to start creating VR 180 experiences
              </p>
              <Button className="bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 text-white font-medium shadow-lg hover:shadow-xl transition-all duration-200 transform hover:scale-105 px-6 py-2">
                Upload Your First Video
              </Button>
            </div>
          ) : (
            <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
              <TabsList>
                <TabsTrigger value="all">All Projects</TabsTrigger>
                <TabsTrigger value="completed">Completed</TabsTrigger>
                <TabsTrigger value="processing">Processing</TabsTrigger>
                <TabsTrigger value="failed">Failed</TabsTrigger>
              </TabsList>

              <TabsContent value={activeTab} className="space-y-4">
                {filteredProjects.length === 0 ? (
                  <div className="text-center py-8">
                    <p className="text-muted-foreground">No projects found in this category.</p>
                  </div>
                ) : (
                  <div className="grid gap-4">
                    {filteredProjects.map((project) => (
                      <div
                        key={project.job_id}
                        className="border border-white/20 rounded-2xl p-6 hover:bg-white/5 transition-all duration-300 shadow-lg hover:shadow-xl bg-black/20 backdrop-blur-sm"
                      >
                        <div className="flex items-start gap-4">
                          {/* Thumbnail */}
                          <div className="w-32 h-20 bg-gradient-to-br from-primary/20 to-secondary/20 rounded-xl overflow-hidden flex-shrink-0 flex items-center justify-center relative group">
                            {project.status === "done" && project.output ? (
                              <div className="relative w-full h-full">
                                <video
                                  className="w-full h-full object-cover"
                                  muted
                                  preload="metadata"
                                  onLoadedData={(e) => {
                                    // Seek to 1 second to get a good thumbnail
                                    e.currentTarget.currentTime = 1;
                                  }}
                                >
                                  <source src={`${API_BASE_URL}/download/${project.job_id}`} type="video/mp4" />
                                </video>
                                <div className="absolute inset-0 bg-black/20 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                                  <Play className="w-6 h-6 text-white" />
                                </div>
                              </div>
                            ) : project.status === "processing" || project.status === "queued" ? (
                              <div className="flex flex-col items-center gap-2">
                                <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
                                <div className="text-xs text-primary font-semibold">Processing</div>
                              </div>
                            ) : project.status === "failed" ? (
                              <div className="flex flex-col items-center gap-2">
                                <AlertCircle className="w-8 h-8 text-red-400" />
                                <div className="text-xs text-red-400 font-semibold">Failed</div>
                              </div>
                            ) : (
                              <div className="flex flex-col items-center gap-2">
                                <div className="text-2xl">📁</div>
                                <div className="text-xs text-gray-400 font-semibold">Queued</div>
                              </div>
                            )}
                          </div>

                          {/* Project Info */}
                          <div className="flex-1 min-w-0">
                            <div className="flex items-start justify-between mb-3">
                              <div>
                                <h3 className="font-bold text-white truncate text-lg">
                                  {project.status === "done" && project.output_filename 
                                    ? project.output_filename 
                                    : project.input_file}
                                </h3>
                                <p className="text-base text-gray-300 font-medium">
                                  {project.status === "done" && project.output_filename 
                                    ? "VR180 Video Ready" 
                                    : formatFileSize(project.file_size)}
                                </p>
                              </div>
                              <div className="flex items-center gap-3">
                                <Badge variant="secondary" className={cn("text-white font-semibold px-3 py-1 rounded-xl", getStatusColor(project.status))}>
                                  {getStatusIcon(project.status)}
                                  <span className="ml-2">{getStatusText(project.status)}</span>
                                </Badge>
                                <Button variant="ghost" size="icon" className="h-10 w-10 hover:bg-white/10 rounded-xl">
                                  <MoreHorizontal className="h-5 w-5" />
                                </Button>
                              </div>
                            </div>

                            <div className="flex items-center gap-4 text-base text-gray-400 mb-4">
                              <span className="bg-primary/20 text-primary px-3 py-1 rounded-full text-sm font-semibold">VR 180°</span>
                              <span>{formatDate(project.created_at)}</span>
                              {project.message && <span className="truncate text-gray-300">{project.message}</span>}
                            </div>

                            {/* Progress Bar */}
                            {(project.status === "processing" || project.status === "queued") && (
                              <div className="mb-4">
                                <div className="flex items-center justify-between text-base mb-2">
                                  <span className="text-gray-300 font-semibold">Converting to VR...</span>
                                  <span className="text-primary font-bold">{project.percent}%</span>
                                </div>
                                <Progress value={project.percent} className="h-3 bg-white/10 rounded-full" />
                              </div>
                            )}

                            {/* Action Buttons */}
                            <div className="flex items-center gap-3">
                              {project.status === "done" && project.output && (
                                <>
                                  <Button
                                    size="sm"
                                    onClick={() => openVRPreview(project)}
                                    className="bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 text-white font-semibold shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105 px-6 py-2 rounded-xl"
                                  >
                                    <Play className="h-4 w-4 mr-2" />
                                    Preview VR
                                  </Button>
                                  <Button
                                    size="sm"
                                    variant="outline"
                                    onClick={() => {
                                      if (project.output) {
                                        // Create a temporary link to download the file
                                        const link = document.createElement('a')
                                        link.href = `${API_BASE_URL}/download/${project.job_id}`
                                        link.download = project.output_filename || `${project.job_id}_sbs.mp4`
                                        document.body.appendChild(link)
                                        link.click()
                                        document.body.removeChild(link)
                                      }
                                    }}
                                    className="border-primary/30 hover:border-primary hover:bg-primary/10 text-primary hover:text-primary transition-all duration-300 bg-transparent px-6 py-2 rounded-xl font-semibold"
                                  >
                                    <Download className="h-4 w-4 mr-2" />
                                    Download
                                  </Button>
                                </>
                              )}
                              {project.status === "failed" && (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="border-red-400/30 hover:border-red-400 hover:bg-red-400/10 text-red-400 hover:text-red-300 transition-all duration-300 bg-transparent px-6 py-2 rounded-xl font-semibold"
                                >
                                  Retry Conversion
                                </Button>
                              )}
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => showProjectDetails(project)}
                                className="hover:bg-white/10 text-gray-300 hover:text-white transition-all duration-300 px-4 py-2 rounded-xl"
                              >
                                <Eye className="h-4 w-4 mr-2" />
                                Details
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => deleteProject(project)}
                                className="text-red-400 hover:text-red-300 hover:bg-red-400/10 transition-all duration-300 px-4 py-2 rounded-xl"
                              >
                                <Trash2 className="h-4 w-4 mr-2" />
                                Delete
                              </Button>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </TabsContent>
            </Tabs>
          )}
        </CardContent>
      </Card>

      {/* VR Preview Modal */}
      {vrPreview.isOpen && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full h-full max-w-6xl max-h-[80vh]">
            <VRPreview
              title={vrPreview.example ? vrExample.name : (vrPreview.project?.output_filename || vrPreview.project?.input_file || "")}
              videoUrl={
                vrPreview.example
                  ? vrExample.videoUrl
                  : vrPreview.project?.output 
                    ? `${API_BASE_URL}/download/${vrPreview.project.job_id}`
                    : "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"
              }
              onClose={closeVRPreview}
            />
          </div>
        </div>
      )}

      {/* Project Details Modal */}
      {selectedProject && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full max-w-2xl bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-white">Project Details</h2>
              <Button
                variant="ghost"
                size="icon"
                onClick={closeProjectDetails}
                className="text-white hover:bg-white/20"
              >
                <X className="h-5 w-5" />
              </Button>
            </div>
            
            <div className="space-y-4">
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">File Information</h3>
                <div className="space-y-2 text-gray-300">
                  <p><span className="font-medium">Job ID:</span> {selectedProject.job_id}</p>
                  <p><span className="font-medium">Input File:</span> {selectedProject.input_file}</p>
                  <p><span className="font-medium">Output File:</span> {selectedProject.output_filename || 'N/A'}</p>
                  <p><span className="font-medium">File Size:</span> {formatFileSize(selectedProject.file_size)}</p>
                  <p><span className="font-medium">Created:</span> {formatDate(selectedProject.created_at)}</p>
                </div>
              </div>
              
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">Processing Status</h3>
                <div className="space-y-2 text-gray-300">
                  <p><span className="font-medium">Status:</span> {getStatusText(selectedProject.status)}</p>
                  <p><span className="font-medium">Stage:</span> {selectedProject.stage}</p>
                  <p><span className="font-medium">Progress:</span> {selectedProject.percent}%</p>
                  {selectedProject.message && (
                    <p><span className="font-medium">Message:</span> {selectedProject.message}</p>
                  )}
                </div>
              </div>
              
              {selectedProject.status === "done" && selectedProject.output && (
                <div className="pt-4 border-t border-white/20">
                  <div className="flex gap-3">
                    <Button
                      onClick={() => {
                        closeProjectDetails()
                        openVRPreview(selectedProject)
                      }}
                      className="bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 text-white font-semibold shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105"
                    >
                      <Play className="h-4 w-4 mr-2" />
                      Preview VR
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => {
                        const link = document.createElement('a')
                        link.href = `${API_BASE_URL}/download/${selectedProject.job_id}`
                        link.download = selectedProject.output_filename || `${selectedProject.job_id}_sbs.mp4`
                        document.body.appendChild(link)
                        link.click()
                        document.body.removeChild(link)
                      }}
                      className="border-primary/30 hover:border-primary hover:bg-primary/10 text-primary hover:text-primary transition-all duration-300 bg-transparent"
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Download
                    </Button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
