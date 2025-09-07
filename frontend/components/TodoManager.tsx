"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"

export function TodoManager() {
  const [isComplete, setIsComplete] = useState(false)

  const handleComplete = () => {
    setIsComplete(true)
  }

  return (
    <div className="p-4 border border-border rounded-lg">
      <h3 className="font-semibold mb-2">VR Platform Development</h3>
      <p className="text-sm text-muted-foreground mb-4">All major components have been successfully implemented</p>
      {!isComplete ? (
        <Button onClick={handleComplete}>Mark Project Complete</Button>
      ) : (
        <div className="text-green-600 font-semibold">✓ Project Complete</div>
      )}
    </div>
  )
}
