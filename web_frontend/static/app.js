const videoPlayer = document.querySelector("video")
const inputContainer = document.getElementById("inputContainer")
const userInput = document.getElementById("userInput")
const recordButton = document.getElementById("recordButton")
const micIcon = document.getElementById("micIcon")
const sendArrow = document.getElementById("sendArrow")
const timer = document.getElementById("timer")
const ripple = document.getElementById("ripple")

const recordingTime = parseInt(recordButton.dataset.recordingTime, 10)
userInput.disabled = false
recordButton.disabled = false

const mimeCodec = 'video/mp4; codecs="avc1.42E01E, mp4a.40.2"'

// if (window.MediaSource && MediaSource.isTypeSupported(mimeCodec)) {
//   console.log("MediaSource is supported!")
// } else {
//   // Provide fallback or inform the user about browser incompatibility
//   console.log("MediaSource is not supported!")
// }

let mediaRecorder
let recording = false
let secondsElapsed = 0
let timerInterval
let audioChunks = []

const socket = io({
  transports: ["websocket"],
  transportOptions: { websocket: { perMessageDeflate: false } },
  reconnection: true, // Enable automatic reconnection (default is true)
  reconnectionAttempts: 5, // Maximum number of reconnection attempts
  reconnectionDelay: 1000, // Initial delay between reconnection attempts (in milliseconds)
  reconnectionDelayMax: 10000, // Maximum delay between reconnection attempts
  randomizationFactor: 0.5, // Randomization factor for the delay
})

socket.on("connect", () => {
  socket.emit("client_connected") // Emit the connect event to check for ongoing processing
})

socket.on("processing", (message) => {
  // Inform the user that processing is ongoing
  showToast(message)
})

socket.on("ready", () => {
  console.log("ready for processing!")
})

socket.on("progress", (message) => {
  showToast(`generation: ${message}%`)
})

socket.on("disconnect", (reason) => {
  if (reason === "io server disconnect") {
    // the disconnection was initiated by the server, you need to reconnect manually
    socket.connect()
  }
  console.log("Disconnected:", reason)
})

socket.io.on("reconnect", () => {
  console.log("Reconnected to the server.")
})

socket.io.on("reconnect_attempt", (attemptNumber) => {
  console.log(`Reconnect attempt ${attemptNumber}`)
})

socket.io.on("reconnect_error", (error) => {
  console.log("Reconnect error:", error)
})

socket.io.on("reconnect_failed", () => {
  console.log("Reconnect failed after maximum attempts.")
})

socket.on("connect_error", (error) => {
  console.error("Socket.IO connection error:", error)
})

function showToast(message) {
  const toastMessage = document.getElementById("toastMessage")
  const toastText = document.getElementById("toastText")
  const toastCloseBtn = document.getElementById("toastCloseBtn")

  toastText.textContent = message
  toastMessage.classList.remove("toast-hidden")
  toastMessage.classList.add("toast-show")

  toastCloseBtn.onclick = () => {
    toastMessage.classList.remove("toast-show")
    toastMessage.classList.add("toast-hidden")
  }

  // Hide the toast after 8 seconds
  setTimeout(() => {
    toastMessage.classList.remove("toast-show")
    toastMessage.classList.add("toast-hidden")
  }, 8000)
}

userInput.addEventListener("input", () => {
  sendArrow.style.display = userInput.value.trim() ? "block" : "none"
  if (!userInput.value.trim()) {
    userInput.placeholder = "Mr. President, how are you?."
    clearInterval(timerInterval) // Stop timer if running
    secondsElapsed = 0
  }
})

userInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && userInput.value.trim()) {
    event.preventDefault() // Prevent new line
    sendDataToServer(userInput.value.trim(), "text")
    userInput.value = "" // clear the user input
  }
})

sendArrow.addEventListener("click", () => {
  if (userInput.value.trim()) {
    sendDataToServer(userInput.value.trim(), "text")
    userInput.value = ""
  }
})

function sendDataToServer(data, type) {
  // Create Blob for text data
  if (type == "text") {
    data = new Blob([data], { type: "text/plain;charset=utf-8" })
  }

  // Check if a connection to the server exists
  if (!socket.connected) {
    alert("Not connected to the server. Please try again.")
    return
  }

  userInput.disabled = true
  recordButton.disabled = true

  socket.emit("process_video", {
    data: data,
    dataType: type,
  }) // Emit the event with data
}

let mediaSource,
  sourceBuffer,
  queue = [] // queue to store segments until MediaSource is open

let isProcessingFinished = false // Flag to track processing completion

socket.on("processing", () => {
  // Prepare MediaSource when processing starts
  console.log("processing started, preparing MediaSource.")
  mediaSource = new MediaSource()
  videoPlayer.src = URL.createObjectURL(mediaSource)
  mediaSource.addEventListener("sourceclose", handleSourceClose) // Handle sourceclose
  mediaSource.addEventListener("error", handleSourceError)
  createSourceBuffer()
})

function createSourceBuffer() {
  if (mediaSource && mediaSource.readyState === "open") {
    sourceBuffer = mediaSource.addSourceBuffer(mimeCodec)
    sourceBuffer.addEventListener("updateend", handleUpdateEnd)
    sourceBuffer.addEventListener("error", handleSourceBufferError)
    appendNextSegment() // Append any queued segments
  } else {
    // If MediaSource isn't open yet, retry after a short delay
    setTimeout(createSourceBuffer, 100) // You can adjust the delay
  }
}

function handleUpdateEnd() {
  // console.log("Fired updateend!")
  // const buffered = sourceBuffer.buffered
  // console.log("Buffered ranges:", buffered)

  // for (let i = 0; i < buffered.length; i++) {
  //   console.log("Range:", i, "Start:", buffered.start(i), "End:", buffered.end(i))
  // }
  appendNextSegment()
  videoPlayer.play()
}

function appendNextSegment() {
  if (!sourceBuffer.updating && queue.length > 0) {
    sourceBuffer.appendBuffer(queue.shift()) // Dequeue and append
  }
}

function handleSourceBufferError(event) {
  console.error("SourceBuffer Error:", event)
  //   console.log("Error code:", event.target.error?.code) // Specific error code (if available)
  //   console.log("MediaSource state:", mediaSource.readyState)
  //   console.log(
  //     "SourceBuffer state:",
  //     sourceBuffer.updating ? "updating" : "not updating"
  //   )
  //   console.log("Buffered ranges:", sourceBuffer.buffered)
}

function handleSourceClose() {
  // console.log("MediaSource closed with state:", mediaSource.readyState)
  console.error("MediaSource has been closed.")
}

function handleSourceError(error) {
  console.error("MediaSource error:", error)
}

socket.on("video_segment", ({ data, pts }) => {
  if (isProcessingFinished) return // Don't append after processing is done

  const videoBlob = Uint8Array.from(atob(data), (c) => c.charCodeAt(0))

  if (!sourceBuffer.updating) {
    sourceBuffer.timestampOffset = pts
    sourceBuffer.appendBuffer(videoBlob)
  } else {
    queue.push(videoBlob) // If MediaSource is not open or updating, add to queue
    appendNextSegment() // Attempt to append immediately
  }
})

socket.on("error", function (errorMessage) {
  showToast(errorMessage)
  userInput.disabled = false
  recordButton.disabled = false
})

socket.on("done", () => {
  showToast("Processing Finished!")
  userInput.disabled = false
  recordButton.disabled = false
})

function stopRecording() {
  if (mediaRecorder && recording) {
    mediaRecorder.stop() // stop ther recorder first
    recording = false // update the recording status
  }
  micIcon.classList.remove("ripple")
  // sendArrow.style.display = "block"
  recordButton.title = `Record Audio upto ${recordingTime}sec`
  userInput.disabled = false
  userInput.placeholder = "Audio recording sent!"
  userInput.value = ""
  secondsElapsed = 0
}

function startTimer() {
  timerInterval = setInterval(() => {
    // createRipple()
    secondsElapsed++
    userInput.value = `Listening...Time elapsed: ${secondsElapsed} Sec`
    if (secondsElapsed >= recordingTime) {
      clearInterval(timerInterval) // Clear the timer on stop
      stopRecording()
    }
  }, 1000)
}

// const audioPlayback = document.createElement("audio") // Create audio element
// function playAudio(blob) {
//   audioPlayback.src = URL.createObjectURL(blob) // Set audio source
//   audioPlayback.controls = true // Add playback controls
//   document.body.appendChild(audioPlayback) // Add to DOM
//   audioPlayback.play() // Start playback
// }

// Audio recording logic using MediaRecorder API
navigator.mediaDevices
  .getUserMedia({ audio: true })
  .then((stream) => {
    mediaRecorder = new MediaRecorder(stream) //, { mimeType: "audio/webm;codecs=opus" })

    recordButton.addEventListener("click", () => {
      if (!recording) {
        // Start recording logic here
        recording = true
        audioChunks = [] // Reset chunks for new recording
        mediaRecorder.start()

        sendArrow.style.display = "none"
        recordButton.title = "Stop Recording"
        userInput.disabled = true
        userInput.value = "Listening..."

        micIcon.classList.add("ripple")
        startTimer()
      } else {
        stopRecording() // Call your existing stopRecording function
      }
    })

    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data)
      }
      // If recording has stopped and data is available, create the blob
      if (!recording && audioChunks.length > 0) {
        const audioBlob = new Blob(audioChunks, {
          type: "audio/webm;codecs=opus",
        })
        // playAudio(audioBlob) // Play the recorded audio
        sendDataToServer(audioBlob, "audio") // Send the audio blob to server
        audioChunks = [] // clear the array for the next recording
      }
    })
  })
  .catch((error) => {
    showToast("Error accessing microphone.")
  })
