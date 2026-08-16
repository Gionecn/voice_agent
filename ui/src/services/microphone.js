export class MicrophoneCapture {
  constructor(onAudio) {
    this.onAudio = onAudio
    this.stream = null
    this.context = null
    this.source = null
    this.worklet = null
    this.silentGain = null
  }

  async start() {
    // Create the context synchronously from the button click call stack. If it
    // is created after awaiting the WebSocket, Chromium may keep it suspended.
    this.context = new AudioContext()
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    })
    await this.context.audioWorklet.addModule('/audio-worklet.js')
    await this.context.resume()

    this.source = this.context.createMediaStreamSource(this.stream)
    this.worklet = new AudioWorkletNode(this.context, 'pcm-capture-processor', {
      processorOptions: { targetSampleRate: 16000, chunkDuration: 0.1 },
    })
    this.silentGain = this.context.createGain()
    this.silentGain.gain.value = 0
    this.worklet.port.onmessage = (event) => this.onAudio(event.data)
    this.source.connect(this.worklet)
    this.worklet.connect(this.silentGain)
    this.silentGain.connect(this.context.destination)
  }

  async stop() {
    this.worklet?.disconnect()
    this.source?.disconnect()
    this.silentGain?.disconnect()
    this.stream?.getTracks().forEach((track) => track.stop())
    if (this.context && this.context.state !== 'closed') {
      await this.context.close()
    }
    this.stream = null
    this.context = null
  }
}
