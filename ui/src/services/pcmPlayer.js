export class PcmPlayer {
  constructor(sampleRate = 24000) {
    this.sampleRate = sampleRate
    this.context = null
    this.nextStartTime = 0
    this.sources = new Set()
  }

  async unlock() {
    this.context ||= new AudioContext()
    if (this.context.state === 'suspended') await this.context.resume()
  }

  async enqueue(arrayBuffer) {
    await this.unlock()
    const pcm = new Int16Array(arrayBuffer)
    const audioBuffer = this.context.createBuffer(1, pcm.length, this.sampleRate)
    const samples = audioBuffer.getChannelData(0)
    for (let index = 0; index < pcm.length; index += 1) {
      samples[index] = pcm[index] / (pcm[index] < 0 ? 0x8000 : 0x7fff)
    }

    const source = this.context.createBufferSource()
    source.buffer = audioBuffer
    source.connect(this.context.destination)
    const now = this.context.currentTime
    this.nextStartTime = Math.max(this.nextStartTime, now + 0.03)
    source.start(this.nextStartTime)
    this.nextStartTime += audioBuffer.duration
    this.sources.add(source)
    source.onended = () => this.sources.delete(source)
  }

  clear() {
    for (const source of this.sources) {
      try {
        source.stop()
      } catch {
        // A source may already have ended between iteration and stop().
      }
    }
    this.sources.clear()
    this.nextStartTime = this.context?.currentTime || 0
  }

  async close() {
    this.clear()
    if (this.context && this.context.state !== 'closed') await this.context.close()
    this.context = null
  }
}

