class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super()
    this.targetSampleRate = options.processorOptions?.targetSampleRate || 16000
    this.chunkDuration = options.processorOptions?.chunkDuration || 0.1
    this.samples = []
    this.inputChunkSize = Math.round(sampleRate * this.chunkDuration)
  }

  process(inputs) {
    const channel = inputs[0]?.[0]
    if (!channel) return true

    for (let index = 0; index < channel.length; index += 1) {
      this.samples.push(channel[index])
    }

    while (this.samples.length >= this.inputChunkSize) {
      const input = this.samples.splice(0, this.inputChunkSize)
      const outputSize = Math.round(this.targetSampleRate * this.chunkDuration)
      const pcm = new Int16Array(outputSize)
      const ratio = input.length / outputSize

      for (let index = 0; index < outputSize; index += 1) {
        const position = index * ratio
        const left = Math.floor(position)
        const right = Math.min(left + 1, input.length - 1)
        const fraction = position - left
        const sample = input[left] + (input[right] - input[left]) * fraction
        const clamped = Math.max(-1, Math.min(1, sample))
        pcm[index] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff
      }

      this.port.postMessage(pcm.buffer, [pcm.buffer])
    }
    return true
  }
}

registerProcessor('pcm-capture-processor', PcmCaptureProcessor)

