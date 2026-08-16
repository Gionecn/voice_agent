export class VoiceSocket {
  constructor({ onEvent, onAudio, onClose }) {
    this.onEvent = onEvent
    this.onAudio = onAudio
    this.onClose = onClose
    this.socket = null
  }

  connect() {
    return new Promise((resolve, reject) => {
      const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
      this.socket = new WebSocket(`${protocol}//${location.host}/ws/voice`)
      this.socket.binaryType = 'arraybuffer'
      this.socket.onopen = resolve
      this.socket.onerror = () => reject(new Error('无法连接语音服务'))
      this.socket.onclose = (event) => this.onClose?.(event)
      this.socket.onmessage = (message) => {
        if (message.data instanceof ArrayBuffer) {
          this.onAudio?.(message.data)
          return
        }
        try {
          this.onEvent?.(JSON.parse(message.data))
        } catch (error) {
          console.error('无法解析服务端事件', error)
        }
      }
    })
  }

  sendAudio(audio) {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(audio)
    }
  }

  send(type, payload = {}) {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ type, ...payload }))
    }
  }

  close() {
    if (this.socket?.readyState === WebSocket.OPEN) {
      this.send('session.stop')
    }
    this.socket?.close()
    this.socket = null
  }
}

