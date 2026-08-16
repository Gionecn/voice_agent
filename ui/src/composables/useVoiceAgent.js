import { computed, onBeforeUnmount, ref } from 'vue'
import { MicrophoneCapture } from '../services/microphone'
import { PcmPlayer } from '../services/pcmPlayer'
import { VoiceSocket } from '../services/voiceSocket'

const STATE_LABELS = {
  disconnected: '尚未连接',
  connecting: '正在连接',
  listening: '正在聆听',
  thinking: '正在思考',
  speaking: '正在回答',
  error: '发生错误',
}

export function useVoiceAgent() {
  const state = ref('disconnected')
  const partialTranscript = ref('')
  const messages = ref([])
  const error = ref('')
  const player = new PcmPlayer(24000)
  let socket = null
  let microphone = null

  const connected = computed(() => !['disconnected', 'connecting'].includes(state.value))
  const statusLabel = computed(() => STATE_LABELS[state.value] || state.value)

  function handleEvent(event) {
    switch (event.type) {
      case 'session.ready':
        state.value = 'listening'
        break
      case 'state.changed':
        state.value = event.state
        break
      case 'stt.partial':
        partialTranscript.value = event.text
        break
      case 'stt.final':
        partialTranscript.value = ''
        messages.value.push({ role: 'user', text: event.text })
        break
      case 'agent.started':
        messages.value.push({ role: 'assistant', text: '' })
        break
      case 'agent.delta': {
        const last = messages.value.at(-1)
        if (last?.role === 'assistant') last.text += event.text
        break
      }
      case 'playback.clear':
        player.clear()
        break
      case 'error':
        error.value = `${event.source}: ${event.message}`
        state.value = 'error'
        break
    }
  }

  async function start() {
    if (connected.value || state.value === 'connecting') return
    state.value = 'connecting'
    error.value = ''
    await player.unlock()
    socket = new VoiceSocket({
      onEvent: handleEvent,
      onAudio: (audio) => player.enqueue(audio),
      onClose: () => {
        if (state.value !== 'disconnected') state.value = 'disconnected'
      },
    })
    microphone = new MicrophoneCapture((audio) => socket.sendAudio(audio))
    try {
      // Start both immediately from the user gesture. Audio produced before
      // the socket opens is safely ignored by VoiceSocket.sendAudio().
      await Promise.all([socket.connect(), microphone.start()])
    } catch (cause) {
      error.value = cause.message || String(cause)
      state.value = 'error'
      await stop()
    }
  }

  async function interrupt() {
    player.clear()
    socket?.send('interrupt')
  }

  async function stop() {
    await microphone?.stop()
    microphone = null
    socket?.close()
    socket = null
    player.clear()
    state.value = 'disconnected'
    partialTranscript.value = ''
  }

  onBeforeUnmount(async () => {
    await stop()
    await player.close()
  })

  return {
    state,
    statusLabel,
    connected,
    partialTranscript,
    messages,
    error,
    start,
    stop,
    interrupt,
  }
}
