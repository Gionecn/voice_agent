<script setup>
import ConversationPanel from './components/ConversationPanel.vue'
import StatusIndicator from './components/StatusIndicator.vue'
import VoiceControls from './components/VoiceControls.vue'
import { useVoiceAgent } from './composables/useVoiceAgent'

const voice = useVoiceAgent()
</script>

<template>
  <main class="shell">
    <header class="topbar">
      <div class="brand">
        <span class="brand__product">AI Voice</span>
      </div>
      <StatusIndicator :state="voice.state.value" :label="voice.statusLabel.value" />
    </header>

    <section class="hero">
      <div class="hero__content">
        <p class="eyebrow">STREAMING VOICE AGENT</p>
        <p class="hero__subtitle">
          释放双手，面向自然对话打造的实时语音助手，让每一次交流都更流畅、更直接。
        </p>
        <div class="model-list" aria-label="语音助手技术模型">
          <span>Fun-ASR</span>
          <i aria-hidden="true"></i>
          <span>LLM</span>
          <i aria-hidden="true"></i>
          <span>Qwen 双向 TTS</span>
        </div>
      </div>
    </section>

    <ConversationPanel
      :messages="voice.messages.value"
      :partial-transcript="voice.partialTranscript.value"
    />

    <p v-if="voice.error.value" class="error">{{ voice.error.value }}</p>
    <VoiceControls
      :connected="voice.connected.value"
      :state="voice.state.value"
      @start="voice.start"
      @stop="voice.stop"
      @interrupt="voice.interrupt"
    />
  </main>
</template>
