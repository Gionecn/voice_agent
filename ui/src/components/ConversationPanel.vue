<script setup>
defineProps({
  messages: { type: Array, required: true },
  partialTranscript: { type: String, default: '' },
})
</script>

<template>
  <section class="conversation" aria-live="polite">
    <div v-if="messages.length === 0 && !partialTranscript" class="conversation__empty">
      点击开始后直接说话。建议佩戴耳机，以获得更好的打断体验。
    </div>
    <article v-for="(message, index) in messages" :key="index" class="message" :class="`message--${message.role}`">
      <span class="message__role">{{ message.role === 'user' ? '' : '助手' }}</span>
      <p>{{ message.text || '…' }}</p>
    </article>
    <article v-if="partialTranscript" class="message message--partial">
      <span class="message__role">识别中</span>
      <p>{{ partialTranscript }}</p>
    </article>
  </section>
</template>

