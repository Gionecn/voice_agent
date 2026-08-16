<script setup>
defineProps({
  connected: Boolean,
  state: { type: String, required: true },
})
defineEmits(['start', 'stop', 'interrupt'])
</script>

<template>
  <div class="controls">
    <button v-if="!connected" class="button button--primary" :disabled="state === 'connecting'" @click="$emit('start')">
      {{ state === 'connecting' ? '正在连接…' : '开始对话' }}
    </button>
    <template v-else>
      <button class="button button--quiet" @click="$emit('stop')">结束会话</button>
    </template>
  </div>
</template>

