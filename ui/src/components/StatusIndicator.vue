<script setup>
defineProps({
  state: { type: String, required: true },
  label: { type: String, required: true },
})
</script>

<template>
  <div class="status" :data-state="state">
    <span class="status__dot" aria-hidden="true"></span>
    <span>{{ label }}</span>
  </div>
</template>

