const dropZone = document.getElementById('dropZone');
const imageInput = document.getElementById('imageInput');
const dzIcon = document.getElementById('dzIcon');
const dzTitle = document.getElementById('dzTitle');
const dzSub = document.getElementById('dzSub');

// Drag-and-drop
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    imageInput.files = e.dataTransfer.files;
    dzIcon.textContent = '✅';
    dzTitle.textContent = file.name;
    dzSub.textContent = (file.size / 1024).toFixed(1) + ' KB';
  }
});

imageInput.addEventListener('change', () => {
  const file = imageInput.files[0];
  if (file) {
    dzIcon.textContent = '✅';
    dzTitle.textContent = file.name;
    dzSub.textContent = (file.size / 1024).toFixed(1) + ' KB';
  }
});

async function uploadImage() {
  const file = imageInput.files[0];
  if (!file) { alert('Please select an image first.'); return; }

  const btn = document.getElementById('predictBtn');
  const loader = document.getElementById('loader');
  const resultBox = document.getElementById('result');

  btn.disabled = true;
  loader.style.display = 'block';
  resultBox.classList.add('hidden');
  resultBox.classList.remove('visible');

  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch("https://real-vs-ai-classifier-2.onrender.com/predict", {
    method: "POST",
    body: formData
});



    if (!response.ok) throw new Error(response.statusText);

    const data = await response.json();
    const isAI = data.prediction.toLowerCase().includes('ai');

    document.getElementById('resultText').textContent = data.prediction;
    document.getElementById('resultSub').textContent = data.confidence
      ? 'Confidence: ' + (data.confidence * 100).toFixed(1) + '%'
      : 'Analysis complete';

    const dot = document.getElementById('resultDot');
    const badge = document.getElementById('resultBadge');
    dot.className = 'result-dot' + (isAI ? ' ai' : '');
    badge.className = 'badge' + (isAI ? ' ai' : ' real');
    badge.textContent = isAI ? 'AI' : 'Real';

    resultBox.classList.remove('hidden');
    requestAnimationFrame(() => resultBox.classList.add('visible'));

  } catch (err) {
    document.getElementById('resultText').textContent = 'Request failed';
    document.getElementById('resultSub').textContent = err.message;
    resultBox.classList.remove('hidden');
    requestAnimationFrame(() => resultBox.classList.add('visible'));
  } finally {
    loader.style.display = 'none';
    btn.disabled = false;
  }
}