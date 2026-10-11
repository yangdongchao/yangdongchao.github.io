const menuButton = document.querySelector('.menu-toggle');
const navigation = document.querySelector('#main-nav');
function closeMenu() {
  navigation?.classList.remove('is-open');
  menuButton?.setAttribute('aria-expanded', 'false');
  menuButton?.setAttribute('aria-label', 'Open navigation');
}
menuButton?.addEventListener('click', () => {
  const open = navigation.classList.toggle('is-open');
  menuButton.setAttribute('aria-expanded', String(open));
  menuButton.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
});
navigation?.querySelectorAll('a').forEach(link => link.addEventListener('click', closeMenu));
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeMenu(); });

document.querySelectorAll('[data-waveform]').forEach(wave => {
  // Decorative illustration, not a measured waveform.
  for (let i = 0; i < 65; i++) {
    const bar = document.createElement('i');
    bar.style.height = `${5 + Math.abs(Math.sin(i * 1.7) * Math.cos(i * .21)) * 37}px`;
    wave.append(bar);
  }
});

const formatTime = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
document.querySelectorAll('[data-player]').forEach(player => {
  const audio = player.querySelector('audio');
  const button = player.querySelector('.play-button');
  const seek = player.querySelector('.audio-seek');
  const time = player.querySelector('.audio-time');
  const label = button.getAttribute('aria-label').replace(/^Play /, '');
  const error = player.querySelector('.audio-error');
  button.addEventListener('click', async () => {
    if (!audio.paused) { audio.pause(); return; }
    document.querySelectorAll('audio').forEach(other => { if (other !== audio) other.pause(); });
    try { await audio.play(); error.hidden = true; }
    catch { error.hidden = false; }
  });
  audio.addEventListener('play', () => {
    button.firstElementChild.textContent = 'Ⅱ';
    button.setAttribute('aria-label', `Pause ${label}`);
  });
  audio.addEventListener('pause', () => {
    button.firstElementChild.textContent = '▶';
    button.setAttribute('aria-label', `Play ${label}`);
  });
  function update() {
    const duration = Number.isFinite(audio.duration) ? audio.duration : 0;
    seek.value = duration ? audio.currentTime / duration * 100 : 0;
    seek.setAttribute('aria-valuetext', `${formatTime(audio.currentTime)} of ${formatTime(duration)}`);
    time.textContent = formatTime(audio.currentTime || duration);
  }
  audio.addEventListener('loadedmetadata', update);
  if (audio.readyState >= 1) update();
  audio.addEventListener('timeupdate', update);
  audio.addEventListener('ended', () => { audio.currentTime = 0; update(); });
  audio.addEventListener('error', () => { error.hidden = false; });
  seek.addEventListener('input', () => {
    if (Number.isFinite(audio.duration)) audio.currentTime = seek.value / 100 * audio.duration;
  });
});

const papers = [
  {year:2025,topic:'models',title:'Omni-AutoThink: Adaptive Multimodal Reasoning via Reinforcement Learning',summary:'Adaptive SFT and GRPO teach an omni model when to reason across text, audio and vision.',venue:'2025 · Preprint',paper:'https://arxiv.org/abs/2512.03783',code:'https://github.com/yangdongchao/Omni-AutoThink'},
  {year:2025,topic:'models',title:'LongCat-Flash-Omni Technical Report',summary:'Team contribution to an open omni-modal system for real-time audio-visual interaction.',venue:'2025 · Technical report',paper:'https://arxiv.org/abs/2511.00279',code:'https://github.com/meituan-longcat/LongCat-Flash-Omni'},
  {year:2026,topic:'models',title:'UniAudio 2.0: A Unified Audio Language Model with Text-Aligned Factorized Audio Tokenization',summary:'Factorized audio tokens and a unified architecture for understanding and generation.',venue:'2026 · Preprint',paper:'https://arxiv.org/abs/2602.04683',code:'https://github.com/yangdongchao/UniAudio2',demo:'https://dongchaoyang.top/UniAudio2Demo/'},
  {year:2026,topic:'models',title:'HeartMuLa: A Family of Open Sourced Music Foundation Models',summary:'Open music models spanning audio–text alignment, lyric recognition, coding, and song generation.',venue:'2026 · Preprint',paper:'https://arxiv.org/abs/2601.10547',code:'https://github.com/HeartMuLa/heartlib',demo:'https://heartmula.github.io/'},
  {year:2026,topic:'speech',title:'UniSRM: A Unified Speech Reward Model for Reasoning-Based Fine-grained Assessment',summary:'Interpretable, multidimensional evaluation of speech generation.',venue:'ACL 2026 · Main',paper:'https://arxiv.org/abs/2605.23261'},
  {year:2026,topic:'speech',title:'DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling with Large Language Models',summary:'Understanding-driven tokens and acoustic tokens for joint speech modeling.',venue:'AAAI 2026',paper:'https://arxiv.org/abs/2508.08961'},
  {year:2025,topic:'tokens',title:'ALMTokenizer: A Low-bitrate and Semantic-rich Audio Codec Tokenizer for Audio Language Modeling',summary:'Query-based compression and semantic-rich representations for downstream audio modeling.',venue:'ICML 2025',paper:'https://arxiv.org/abs/2504.10344'},
  {year:2025,topic:'speech',title:'SimpleSpeech 2: Towards Simple and Efficient Text-to-Speech with Flow-based Scalar Latent Transformer Diffusion Models',summary:'Flow matching in a scalar latent space for simplified non-autoregressive speech synthesis.',venue:'IEEE TASLP 2025',paper:'https://arxiv.org/abs/2408.13893',code:'https://github.com/yangdongchao/SimpleSpeech',demo:'https://dongchaoyang.top/SimpleSpeech2_demo/'},
  {year:2025,topic:'speech',title:'MoonCast: High-Quality Zero-Shot Podcast Generation',summary:'Long-form, multi-speaker podcast generation with natural conversational delivery.',venue:'2025',paper:'https://arxiv.org/abs/2503.14345',demo:'https://mooncastdemo.github.io/'},
  {year:2024,topic:'models',title:'UniAudio: Towards Universal Audio Generation with Large Language Models',summary:'Unified multi-task audio generation with a Global–Local Transformer.',venue:'ICML 2024',paper:'https://arxiv.org/abs/2310.00704',code:'https://github.com/yangdongchao/UniAudio'},
  {year:2024,topic:'tokens',title:'UniAudio 1.5: Large Language Model-Driven Audio Codec is a Few-Shot Audio Task Learner',summary:'LLM-Codec maps audio into a frozen language model’s vocabulary for in-context learning.',venue:'NeurIPS 2024',paper:'https://arxiv.org/abs/2406.10056',code:'https://github.com/yangdongchao/LLM-Codec'},
  {year:2024,topic:'speech',title:'SimpleSpeech: Towards Simple and Efficient Text-to-Speech with Scalar Latent Transformer Diffusion Models',summary:'Scalar-quantized representations and Transformer diffusion for speech synthesis.',venue:'Interspeech 2024 · ISCA Best Student Paper Award',paper:'https://arxiv.org/abs/2406.02328',code:'https://github.com/yangdongchao/SimpleSpeech'},
  {year:2024,topic:'speech',title:'InstructTTS: Modelling Expressive TTS in Discrete Latent Space with Natural Language Style Prompt',summary:'Early natural-language speaking-style control, with free-form descriptions and discrete diffusion.',venue:'IEEE TASLP 2024 · First released January 2023',paper:'https://arxiv.org/abs/2301.13662',demo:'https://dongchaoyang.top/InstructTTS/'},
  {year:2024,topic:'speech',title:'NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models',summary:'Speech attributes are modeled in factorized subspaces for zero-shot synthesis.',venue:'ICML 2024',paper:'https://arxiv.org/abs/2403.03100'},
  {year:2024,topic:'speech',title:'PromptTTS 2: Describing and Generating Voices with Text Prompt',summary:'Voice variability and natural-language prompts for controllable speech synthesis.',venue:'ICLR 2024',paper:'https://arxiv.org/abs/2309.02285'},
  {year:2024,topic:'models',title:'AudioGPT: Understanding and Generating Speech, Music, Sound, and Talking Head',summary:'A conversational system coordinating audio understanding and generation models.',venue:'AAAI 2024',paper:'https://arxiv.org/abs/2304.12995'},
  {year:2023,topic:'sound',title:'Diffsound: Discrete Diffusion Model for Text-to-sound Generation',summary:'Text-conditioned sound generation through discrete diffusion over acoustic representations.',venue:'IEEE TASLP 2023 · First released 2022',paper:'https://arxiv.org/abs/2207.09983',code:'https://github.com/yangdongchao/Text-to-sound-Synthesis',demo:'https://dongchaoyang.top/text-to-sound-synthesis-demo/'},
  {year:2023,topic:'tokens',title:'HiFi-Codec: Group-residual Vector Quantization for High Fidelity Audio Codec',summary:'Compact audio codes and the AcademiCodec open-source training toolkit.',venue:'2023 · Preprint',paper:'https://arxiv.org/abs/2305.02765',code:'https://github.com/yangdongchao/AcademiCodec'},
  {year:2023,topic:'sound',title:'Make-An-Audio: Text-To-Audio Generation with Prompt-Enhanced Diffusion Models',summary:'Prompt enhancement and latent diffusion for text-conditioned audio generation.',venue:'ICML 2023',paper:'https://arxiv.org/abs/2301.12661',code:'https://github.com/Text-to-Audio/Make-An-Audio'},
  {year:2022,topic:'sound',title:'Detect What You Want: Target Sound Detection',summary:'Detecting a specified sound in a mixture using a reference audio example.',venue:'DCASE 2022',paper:'https://arxiv.org/abs/2112.10153'},
  {year:2022,topic:'sound',title:'RaDur: A Reference-aware and Duration-robust Network for Target Sound Detection',summary:'Improved robustness to reference quality and short-duration sound events.',venue:'Interspeech 2022',paper:'https://arxiv.org/abs/2204.02143'},
  {year:2022,topic:'sound',title:'A Mutual Learning Framework for Few-shot Sound Event Detection',summary:'Mutual updates of class prototypes and feature representations for few-shot detection.',venue:'ICASSP 2022',paper:'https://arxiv.org/abs/2110.04474'},
  {year:2022,topic:'sound',title:'A Mixed Supervised Learning Framework for Target Sound Detection',summary:'Learning target sound categories from weak annotations with stronger supervision on base categories.',venue:'DCASE 2022',paper:'https://arxiv.org/abs/2204.02088'}
];
const publicationList = document.querySelector('#publication-list');
if (publicationList) {
  const search = document.querySelector('#paper-search');
  const buttons = [...document.querySelectorAll('[data-topic]')];
  const count = document.querySelector('#pub-count');
  const allowed = buttons.map(button => button.dataset.topic);
  let topic = new URLSearchParams(location.search).get('topic') || 'all';
  if (!allowed.includes(topic)) topic = 'all';
  function render() {
    const query = search.value.trim().toLowerCase();
    const shown = [...papers].sort((a, b) => b.year - a.year).filter(paper => (topic === 'all' || paper.topic === topic) && `${paper.title} ${paper.summary} ${paper.venue} ${paper.year}`.toLowerCase().includes(query));
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.topic === topic)));
    count.textContent = `${shown.length} of ${papers.length} papers in this research index`;
    publicationList.replaceChildren();
    for (const paper of shown) {
      const article = document.createElement('article'); article.className = 'publication';
      const year = document.createElement('time'); year.textContent = paper.year;
      const body = document.createElement('div');
      const heading = document.createElement('h2');
      const title = document.createElement('a'); title.href = paper.paper; title.textContent = paper.title; heading.append(title);
      const summary = document.createElement('p'); summary.textContent = paper.summary;
      const venue = document.createElement('span'); venue.className = 'pub-venue'; venue.textContent = paper.venue;
      const links = document.createElement('div'); links.className = 'work-links';
      for (const [key, name] of [['paper','Paper'],['code','Code'],['demo','Demos']]) {
        if (paper[key]) { const a = document.createElement('a'); a.href = paper[key]; a.textContent = `${name} ↗`; links.append(a); }
      }
      body.append(heading, summary, venue, links); article.append(year, body); publicationList.append(article);
    }
    document.querySelector('#empty-state').hidden = shown.length !== 0;
  }
  buttons.forEach(button => button.addEventListener('click', () => {
    topic = button.dataset.topic;
    const url = new URL(location.href); topic === 'all' ? url.searchParams.delete('topic') : url.searchParams.set('topic', topic);
    history.replaceState(null, '', url); render();
  }));
  search.addEventListener('input', render);
  document.querySelector('#clear-search').addEventListener('click', () => {
    topic = 'all'; search.value = ''; history.replaceState(null, '', location.pathname); render(); search.focus();
  });
  render();
}

// Keep the research film and individual samples from overlapping.
document.querySelectorAll('audio, video').forEach(media => {
  media.addEventListener('play', () => {
    document.querySelectorAll('audio, video').forEach(other => { if (other !== media) other.pause(); });
  });
});
