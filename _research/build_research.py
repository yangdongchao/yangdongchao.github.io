"""Build static homepage presentations from reviewed editorial JSON. No dependencies."""
import json,re
from pathlib import Path
from html import escape
ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'_research/research-decks.json').read_text())
M=json.loads((ROOT/'_research/research-summaries.json').read_text())
GROUPS=[('UniAudio series', ['uniaudio', 'uniaudio15', 'uniaudio2']), ('Text-to-audio series', ['diffsound', 'make-an-audio']), ('Audio Tokenizer series', ['hifi-codec', 'almtokenizer', 'reasoningcodec']), ('Speech generative models', ['instructtts', 'simplespeech', 'simplespeech2']), ('Omni models', ['longcat-omni', 'omni-autothink'])]
ORDER=[k for _,keys in GROUPS for k in keys]+["heartmula"]
def deck(key):
 name,sub,slides=D[key];count=len(slides);content=''
 for n,(label,title,lead,body,foot) in enumerate(slides):
  content+=f'<section class="research-slide" id="{key}-slide-{n+1}" aria-label="{n+1} of {count}: {escape(label)}"><p class="deck-kicker">{escape(name.upper())} / {escape(label.upper())}</p><h4>{escape(title)}</h4><p class="deck-lead">{escape(lead)}</p><div class="deck-body">{body}</div><footer class="deck-source">{foot}</footer></section>\n'
 buttons=''.join(f'<button type="button" data-page="{n}" aria-controls="{key}-slide-{n+1}">{n+1:02d} <span>{escape(s[0])}</span></button>' for n,s in enumerate(slides))
 return f'<div class="research-deck" data-deck="{key}" role="region" aria-label="{escape(name)} research presentation" tabindex="0"><div class="deck-top"><span>RESEARCH EXPLAINED</span><span>{count} slides · explore at your pace</span></div><div class="deck-pages">{content}</div><div class="deck-controls" hidden><div class="deck-tabs" aria-label="Choose a slide">{buttons}</div><div class="deck-arrows"><button type="button" data-step="-1" aria-label="Previous slide">←</button><span class="deck-counter" aria-live="polite" aria-atomic="true">1 / {count}</span><button type="button" data-step="1" aria-label="Next slide">→</button></div></div></div>'
def article(key,gallery=False):
 name,subtitle,slides=D[key];m=M[key]
 opening=f'<article id="{key}" class="research-story"><div class="story-copy"><p class="story-meta">{escape(m["date"])}</p><h3>{escape(name)}</h3><p class="story-subtitle">{escape(subtitle)}</p><p class="story-intro">{escape(m["intro"])}</p><p class="story-meaning">{escape(m["meaning"])}</p><div class="work-links"><a href="{m["paper"]}">Paper ↗</a>'
 if not gallery:opening+=f'<a href="presentations.html#{key}">Open presentation ↗</a>'
 opening+='</div></div>'
 if gallery:return opening+deck(key)+'</article>\n'
 return opening+f'<details class="story-details"><summary>Explore the idea <span>Problem · method · evidence · significance</span></summary>{deck(key)}</details></article>\n'
page=ROOT/'index.html';html=page.read_text()
start=html.index('  <section id="selected-work"');end=html.index('  <section id="research"',start)
section='  <section id="selected-work" class="section" aria-labelledby="work-title"><div class="section-heading"><div><p class="eyebrow">RESEARCH</p><h2 id="work-title">Selected work</h2></div><a class="text-link" href="presentations.html">Explore all 14 presentations ↗</a></div><p class="section-deck">Five connected research lines: unified audio models, text-to-audio generation, audio tokenizers, speech generation, and omni models. Explore each work’s motivation, method, evidence, and significance.</p>\n'
section+='<div class="research-map" aria-label="Research themes">'+''.join(f'<a href="#{keys[0]}"><span>{i+1:02d}</span>{escape(group)}</a>' for i,(group,keys) in enumerate(GROUPS))+'</div>'
GROUP_INTROS={
 'UniAudio series':'1.0 → 1.5 → 2.0: from unified audio generation to learning tasks from examples, then joint understanding and generation.',
 'Text-to-audio series':'DiffSound and Make-An-Audio explore how text can specify environmental sound, through discrete diffusion and prompt-enhanced latent diffusion.',
 'Audio Tokenizer series':'HiFi-Codec, ALMTokenizer, and ReasoningCodec develop the audio interface: compact coding, semantic-rich compression, then separate roles for reasoning and reconstruction.',
 'Speech generative models':'InstructTTS studies natural-language style control; SimpleSpeech and SimpleSpeech 2 develop speech synthesis in scalar latent space.',
 'Omni models':'My participation in LongCat-Omni and my first-author work Omni-AutoThink extend this research toward multimodal interaction and adaptive reasoning.'
}
for group,keys in GROUPS:
 section+=f'<div class="story-group"><h2>{escape(group)}</h2><p class="story-group-intro">{escape(GROUP_INTROS[group])}</p>'
 if 'hifi-codec' in keys:section+='<span id="audio-tokens"></span>'
 section+=''.join(article(k) for k in keys)+'</div>'
section+='</section>\n<section class="section application-case" aria-labelledby="application-heading"><p class="eyebrow">RESEARCH IN PRACTICE</p><h2 id="application-heading">From methods to a music system</h2>'+article('heartmula')+'</section>\n'
section+='<section id="current-research" class="section" aria-labelledby="current-title"><div class="section-heading"><div><p class="eyebrow">CURRENT DIRECTION</p><h2 id="current-title">Toward omni-modal interaction</h2></div></div><p class="section-deck">My recent work extends audio modeling into multimodal systems, adaptive reasoning, and ongoing research on full-duplex interaction.</p><article id="duplex" class="ongoing-research"><p class="story-meta">ONGOING RESEARCH</p><h3>Full-duplex omni-modal interaction</h3><p>I am currently working on full-duplex interaction models across modalities. This direction brings together audio generation, multimodal understanding and interaction over time.</p><p class="ongoing-question">The research goal: models that can keep perceiving while responding, and adjust as the interaction evolves.</p></article></section>\n\n'
html=html[:start]+section+html[end:]
html=html.replace('Audio generation & foundation models</p>','Audio foundation models & omni-modal interaction</p>')
html=re.sub(r'<p class="hero-bio">.*?</p>','<p class="hero-bio">I build models that understand and generate audio. My research connects <a href="#instructtts">language-based control</a>, <a href="#audio-tokens">audio representations</a>, and <a href="#uniaudio">unified audio modeling</a>. <a href="#heartmula">HeartMuLa</a> brings methods from these threads into music generation. I also work on omni models, including <a href="#longcat-omni">LongCat-Omni</a> and <a href="#omni-autothink">Omni-AutoThink</a>, and am currently exploring <a href="#duplex">full-duplex omni-modal interaction</a>.</p>',html,count=1)
page.write_text(html)
nav=''.join(f'<a href="#{k}">{escape(D[k][0])}</a>' for k in ORDER)
gallery='<!DOCTYPE html>\n<html lang="en" class="presentation-root"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Research explained — Dongchao Yang</title><link rel="stylesheet" href="style.css"><link rel="stylesheet" href="research-decks.css"><link rel="stylesheet" href="research-stories.css"><script src="research-decks.js" defer></script></head><body><main class="container presentation-gallery"><a class="text-link" href="index.html">← Back to homepage</a><h1>Research, explained.</h1><p>Fourteen research presentations. The problem behind each one, the technical idea, the evidence, and its place in the research trajectory.</p><nav class="gallery-nav" aria-label="Presentations">'+nav+'</nav>'+''.join('<section class="gallery-group"><h2>'+escape(g)+'</h2>'+''.join(article(k,True) for k in keys)+'</section>' for g,keys in GROUPS)+'<section class="gallery-group"><h2>Research in practice</h2>'+article('heartmula',True)+'</section>'+'</main></body></html>\n'
(ROOT/'presentations.html').write_text(gallery)
print(f'Built {len(ORDER)} research introductions and {sum(len(D[k][2]) for k in ORDER)} slides in homepage and gallery.')
