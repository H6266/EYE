# Ɔkwankyerɛfo Pa — Auteur Studio Project

**60-second cinematic advertisement for Ghanaian voice-guided USSD accessibility solution**

## Project Overview

This project contains a complete Auteur Studio timeline, assets, prompts, and production metadata for the Ɔkwankyerɛfo Pa advertisement.

### Timeline Structure (65 seconds total)

| Time | Scene | Track | Description |
|------|-------|-------|-------------|
| 0–8s | Establishing Veranda | V1 | Wide shot of peaceful Ghanaian veranda at golden hour |
| 8–16s | Kwame: USSD Struggle | V1 | Close-up of Kwame with feature phone, thoughtful expression |
| 16–24s | Extreme Close-up: Keypad | V1 | Macro shot of fingertips on physical keypad |
| 24–32s | Language Selection | V1 | Phone shows USSD menu, Kwame presses "2" for Twi |
| 32–45s | Voice-Guided Transaction | V1 | Over-shoulder: USSD menu navigation with voice guidance |
| 45–57s | PIN Entry: Privacy | V1 | Kwame shields keypad, enters PIN privately |
| 57–60s | Confirmation Smile | V1 | Intimate close-up: confident smile, "Transaction Successful" |
| 60–65s | Brand Reveal | V1 | Navy blue brand card with sound waves, logo, tagline |

### Audio Tracks

| Track | Asset | Duration | Description |
|-------|-------|----------|-------------|
| A1 (Voice) | Main Voiceover | 60s | English, warm Ghanaian delivery |
| A1 (Voice) | Phone IVR System | 44s | Clear instructional voice (English + Twi) |
| A2 (Music) | Piano Intro | 16s | Soft cinematic piano (0–16s) |
| A2 (Music) | Uplifting Score | 39s | Building instrumental with kora textures (16–55s) |
| A2 (Music) | Brand Signature | 5s | 3-note marimba/kalimba audio logo (55–60s) |

### Subtitle Track (S1)

7 cues synced to voiceover, cinematic style.

### Character: Kwame

- **Age:** 30 (28–35 range)
- **Appearance:** Dark sunglasses, blue kente-patterned shirt, confident intelligent expression
- **Direction:** Intelligent, passionate, thoughtful, confident, determined. NOT helpless.

## Image Generation Prompts

All 8 scenes have detailed prompts in `OKWANKYEREFO_PA_IMAGE_PROMPTS` optimized for:
- Cinematic 35mm anamorphic look
- Natural Ghanaian skin tones
- Golden hour / warm lighting
- Authentic Ghanaian environment
- Professional commercial quality

## Loading in Auteur Studio

```typescript
import { loadOkwankyerɛfoPaProject } from './src/data/okwankyerefoPaProject';

const project = loadOkwankyerɛfoPaProject();
// project.assets, project.tracks, project.clips, project.character, project.imagePrompts, project.meta
```

## Production Requirements

- **Camera:** ARRI Alexa Mini LF / Sony VENICE 2 + anamorphic lenses
- **Location:** Authentic Ghanaian residential veranda (Accra/Kumasi suburb)
- **Time:** Golden hour (5:30–6:30 PM)
- **Post:** DaVinci Resolve, ACES 1.3, Kodak 2383 film print emulation
- **Deliverables:** 1080p ProRes 422 HQ, H.264 web, 9:16 cutdowns, SRT/VTT subtitles

## Brand Guidelines

- **Primary Navy:** #0A1628
- **Primary Gold:** #D4A843
- **Typography:** Custom serif (logo), Inter/DM Sans 300 (tagline)
- **Logo Clear Space:** Height of "Ɔ" character

## Security & Authenticity Notes

- ✅ Never shows PIN/OTP/password
- ✅ User retains control of phone throughout
- ✅ Not portrayed as ordinary phone call
- ✅ Authentic USSD session visual language
- ✅ Ghanaian casting for Kwame (visually impaired actor preferred)
- ✅ Twi language integration authentic

---

*Generated for Auteur Studio — Next-Generation Neural Image Intelligence Platform*