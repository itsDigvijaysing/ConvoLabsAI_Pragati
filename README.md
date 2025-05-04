# ConvoLabsAI – VoiceEd India

## Vision

“To make quality education as accessible as a phone call—for every child, in every corner of India.”

## Project Overview

VoiceEd India is a multilingual, toll-free, AI-powered education platform that delivers interactive learning through standard phone calls—no internet or smartphone required. Built for school students in rural and underserved communities, it addresses the core barriers of connectivity, language diversity, and affordability in education.

## Problem Statement

Despite having over 500 million mobile users, millions of Indian students lack access to quality education due to:
- Poverty
- Limited or no internet access (only ~50% penetration)
- Language barriers (most digital content is English/Hindi)
- Lack of educational infrastructure in rural areas

## Objective

To provide real-time, interactive, and personalized learning via a simple phone call, removing the need for smartphones, internet, or paid services.

## Target Audience

- School students aged 10–18
- Rural and underserved communities
- Students with no access to internet, formal tutoring, or local-language content

### AI Models
- Large Language Models: Mistral, LLaMA
- STT (Speech-to-Text): Mozilla DeepSpeech, Whisper
- TTS (Text-to-Speech): Coqui TTS, Festival

### Telephony Integration
- Open-source: Asterisk, FreeSWITCH
- Commercial APIs (optional): Twilio, Exotel

## High-Level Solution Architecture

1. User makes a toll-free call from any mobile phone.
2. Voice is captured and processed by a telephony stack (e.g., Asterisk / Twilio).
3. Audio is converted to text using STT (e.g., DeepSpeech).
4. Text input is passed to the LLM (Mistral/LLaMA) using LangChain or LlamaIndex.
5. If needed, the system uses RAG with a syllabus-aligned corpus for better accuracy.
6. LLM response is converted to audio using TTS (e.g., Coqui).
7. Audio is played back to the caller over the phone.

## Core Features

- No Internet Required: Access via GSM call
- Multilingual Support: Supports Indian regional languages
- Ultra-Accessible: Works on basic phones
- AI-Powered Tutor: Personalized, contextual responses
- Cloud-Native & Scalable: Supports thousands of concurrent users
- Open-Source & Affordable: No license fees, low operational cost
- Privacy-Conscious: Data anonymized and stored only with consent

## Datasets Used

| Type              | Examples                                                 |
|-------------------|----------------------------------------------------------|
| Curriculum        | CBSE public domain content                               |

## Cloud Dependency & Optimization

The solution is cloud-based to run LLMs and STT/TTS models efficiently.

### Cost-Effective Cloud Strategy
- Use of open-source tools to eliminate license costs
- Deploy on AWS Lambda, spot instances, or on-premise servers
- Use time limits and caching to reduce compute
- Partner with NGOs or government for infrastructure and subsidies

## Scalability

- Works on any basic phone via toll-free number
- Cloud-native, modular architecture
- Real-time LLM-powered conversations
- Extensible to more subjects and languages
- Minimal cost per interaction enables mass-scale rollout
