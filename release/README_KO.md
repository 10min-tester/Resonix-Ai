# Resonix AI v0.9.1 for Windows

Resonix AI는 로컬 PC에서 실행되는 AI 기반 음원 분석 및 개선 도구입니다.
원본 음원을 분석한 뒤 선택한 청취 목표에 맞춰 톤 밸런스, 스테이징 보존, 헤드룸, 출력 포맷을 조정합니다.

## 실행 방법

1. `ResonixAI-v0.9.1-Windows.zip` 파일을 다운로드합니다.
2. 원하는 위치에 압축을 해제합니다.
3. `ResonixAI-Windows` 폴더 안의 `ResonixAI.exe`를 실행합니다.
4. 브라우저가 자동으로 열리면 음원을 업로드해 처리합니다.

브라우저가 자동으로 열리지 않으면 아래 주소를 직접 입력합니다.

```text
http://127.0.0.1:8000/
```

8000번 포트가 사용 중이면 앱이 8001 이후 포트를 자동으로 찾습니다.

## 주요 기능

- AI 기반 음원 분석 및 자동 처리
- 7가지 청취 목표 선택: Restore, Hi-Fi Clean, Hi-Fi Bright, Warm Analog, Loud Modern, Bass Boost, Voice Focus
- 여러 청취 목표 동시 선택 및 혼합 적용
- 원본 음량 매칭 또는 AI 목표 음량 선택
- 스테레오/미드사이드 기반 처리로 스테이징 손상 완화
- 피크 인지 라우드니스 보정, linked peak limiter, 트루피크 헤드룸 확보
- 출력 샘플레이트 자동/원본/44.1 kHz/48 kHz/96 kHz 선택
- 16-bit PCM 또는 24-bit PCM WAV 출력
- 단일/다중 음원 처리, A/B 파형 비교, ZIP 다운로드

## v0.9.1 핫픽스

- 순간 피크가 높은 음원에서 목표 LUFS를 억지로 올리지 않도록 개선했습니다.
- 기존 tanh 기반 리미터를 linked peak limiter로 교체했습니다.
- 기본 리미터 ceiling을 더 보수적으로 조정했습니다.
- 최종 WAV 저장 전 sample peak guard를 추가했습니다.
- 클리핑처럼 들릴 수 있는 출력 왜곡 가능성을 줄였습니다.

## 주의 사항

- `ResonixAI.exe`만 따로 실행하지 마세요. `_internal` 폴더가 exe와 같은 위치에 있어야 합니다.
- Windows SmartScreen 경고가 뜰 수 있습니다. 코드 서명 인증서가 아직 적용되지 않았기 때문입니다.
- 첫 실행은 라이브러리 초기화 때문에 시간이 조금 걸릴 수 있습니다.
- 처리 결과와 로그는 사용자 폴더 아래 `.packaged_audio_ai`에 저장됩니다.

## 로그 위치

```text
%USERPROFILE%\.packaged_audio_ai\logs\app.log
```
