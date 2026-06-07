# Resonix AI v1.0.0 for Windows

Resonix AI는 로컬 PC에서 실행되는 AI 기반 음원 분석 및 개선 도구입니다.
원본 음원을 분석한 뒤 선택한 청취 목표에 맞춰 톤 밸런스, 스테이징 보존, 헤드룸, 출력 포맷을 조정합니다.

## 실행 방법

1. `ResonixAI-v1.0.0-Windows.zip` 파일을 다운로드합니다.
2. 원하는 위치에 압축을 해제합니다.
3. `ResonixAI-Windows` 폴더 안의 `ResonixAI.exe`를 실행합니다.
4. 브라우저가 자동으로 열리면 음원을 업로드해 처리합니다.

브라우저가 자동으로 열리지 않으면 아래 주소를 직접 입력합니다.

```text
http://127.0.0.1:8000/
```

## v1.0.0 주요 변경

- 기본 출력 음량 모드를 `원본 음량 유지`로 변경했습니다.
- A/B 플레이어에 `공정 A/B 비교 (Level-matched A/B)`를 추가했습니다.
- 원본/처리본 중 더 큰 쪽을 낮춰서 볼륨 착시를 줄입니다.
- 분석 리포트에 품질 판정을 추가했습니다.
- 헤드룸, 트루피크, 클리핑 위험, 스테레오 보존 여부를 한눈에 확인할 수 있습니다.

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

## 주의 사항

- `ResonixAI.exe`만 따로 실행하지 마세요. `_internal` 폴더가 exe와 같은 위치에 있어야 합니다.
- Windows SmartScreen 경고가 뜰 수 있습니다. 코드 서명 인증서가 아직 적용되지 않았기 때문입니다.
- 첫 실행은 라이브러리 초기화 때문에 시간이 조금 걸릴 수 있습니다.
- 처리 결과와 로그는 사용자 폴더 아래 `.packaged_audio_ai`에 저장됩니다.

## 로그 위치

```text
%USERPROFILE%\.packaged_audio_ai\logs\app.log
```
