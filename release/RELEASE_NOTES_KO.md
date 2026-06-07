# Resonix AI v0.9.0 릴리즈 노트

## 주요 기능

- AI 청감 목표 기반 오디오 복원/마스터링
- 다중 목표 선택 및 AI 적용량 조절
- 스테레오 미드/사이드 기반 처리와 트루 피크 헤드룸
- 원본/향상 A/B 파형 비교
- 다중 파일 배치 처리 및 ZIP 다운로드
- 배치 결과 개별 파일 선택 재생 및 개별 WAV 다운로드
- 출력 샘플레이트 자동/원본/44.1 kHz/48 kHz/96 kHz 선택
- 16-bit PCM 또는 24-bit PCM 출력 선택
- 고급 수동 DSP override

## 배포 형태

- Windows one-folder ZIP 배포
- 압축 해제 후 `ResonixAI.exe` 실행
- `_internal` 폴더는 실행에 필요하므로 삭제하지 마세요.

## 알려진 사항

- Windows SmartScreen 경고가 나타날 수 있습니다. 아직 코드 서명 인증서가 적용되지 않았습니다.
- 첫 실행과 첫 처리 요청은 라이브러리 초기화 때문에 약간 느릴 수 있습니다.
- 기본 노이즈 처리는 빠른 경로를 사용합니다. 실험적으로 GTCRN 모델 경로를 켜려면 `RESONIX_ENABLE_GTCRN=1` 환경변수를 설정해야 합니다.

## 로그 위치

```text
%USERPROFILE%\.packaged_audio_ai\logs\app.log
```
