# Resonix AI v1.0.0 릴리즈 노트

실사용에서 결과를 더 공정하게 판단할 수 있도록 볼륨 기준과 A/B 비교 방식을 개선한 첫 안정 배포 버전입니다.

## 주요 변경 사항

- 기본 볼륨 모드를 `원본 음량 유지`로 변경
- A/B 플레이어에 `공정 A/B 비교 (Level-matched A/B)` 추가
- 원본/처리본 중 더 크게 들리는 쪽을 낮춰 볼륨 착시 완화
- 리포트에 자동 품질 판정 추가
- 헤드룸 안정성, 트루피크 안전성, 클리핑 위험, 스테레오 보존 여부 표시
- A/B 비교에 적용된 원본/처리본 재생 gain 표시

## 포함된 기존 개선

- 피크 인지 라우드니스 보정
- linked peak limiter
- 최종 WAV 저장 전 sample peak guard
- 스테레오/미드사이드 기반 처리
- 다중 청취 목표 선택 및 혼합 적용
- 단일/다중 음원 처리와 ZIP 다운로드

## 사용 방법

1. `ResonixAI-v1.0.0-Windows.zip` 다운로드
2. 압축 해제
3. `ResonixAI-Windows` 폴더 안의 `ResonixAI.exe` 실행

## 주의 사항

- `ResonixAI.exe`만 따로 실행하지 마세요.
- `_internal` 폴더가 exe와 같은 위치에 있어야 합니다.
- Windows SmartScreen 경고가 뜰 수 있습니다.
