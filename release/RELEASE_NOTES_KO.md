# Resonix AI v0.9.1 릴리즈 노트

클리핑처럼 들릴 수 있는 출력 왜곡을 줄이기 위한 핫픽스입니다.

## 수정 사항

- 라우드니스 정규화 시 순간 피크를 고려하도록 개선
- 평균 음량은 낮지만 순간 피크가 높은 음원에서 과도한 게인 상승 방지
- 기존 tanh 기반 리미터를 linked peak limiter로 교체
- 기본 리미터 ceiling을 더 보수적으로 조정
- 최종 WAV 저장 전 sample peak guard 추가
- 출력 트루피크 헤드룸 보호 강화

## 사용 방법

1. `ResonixAI-v0.9.1-Windows.zip` 다운로드
2. 압축 해제
3. `ResonixAI-Windows` 폴더 안의 `ResonixAI.exe` 실행

## 주의 사항

- `ResonixAI.exe`만 따로 실행하지 마세요.
- `_internal` 폴더가 exe와 같은 위치에 있어야 합니다.
- Windows SmartScreen 경고가 뜰 수 있습니다.
