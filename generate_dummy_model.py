import torch
import torch.nn as nn
import os

class DummyAudioAI(nn.Module):
    def __init__(self):
        super(DummyAudioAI, self).__init__()
        # 실제 AI 모델처럼 보이기 위해 가벼운 1D 합성곱(Convolution) 레이어를 하나 만듭니다.
        # in_channels=1, out_channels=1, kernel_size=5
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding=2)
        
        # 소리가 깨지는 것을 막기 위해 값을 -1.0 ~ 1.0 사이로 자르는 활성화 함수
        self.activation = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        
        # 필터 가중치를 임의로 살짝 증폭(볼륨 업)시키는 형태의 스무딩 필터로 설정합니다.
        nn.init.constant_(self.conv.weight, 1.2 / 5.0) 
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        # x의 입력 형태는 [batch_size=1, sequence_length] 형태의 2차원 텐서입니다.
        
        # Conv1d는 [batch, channel, length] 형태를 요구하므로 중간에 channel 차원을 하나 추가합니다.
        x = x.unsqueeze(1) 
        
        # GPU에서 연산될 가상의 AI 레이어 통과
        x = self.conv(x)
        x = self.activation(x)
        
        # 다시 원래 형태인 [batch_size=1, sequence_length]로 되돌립니다.
        return x.squeeze(1)

if __name__ == "__main__":
    print("[INFO] 테스트용 가짜 ONNX 모델 생성을 시작합니다...")
    model = DummyAudioAI()
    model.eval()

    # 입력 형태를 잡아주기 위한 가상의 더미 텐서 생성 (길이는 중요하지 않음)
    dummy_input = torch.randn(1, 16000)

    os.makedirs("models", exist_ok=True)
    export_path = os.path.join("models", "light_upsampler.onnx")

    # ONNX 형식으로 모델 추출
    torch.onnx.export(
        model,                      # 실행할 모델 객체
        dummy_input,                # 모델 입력값 (텐서 형태 및 타입 명시용)
        export_path,                # 저장 위치
        export_params=True,         # 모델 파일 안에 학습된 가중치를 저장할지 여부
        opset_version=14,           # ONNX 버전
        do_constant_folding=True,   # 최적화
        input_names=['input'],      # 모델 입력값 이름
        output_names=['output'],    # 모델 출력값 이름
        dynamic_axes={
            # 오디오 음원 길이는 매번 다르기 때문에 가변 길이(Dynamic Axes)를 허용해야 합니다.
            'input': {1: 'sequence_length'}, 
            'output': {1: 'sequence_length'}
        }
    )
    print(f"[SUCCESS] ONNX 모델 생성 완료: {export_path}")
    print("[SUCCESS] 이제 main.py에서 이 모델을 이용해 진짜 GPU 가속 추론이 가능해집니다.")
