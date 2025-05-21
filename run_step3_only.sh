#!/bin/bash
# 단계 3만 실행하기 위한 스크립트

# CUDA 메모리 관리 설정
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

# conda 환경 활성화
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mast3r

# 입력 인자 검증
if [ $# -lt 3 ]; then
    echo "사용법: $0 <step2_결과_파일> <출력_디렉토리> [skip_glb]"
    echo "예시: $0 /home/jmc/updat3r/output/Baalashamin_3d_incremental/intermediate/step2_result.pth /home/jmc/updat3r/output/Baalashamin_3d_incremental_fixed true"
    echo "skip_glb: GLB 파일 생성 건너뛰기 (true/false, 기본값: false)"
    exit 1
fi

STEP2_RESULT=$1
OUTPUT_DIR=$2
SKIP_GLB=${3:-false}

# CUDA 메모리 사용량 출력
echo "CUDA 메모리 정보:"
nvidia-smi

# GLB 파일 생성 스킵 옵션 전달
SKIP_GLB_OPTION=""
if [ "$SKIP_GLB" = "true" ]; then
    SKIP_GLB_OPTION="--skip_glb"
    echo "GLB 파일 생성을 건너뜁니다."
fi

# 단계 3만 실행
python test_step3.py \
    --step2_result "$STEP2_RESULT" \
    --output_dir "$OUTPUT_DIR" \
    --device "cuda" \
    --min_conf_thr 1.5 \
    $SKIP_GLB_OPTION

# 결과 보고
echo "작업 완료!"
echo "출력 디렉토리: $OUTPUT_DIR"
echo "CUDA 메모리 상태:"
nvidia-smi 