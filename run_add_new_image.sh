#!/bin/bash
# CUDA 메모리 관리와 TSDF 처리를 위한 최적화된 환경 변수를 설정하는 실행 스크립트

# CUDA 메모리 관리 설정
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,expandable_segments:True"

# TSDF 처리를 위한 반정밀도(half-precision) 활성화
export AMP_ENABLED=1

# 메모리 부족 시 OOM 에러로 크래시되지 않게 하고 재할당 시도
export CUDA_OOM_RETRY=1

# 최적의 CUDA 성능을 위한 설정
export CUDA_LAUNCH_BLOCKING=0

# 애플리케이션 실행
# 예시:
# ./run_add_new_image.sh /home/jmc/updat3r/output/Baalashamin_3d /home/jmc/updat3r/data/1_Baalashamin/new_image.jpg /home/jmc/updat3r/output/Baalashamin_3d_incremental

# 입력 인자 검증
if [ $# -lt 3 ]; then
    echo "사용법: $0 <기존_데이터_경로> <새_이미지_경로> <출력_디렉토리> [단계] [skip_glb]"
    echo "예시: $0 /home/jmc/updat3r/output/Baalashamin_3d /home/jmc/updat3r/data/1_Baalashamin/new_image.jpg /home/jmc/updat3r/output/Baalashamin_3d_incremental 3 true"
    echo "단계: 0=모든 단계, 1=데이터 로드, 2=모델 로드 및 쌍 처리, 3=3D 모델에 통합"
    echo "skip_glb: 단계 3에서 GLB 파일 생성 건너뛰기 (기본값: false)"
    exit 1
fi

EXISTING_DATA=$1
NEW_IMAGE=$2
OUTPUT_DIR=$3
STEP=${4:-0}  # 기본값은 0 (모든 단계 실행)
SKIP_GLB=${5:-false}  # 기본값은 false (GLB 파일 생성)

# conda 환경 활성화 (필요한 경우)
CONDA_ENV="mast3r"
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $CONDA_ENV
fi

# CUDA 메모리 사용량 출력
echo "CUDA 메모리 정보:"
nvidia-smi

# 로드 시 중간 결과 확인
if [ "$STEP" -eq 3 ] && [ -d "$OUTPUT_DIR/intermediate" ]; then
    echo "단계 3만 실행하기 위해 중간 결과 확인 중..."
    if [ -f "$OUTPUT_DIR/intermediate/step2_result.pth" ]; then
        echo "단계 2의 결과 파일을 찾았습니다. 단계 3만 실행합니다."
    else
        echo "경고: 단계 2의 결과 파일을 찾을 수 없습니다. 전체 프로세스를 실행합니다."
        STEP=0
    fi
fi

# GLB 파일 생성 스킵 옵션 전달
SKIP_GLB_OPTION=""
if [ "$SKIP_GLB" = "true" ]; then
    SKIP_GLB_OPTION="--skip_glb"
    echo "GLB 파일 생성을 건너뜁니다."
fi

# 스크립트 실행 (각 단계별 파일 실행)
if [ "$STEP" -eq 0 ] || [ "$STEP" -eq 1 ]; then
    echo "\n========== 실행: 단계 1 - 데이터 로드 및 준비 ==========\n"
    python /home/jmc/updat3r/incremental_step1_data_prep.py \
        --existing_data "$EXISTING_DATA" \
        --new_image "$NEW_IMAGE" \
        --output_dir "$OUTPUT_DIR" \
        --image_size 512
    
    if [ $? -ne 0 ]; then
        echo "오류: 단계 1 실행 중 문제가 발생했습니다."
        exit 1
    fi
fi

if [ "$STEP" -eq 0 ] || [ "$STEP" -eq 2 ]; then
    STEP1_RESULT="$OUTPUT_DIR/intermediate/step1_result.pth"
    
    # 단계 1 결과 확인
    if [ ! -f "$STEP1_RESULT" ]; then
        echo "오류: 단계 1 결과 파일($STEP1_RESULT)을 찾을 수 없습니다."
        exit 1
    fi
    
    echo "\n========== 실행: 단계 2 - 특징 추출 및 매칭 ==========\n"
    python /home/jmc/updat3r/incremental_step2_feature_extraction.py \
        --step1_result "$STEP1_RESULT" \
        --device "cuda" \
        --top_k_matches 10
    
    if [ $? -ne 0 ]; then
        echo "오류: 단계 2 실행 중 문제가 발생했습니다."
        exit 1
    fi
fi

if [ "$STEP" -eq 0 ] || [ "$STEP" -eq 3 ]; then
    STEP2_RESULT="$OUTPUT_DIR/intermediate/step2_result.pth"
    
    # 단계 2 결과 확인
    if [ ! -f "$STEP2_RESULT" ]; then
        echo "오류: 단계 2 결과 파일($STEP2_RESULT)을 찾을 수 없습니다."
        exit 1
    fi
    
    echo "\n========== 실행: 단계 3 - 포즈 최적화 및 3D 모델 통합 ==========\n"
    python /home/jmc/updat3r/incremental_step3_pose_optimization.py \
        --step2_result "$STEP2_RESULT" \
        --output_dir "$OUTPUT_DIR" \
        --device "cuda" \
        --min_conf_thr 1.5 \
        $SKIP_GLB_OPTION
    
    if [ $? -ne 0 ]; then
        echo "오류: 단계 3 실행 중 문제가 발생했습니다."
        exit 1
    fi
fi

# 단계 4: GLB 파일 생성 (상세 3D 모델)
if [ "$STEP" -eq 0 ] || [ "$STEP" -eq 4 ]; then
    # GLB 파일 생성 건너뛰기 여부 확인
    if [ "$SKIP_GLB" = "true" ]; then
        echo "\n========== 스킵: 단계 4 - GLB 파일 생성 ==========\n"
        echo "GLB 파일 생성을 건너뛰습니다."
    else
        echo "\n========== 실행: 단계 4 - GLB 파일 생성 ==========\n"
        
        # 최적화 데이터 디렉토리 경로
        OPTIM_DIR="$OUTPUT_DIR/final/optimization_data"
        GLB_PATH="$OUTPUT_DIR/final/scene_incremental.glb"
        
        # 최적화 데이터 확인
        if [ ! -d "$OPTIM_DIR" ]; then
            echo "오류: 최적화 데이터 디렉토리($OPTIM_DIR)를 찾을 수 없습니다."
            exit 1
        fi
        
        # GLB 파일 생성
        python /home/jmc/updat3r/incremental_step4_glb.py \
            --optimization_dir "$OPTIM_DIR" \
            --output_glb "$GLB_PATH" \
            --filter_noise \
            --new_image_path "$NEW_IMAGE_PATH"
        
        if [ $? -ne 0 ]; then
            echo "오류: 단계 4 실행 중 문제가 발생했습니다."
            exit 1
        fi
        
        echo "GLB 파일 생성 완료: $GLB_PATH"
    fi
fi

# 결과 보고
echo "작업 완료!"
echo "출력 디렉토리: $OUTPUT_DIR"
echo "CUDA 메모리 상태:"
nvidia-smi 