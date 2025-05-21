#!/bin/bash

# 스크립트가 있는 디렉토리로 이동 (선택 사항, 필요에 따라 경로 조정)
# cd /path/to/your/script_directory

# 입력 이미지 디렉토리와 출력 디렉토리 설정
IMAGE_DATA_DIR="/home/jmc/updat3r/data/1_Baalashamin/image_data"
OUTPUT_DIR="/home/jmc/updat3r/output/Baalashamin_3d_processed_ingredients" # 새 출력 폴더

# 기존 출력 디렉토리가 있으면 삭제 (선택 사항)
# rm -rf "$OUTPUT_DIR"

# Python 스크립트 실행
python /home/jmc/updat3r/process_images.py \
    --image_dir "$IMAGE_DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric" \
    --device "cuda" \
    --image_size 512

echo "처리 완료. 결과는 다음 위치에 저장되었습니다: $OUTPUT_DIR"