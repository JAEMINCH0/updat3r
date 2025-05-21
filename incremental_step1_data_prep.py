#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 새 이미지를 기존 3D 모델에 증분식으로 추가하는 스크립트 - 1단계: 데이터 준비

import os
import sys
import torch
import argparse
import json
import numpy as np
import time
from PIL import Image
import hashlib

# mast3r 모듈 경로 추가
sys.path.append('/home/jmc/updat3r/mast3r')

from mast3r.utils.misc import hash_md5
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images

# 유틸리티 함수
def load_optimization_data(optimization_dir):
    """
    기존 최적화 데이터를 로드하는 함수
    
    Args:
        optimization_dir: 최적화 데이터가 저장된 디렉토리 경로
        
    Returns:
        최적화 데이터를 포함하는 사전
    """
    print(f"기존 최적화 데이터 로드 중: {optimization_dir}")
    
    # 1. 이미지 정보 로드
    with open(os.path.join(optimization_dir, "image_info.json"), 'r') as f:
        image_info = json.load(f)
    
    # 2. 카메라 파라미터 로드
    camera_params = torch.load(os.path.join(optimization_dir, "camera_params.pth"))
    
    # 3. 깊이 데이터 로드
    depth_data = torch.load(os.path.join(optimization_dir, "depth_data.pth"))
    
    # 4. 키네매틱 데이터 로드
    kinematic_data = torch.load(os.path.join(optimization_dir, "kinematic_data.pth"))
    
    # 5. 메타데이터 로드
    with open(os.path.join(optimization_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    return {
        'image_info': image_info,
        'camera_params': camera_params,
        'depth_data': depth_data,
        'kinematic_data': kinematic_data,
        'metadata': metadata
    }


def load_new_image(image_path, image_size=512):
    """
    새 이미지를 로드하고 처리하는 함수
    
    Args:
        image_path: 이미지 파일 경로
        image_size: 이미지 크기 조정을 위한 값
        
    Returns:
        처리된 이미지와 관련 정보
    """
    print(f"새 이미지 로드 중: {image_path}")
    
    # 이미지 파일이 존재하는지 확인
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없음: {image_path}")
    
    # 이미지 로드
    img = load_images([image_path], size=image_size, verbose=True)[0]
    img_hash = hash_md5(image_path)
    
    print(f"이미지 로드 완료: {image_path} (해시: {img_hash})")
    return img, img_hash, image_path


def incremental_step1(existing_data_dir, new_image_path, output_dir, image_size=512):
    """
    새 이미지 추가의 첫 번째 단계: 데이터 로드 및 준비
    
    Args:
        existing_data_dir: 기존 3D 모델 데이터 디렉토리
        new_image_path: 새 이미지 파일 경로
        output_dir: 결과를 저장할 디렉토리 경로
        image_size: 이미지 크기
        
    Returns:
        step1_result: 단계 1 결과 데이터
    """
    print("========== 단계 1: 데이터 로드 및 준비 ==========")
    start_time = time.time()
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 최적화 데이터 디렉토리 확인
    optimization_dir = os.path.join(existing_data_dir, "optimization_data")
    if not os.path.isdir(optimization_dir):
        raise FileNotFoundError(f"최적화 데이터 디렉토리를 찾을 수 없음: {optimization_dir}")
    
    # 2. 기존 최적화 데이터 로드
    existing_data = load_optimization_data(optimization_dir)
    
    # 3. 기존 이미지 경로 및 해시 추출
    existing_img_paths = existing_data['image_info']['image_files']
    existing_img_hashes = existing_data['image_info']['image_hashes']
    
    # 4. 새 이미지 로드
    new_img, new_img_hash, new_img_path = load_new_image(new_image_path, image_size)
    
    # 5. 기존 이미지 로드
    print("기존 이미지 로드 중...")
    existing_imgs = load_images(existing_img_paths, size=image_size, verbose=True)
    
    # 6. 결과 저장
    step1_result = {
        'existing_data': existing_data,
        'new_img': new_img,
        'new_img_hash': new_img_hash,
        'new_img_path': new_img_path,
        'existing_imgs': existing_imgs,
        'existing_img_paths': existing_img_paths,
        'existing_img_hashes': existing_img_hashes,
        'existing_data_dir': existing_data_dir,
        'output_dir': output_dir
    }
    
    # 중간 결과 저장 (나중에 디버깅 용이하게)
    os.makedirs(os.path.join(output_dir, "intermediate"), exist_ok=True)
    torch.save(step1_result, os.path.join(output_dir, "intermediate", "step1_result.pth"))
    
    elapsed_time = time.time() - start_time
    print(f"단계 1 완료: 데이터 로드 및 준비 (소요 시간: {elapsed_time:.2f}초)")
    
    # 결과 요약
    print(f"\n===== 단계 1 결과 요약 =====")
    print(f"기존 이미지 개수: {len(existing_img_paths)}")
    print(f"새 이미지: {os.path.basename(new_img_path)} (해시: {new_img_hash})")
    print(f"결과 저장 위치: {os.path.join(output_dir, 'intermediate', 'step1_result.pth')}")
    
    return step1_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="증분식 3D 재구성 - 단계 1: 데이터 준비")
    parser.add_argument("--existing_data", type=str, required=True, 
                        help="기존 3D 모델 데이터 디렉토리 경로 (optimization_data 상위 폴더)")
    parser.add_argument("--new_image", type=str, required=True, 
                        help="추가할 새 이미지 파일 경로")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="결과를 저장할 디렉토리 경로")
    parser.add_argument("--image_size", type=int, default=512, 
                        help="이미지 크기 (기본값: 512)")
    
    args = parser.parse_args()
    
    # 단계 1 실행
    incremental_step1(
        args.existing_data,
        args.new_image,
        args.output_dir,
        args.image_size
    ) 