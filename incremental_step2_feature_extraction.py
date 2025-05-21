#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 새 이미지를 기존 3D 모델에 증분식으로 추가하는 스크립트 - 2단계: 특징 추출 및 매칭

import os
import sys
import torch
import argparse
import time
import numpy as np
from collections import defaultdict

# mast3r 모듈 경로 추가
sys.path.append('/home/jmc/updat3r/mast3r')

from mast3r.model import AsymmetricMASt3R
from mast3r.utils.misc import hash_md5
from mast3r.cloud_opt.sparse_ga import forward_mast3r
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images

# MST와 pairwise_scores 활용하여 가장 가까운 이미지 찾기
def find_closest_images_from_mst(new_img_hash, existing_img_hashes, kinematic_data, top_k=5):
    """
    MST와 pairwise_scores를 활용하여 새 이미지와 가장 관련성 높은 기존 이미지 찾기
    
    Args:
        new_img_hash: 새 이미지 해시
        existing_img_hashes: 기존 이미지 해시 리스트
        kinematic_data: kinematic_data.pth에서 로드한 데이터
        top_k: 반환할 가장 가까운 이미지 수
        
    Returns:
        가장 관련성 높은 이미지 인덱스 리스트
    """
    # pairwise_scores를 사용할 수 있는 경우
    if 'pairwise_scores' in kinematic_data and isinstance(kinematic_data['pairwise_scores'], torch.Tensor):
        pairwise_scores = kinematic_data['pairwise_scores']
        print(f"pairwise_scores 텐서 크기: {pairwise_scores.shape}")
        
        # 연결성을 계산 (각 노드의 엣지 가중치 합)
        n_nodes = pairwise_scores.shape[0]
        node_connectivity = torch.sum(pairwise_scores, dim=1)
        
        # 가장 높은 연결성을 가진 노드 top_k개 선택
        topk_indices = torch.argsort(node_connectivity, descending=True)[:min(top_k, len(node_connectivity))]
        
        print(f"pairwise_scores 기반 top {top_k} 인덱스: {topk_indices.tolist()}")
        return topk_indices.tolist()
    
    # MST를 사용할 수 있는 경우
    elif 'mst' in kinematic_data and isinstance(kinematic_data['mst'], dict):
        mst = kinematic_data['mst']
        root = mst.get('root', 0)  # 루트 노드, 없으면 0을 기본값으로 사용
        edges = mst.get('edges', [])
        
        print(f"MST 정보 - 루트: {root}, 엣지 수: {len(edges)}")
        
        # 각 노드의 연결 수 계산 (MST에서의 중심성)
        node_connections = defaultdict(int)
        for edge in edges:
            if len(edge) >= 2:  # 유효한 엣지인지 확인
                node_connections[edge[0]] += 1
                node_connections[edge[1]] += 1
        
        # root 노드 추가 (항상 포함)
        closest_indices = [root]
        
        # 연결이 많은 상위 노드 추가
        sorted_nodes = sorted(node_connections.items(), key=lambda x: x[1], descending=True)
        for node, _ in sorted_nodes:
            if node not in closest_indices:
                closest_indices.append(node)
                if len(closest_indices) >= top_k:
                    break
        
        print(f"MST 기반 top {len(closest_indices)} 인덱스: {closest_indices}")
        return closest_indices[:top_k]
    
    # 둘 다 없는 경우, 단순히 처음 top_k개 선택
    else:
        print("kinematic_data에 pairwise_scores 또는 mst가 없습니다. 처음 top_k개 이미지 선택.")
        return list(range(min(top_k, len(existing_img_hashes))))


def load_model(model_path=None, device="cuda"):
    """
    MASt3R 모델을 로드하는 함수
    
    Args:
        model_path: 모델 가중치 경로 (None인 경우 기본값 사용)
        device: 장치 (cuda 또는 cpu)
    
    Returns:
        로드된 모델
    """
    print("MASt3R 모델 로드 중...")
    
    # CUDA 메모리 정보 출력
    if device == "cuda" and torch.cuda.is_available():
        print("=== CUDA 정보 ===")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU 개수: {torch.cuda.device_count()}")
        print(f"현재 장치: {torch.cuda.current_device()}")
        print(f"장치 이름: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 메모리 할당: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"CUDA 메모리 캐시: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
        print("==================")
    
    if model_path is None:
        # 기본 모델 경로
        model_path = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    
    # 로컬 경로 확인
    if os.path.isdir("/home/jmc/updat3r/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"):
        model_path = "/home/jmc/updat3r/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    
    print(f"모델 경로: {model_path}")
    
    try:
        model = AsymmetricMASt3R.from_pretrained(model_path)
        print("모델 로드 성공")
        model = model.to(device)
        print(f"모델을 {device}로 이동 성공")
        model.eval()  # 평가 모드로 설정
        print("모델 평가 모드로 설정 성공")
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        print("CPU로 대체합니다.")
        device = "cpu"
        model = AsymmetricMASt3R.from_pretrained(model_path).to(device)
        model.eval()
    
    print("모델 로드 완료")
    return model


def process_new_pairs(pairs_input_paths, new_img_hash, existing_cache_dir, output_cache_dir, model, device="cuda", subsample=8, desc_conf='desc_conf'):
    """
    새 이미지 쌍에 대한 모델 추론 실행
    
    Args:
        pairs_input_paths: 처리할 이미지 경로 쌍의 리스트 [(img1_path, img2_path), ...]
        new_img_hash: 새 이미지 해시 (현재는 직접 사용되지 않으나, 로깅 등에 활용 가능)
        existing_cache_dir: 기존 캐시 디렉토리
        output_cache_dir: 모델 추론 결과를 저장할 캐시 디렉토리
        model: 로드된 모델
        device: 장치
        subsample: 서브샘플링 비율
        desc_conf: 설명자 신뢰도 
        
    Returns:
        처리된 쌍 정보 (딕셔너리): { (img1_path, img2_path) : ((fwd1_path, fwd2_path), corres_path), ... }
    """
    print(f"새 이미지 쌍에 대한 모델 추론 실행 중 (총 {len(pairs_input_paths)}쌍)...")
    
    # MASt3R forward_mast3r 함수 요구 형식으로 입력 변환
    # 이미지 로드 및 데이터 준비
    img_data_cache = {} # 이미지 데이터를 캐싱하여 중복 로드 방지
    pairs_for_model = []
    
    # 원본 이미지 경로와 인덱스 매핑 저장
    # forward_mast3r에 전달되는 pairs_for_model의 idx는 0부터 시작하는 고유 ID여야 함
    # 하지만 반환되는 output_pairs의 instance는 원본 파일 경로를 유지함
    
    # Dust3r 스타일의 이미지 데이터 로드
    all_unique_img_paths = sorted(list(set(p for pair_tuple in pairs_input_paths for p in pair_tuple)))
    
    # load_images 함수를 사용하여 모든 이미지를 한 번에 로드
    # load_images는 각 이미지 경로에 대한 딕셔너리 리스트를 반환
    loaded_images_list = load_images(all_unique_img_paths, size=512, verbose=False)
    
    # 경로를 키로 하는 딕셔너리로 변환하여 쉽게 접근
    for i, img_path in enumerate(all_unique_img_paths):
        img_data = loaded_images_list[i]
        if 'true_shape' not in img_data: # true_shape 보정
            _, _, h, w = img_data['img'].shape
            img_data['true_shape'] = np.array([h, w])
        img_data_cache[img_path] = img_data

    # 모델에 전달할 pairs_for_model 구성
    # MASt3R은 각 이미지에 고유한 idx를 부여해야 함
    # Dust3R의 make_pairs와 유사하게, 이미지 경로 목록을 먼저 만들고,
    # 각 경로에 대해 idx를 부여한 후, 이를 사용하여 쌍을 구성
    
    # 경로별 고유 ID 생성
    path_to_idx_map = {path: i for i, path in enumerate(all_unique_img_paths)}

    for img1_path, img2_path in pairs_input_paths:
        img1_data_item = img_data_cache[img1_path]
        img2_data_item = img_data_cache[img2_path]
        
        pairs_for_model.append([
            {'instance': img1_path, 'idx': path_to_idx_map[img1_path], 
             'img': img1_data_item['img'], 'true_shape': img1_data_item['true_shape']},
            {'instance': img2_path, 'idx': path_to_idx_map[img2_path], 
             'img': img2_data_item['img'], 'true_shape': img2_data_item['true_shape']}
        ])

    if not pairs_for_model:
        print("모델에 전달할 유효한 쌍이 없습니다.")
        return {}

    print(f"MASt3R 모델 추론을 위해 {len(pairs_for_model)}개의 쌍 준비 완료.")

    with torch.no_grad():
        # forward_mast3r는 (pairs, cache_path)를 반환.
        # 여기서 pairs는 처리된 결과 (딕셔너리)
        # {(img_path1, img_path2): ((fwd1_path, fwd2_path), corres_path), ...} 형태
        # cache_path는 사용된 캐시 디렉토리
        output_pairs_dict, used_cache_path = forward_mast3r(
            pairs_for_model, # Dust3R 스타일의 이미지 데이터 리스트를 포함하는 쌍 리스트
            model,
            cache_path=output_cache_dir, # 여기에 결과가 저장됨
            desc_conf=desc_conf,
            device=device,
            subsample=subsample,
            verbose=True # MASt3R 내부 로그 확인용
        )
    
    print(f"MASt3R 모델 추론 완료. 처리된 쌍 수: {len(output_pairs_dict)}. 사용된 캐시: {used_cache_path}")
    
    # forward_mast3r가 반환하는 output_pairs_dict가 이미 원하는 processed_pairs 형식이므로 그대로 사용
    # 키는 (img_path1, img_path2) 튜플
    return output_pairs_dict


def incremental_step2(step1_result_path, model_path=None, device="cuda", top_k_matches=10):
    """
    단계 2: 모델 로드 및 새 이미지 쌍 처리
    
    Args:
        step1_result_path: 단계 1의 결과 파일 경로
        model_path: 모델 가중치 경로 (None인 경우 기본값 사용)
        device: 사용할 장치
        top_k_matches: 새로운 이미지와 매칭할 관련성 높은 기존 이미지의 최대 개수
    
    Returns:
        단계 2의 결과
    """
    print("\n========== 단계 2: 모델 로드 및 새 이미지 쌍 처리 ==========")
    start_time = time.time()
    
    # 단계 1 결과 로드
    step1_result = torch.load(step1_result_path)
    
    # 단계 1 결과에서 필요한 정보 추출
    new_img = step1_result['new_img']
    new_img_hash = step1_result['new_img_hash']
    new_img_path = step1_result['new_img_path']
    existing_imgs = step1_result['existing_imgs'] # 기존 이미지 데이터 리스트 (load_images 결과)
    existing_img_paths = step1_result['existing_img_paths'] # 기존 이미지 경로 리스트
    existing_img_hashes = step1_result['existing_img_hashes']
    existing_data = step1_result['existing_data']
    output_dir = step1_result['output_dir']
    intermediate_dir = os.path.join(output_dir, "intermediate")
    
    try:
        # 캐시 디렉토리 설정
        output_cache_dir = os.path.join(output_dir, "cache")
        os.makedirs(output_cache_dir, exist_ok=True)
        
        existing_cache_dir = None
        optim_data_path = os.path.join(step1_result['existing_data_dir'], 'optimization_data')
        if os.path.isdir(optim_data_path):
            potential_cache_dir = os.path.join(step1_result['existing_data_dir'], "cache")
            if os.path.isdir(potential_cache_dir):
                existing_cache_dir = potential_cache_dir # process_images.py의 캐시
        
        # 1. MASt3R 모델 로드
        model = load_model(model_path, device)
        print(f"모델 로드 완료: {model.__class__.__name__}")
        
        # 2. pairwise_scores 또는 MST 정보를 사용하여 가장 관련성 높은 기존 이미지 찾기
        if 'kinematic_data' in existing_data:
            kinematic_data = existing_data['kinematic_data']
            print("MST/pairwise_scores를 사용하여 가장 관련성 높은 이미지 찾기...")
            closest_indices = find_closest_images_from_mst(
                new_img_hash, existing_img_hashes, kinematic_data, top_k=top_k_matches
            )
            print(f"가장 관련성 높은 {len(closest_indices)}개 이미지 인덱스: {closest_indices}")
        else:
            print("kinematic_data를 찾을 수 없습니다. 처음 10개 이미지 사용")
            closest_indices = list(range(min(top_k_matches, len(existing_img_paths))))
        
        # 3. 선택된 이미지와 새 이미지 간의 쌍 생성
        selected_img_paths = [existing_img_paths[idx] for idx in closest_indices]
        selected_img_hashes = [existing_img_hashes[idx] for idx in closest_indices]
        
        print(f"선택된 이미지 수: {len(selected_img_paths)}")
        
        # 새 이미지와 선택된 이미지들 간의 쌍 생성 (모든 방향)
        pairs_input_paths = []
        for selected_path in selected_img_paths:
            pairs_input_paths.append((new_img_path, selected_path))  # 새 이미지 -> 기존 이미지
            pairs_input_paths.append((selected_path, new_img_path))  # 기존 이미지 -> 새 이미지
        
        print(f"처리할 이미지 쌍 수: {len(pairs_input_paths)}")
        
        # 4. 새 이미지 쌍에 대한 모델 추론 실행
        processed_pairs = process_new_pairs(
            pairs_input_paths,
            new_img_hash, 
            existing_cache_dir, 
            output_cache_dir, 
            model, 
            device=device,
            subsample=8,
            desc_conf='desc_conf'
        )
        
        print(f"모델 추론 완료. 처리된 쌍: {len(processed_pairs)}")
        
        # 5. 캐시 디렉토리에서 특징점 정보 추출
        print("캐시에서 특징점 정보 추출 중...")
        
        # 경로 매핑을 통해 특징점 캐시 찾기
        forward_cache_dir = os.path.join(output_cache_dir, 'forward')
        if not os.path.exists(forward_cache_dir):
            print(f"forward 캐시 디렉토리를 찾을 수 없음: {forward_cache_dir}")
            
            # corres_conf=desc_conf_subsample=8 형식의 디렉토리 이름 찾기
            cache_subdirs = os.listdir(output_cache_dir)
            for subdir in cache_subdirs:
                if subdir.startswith('corres_conf=') and 'subsample=' in subdir:
                    forward_cache_dir = os.path.join(output_cache_dir, subdir)
                    print(f"대체 캐시 디렉토리 사용: {forward_cache_dir}")
                    break
        
        # 새 이미지의 특징점 캐시 찾기
        new_img_feature_files = []
        if os.path.exists(forward_cache_dir):
            for filename in os.listdir(forward_cache_dir):
                if new_img_hash in filename:
                    new_img_feature_files.append(os.path.join(forward_cache_dir, filename))
        
        print(f"새 이미지 특징점 파일 수: {len(new_img_feature_files)}")
        
        # 첫 번째 특징점 파일 로드 (있는 경우)
        new_img_features = None
        if new_img_feature_files:
            try:
                new_img_features = torch.load(new_img_feature_files[0], map_location='cpu')
                print(f"새 이미지 특징점 로드 성공: {os.path.basename(new_img_feature_files[0])}")
            except Exception as e:
                print(f"특징점 로드 오류: {e}")
        
        # 6. detailed_matches 생성
        print("Detailed matches 생성 중...")
        detailed_matches_dir = os.path.join(output_dir, "detailed_matches")
        os.makedirs(detailed_matches_dir, exist_ok=True)
        
        detailed_matches = {}
        for (img1_path, img2_path), ((fwd1_path, fwd2_path), corres_path) in processed_pairs.items():
            # 이미지 해시 구하기
            img1_hash = hash_md5(img1_path)
            img2_hash = hash_md5(img2_path)
            
            # 매칭 결과 로드
            try:
                corres_data = torch.load(corres_path, map_location='cpu')
                # 매칭 정보 저장
                match_file = os.path.join(detailed_matches_dir, f"{img1_hash}-{img2_hash}.pth")
                torch.save(corres_data, match_file)
                detailed_matches[(img1_hash, img2_hash)] = corres_data
                print(f"매칭 저장 성공: {img1_hash}-{img2_hash}")
            except Exception as e:
                print(f"매칭 처리 오류 ({img1_hash}-{img2_hash}): {e}")
        
        # 7. Step3로 넘길 결과 구성
        step2_result = {
            'new_img_hash': new_img_hash,
            'new_img_path': new_img_path,
            'existing_data': existing_data,
            'selected_indices': closest_indices,
            'selected_img_paths': selected_img_paths,
            'selected_img_hashes': selected_img_hashes,
            'processed_pairs': processed_pairs,
            'detailed_matches': detailed_matches,
            'new_img_features': new_img_features,
            'existing_data_dir': step1_result['existing_data_dir'],
            'output_dir': output_dir
        }
        
        # 중간 결과 저장
        torch.save(step2_result, os.path.join(intermediate_dir, "step2_result.pth"))
        
        elapsed_time = time.time() - start_time
        print(f"단계 2 완료: 모델 로드 및 새 이미지 쌍 처리 (소요 시간: {elapsed_time:.2f}초)")
        
        # 결과 요약
        print(f"\n===== 단계 2 결과 요약 =====")
        print(f"처리된 이미지 쌍: {len(processed_pairs)}")
        print(f"생성된 매칭: {len(detailed_matches)}")
        print(f"결과 저장 위치: {os.path.join(intermediate_dir, 'step2_result.pth')}")
        
        return step2_result
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="증분식 3D 재구성 - 단계 2: 특징 추출 및 매칭")
    parser.add_argument("--step1_result", type=str, required=True, 
                        help="단계 1 결과 파일 경로")
    parser.add_argument("--model_path", type=str, default=None, 
                        help="모델 가중치 경로 (None인 경우 기본값 사용)")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="장치 (cuda 또는 cpu)")
    parser.add_argument("--top_k_matches", type=int, default=10, 
                        help="새로운 이미지와 매칭할 관련성 높은 기존 이미지의 최대 개수 (기본값: 10)")
    
    args = parser.parse_args()
    
    # 단계 2 실행
    incremental_step2(
        args.step1_result,
        args.model_path,
        args.device,
        args.top_k_matches
    ) 