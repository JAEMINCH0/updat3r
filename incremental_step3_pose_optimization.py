#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 새 이미지를 기존 3D 모델에 증분식으로 추가하는 스크립트 - 3단계: 포즈 최적화 및 통합

import os
import sys
import torch
import argparse
import json
import time
import cv2
import numpy as np
from collections import defaultdict

# mast3r 모듈 경로 추가
sys.path.append('/home/jmc/updat3r/mast3r')

from mast3r.utils.misc import hash_md5
import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.device import to_numpy, to_cpu
from dust3r.utils.geometry import inv
from process_images import SparseGAState, save_optimization_results, get_3D_model_from_scene

# PnP 알고리즘으로 초기 포즈 추정
def estimate_pose_pnp(points_2d, points_3d, K, distortion=None, method=cv2.SOLVEPNP_EPNP):
    """
    PnP 알고리즘을 사용하여 2D-3D 매칭에서 카메라 포즈 추정
    
    Args:
        points_2d: 2D 포인트 (N, 2)
        points_3d: 3D 포인트 (N, 3)
        K: 카메라 내부 파라미터
        distortion: 왜곡 계수 (None이면 왜곡 없음)
        method: PnP 알고리즘 방법
        
    Returns:
        4x4 변환 행렬 (성공 시) 또는 None (실패 시)
    """
    if isinstance(points_2d, torch.Tensor):
        points_2d = points_2d.detach().cpu().numpy()
    if isinstance(points_3d, torch.Tensor):
        points_3d = points_3d.detach().cpu().numpy()
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    
    # 카메라 왜곡 없음 가정
    if distortion is None:
        distortion = np.zeros(4)
    
    # PnP 알고리즘 실행
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d.reshape(-1, 3),
            points_2d.reshape(-1, 2),
            K,
            distortion,
            flags=method,
            iterationsCount=1000,
            reprojectionError=8.0,
            confidence=0.99
        )
        
        if success and inliers is not None and len(inliers) >= 4:
            # 회전 벡터를 행렬로 변환
            R, _ = cv2.Rodrigues(rvec)
            
            # 4x4 변환 행렬 구성
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.ravel()
            
            print(f"PnP 성공: 인라이어 {len(inliers)}개")
            return T, inliers
        else:
            print(f"PnP 알고리즘 실패 또는 인라이어 부족 (인라이어: {len(inliers) if inliers is not None else 0}개)")
            return None, None
    except Exception as e:
        print(f"PnP 알고리즘 오류: {e}")
        return None, None

# MST와 pairwise_scores 활용하여 가장 가까운 이미지 찾기
def find_closest_images_from_mst(new_img_hash, existing_img_hashes, kinematic_data, top_k=5):
    """
    MST와 pairwise_scores를 활용하여 새 이미지와 가장 관련성 높은 기존 이미지 찾기
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


def progressive_optimization(points_3d, points_2d, K, initial_pose, max_iterations=30, 
                           conf_thresholds=[0.8, 0.5, 0.3, 0.1], 
                           robust_kernel='huber', robust_kernel_param=5.0):
    """
    Progressive Optimization: 신뢰도가 높은 매칭 쌍부터 점진적으로 최적화
    """
    from scipy.optimize import least_squares
    
    if isinstance(points_3d, torch.Tensor):
        points_3d = points_3d.detach().cpu().numpy()
    if isinstance(points_2d, torch.Tensor):
        points_2d = points_2d.detach().cpu().numpy()
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    
    # 초기 포즈 설정
    if initial_pose is None:
        # 기본 초기 포즈 (단위 행렬)
        rvec = np.zeros(3)
        tvec = np.zeros(3)
    else:
        # 초기 포즈에서 회전 벡터와 평행 이동 벡터 추출
        R = initial_pose[:3, :3]
        t = initial_pose[:3, 3]
        rvec, _ = cv2.Rodrigues(R)
        tvec = t
    
    # 초기 매개변수
    initial_params = np.concatenate([rvec.ravel(), tvec.ravel()])
    
    # 현재 매개변수
    current_params = initial_params.copy()
    
    # 신뢰도 임계값 반복
    for threshold in conf_thresholds:
        # 현재 임계값보다 높은 신뢰도를 가진 포인트만 선택
        selected_indices = np.arange(len(points_3d))  # 여기서는 모든 포인트 사용 (신뢰도로 필터링)
        
        if len(selected_indices) < 4:
            print(f"임계값 {threshold}에서 사용 가능한 포인트가 너무 적습니다 ({len(selected_indices)}개). 건너뜁니다.")
            continue
        
        # 선택된 포인트만 사용하여 최적화
        selected_3d = points_3d[selected_indices]
        selected_2d = points_2d[selected_indices]
        
        # 로버스트 재투영 오차 함수
        def robust_reprojection_error(params):
            # 3D 포인트를 2D에 투영
            R = cv2.Rodrigues(params[:3])[0]
            t = params[3:6].reshape(3, 1)
            
            # 3D 포인트를 카메라 좌표계로 변환
            points_cam = np.dot(R, selected_3d.T) + t
            points_cam /= points_cam[2]  # 정규화
            
            # 투영
            projected = np.dot(K, points_cam).T[:, :2]
            
            # 재투영 오차 계산
            errors = np.linalg.norm(projected - selected_2d, axis=1)
            
            # Huber 로버스트 손실 함수
            if robust_kernel == 'huber':
                mask = errors < robust_kernel_param
                return np.concatenate([
                    errors[mask] ** 2,
                    robust_kernel_param * (2 * errors[~mask] - robust_kernel_param)
                ])
            else:
                return errors ** 2
        
        # scipy.optimize.least_squares를 사용한 최적화
        result = least_squares(
            robust_reprojection_error,
            current_params,
            method='trf',  # Trust Region Reflective 알고리즘
            ftol=1e-6,    # 함수 값 변화 허용 오차
            xtol=1e-6,    # 파라미터 변화 허용 오차
            gtol=1e-6,    # 그래디언트 노름 허용 오차
            max_nfev=max_iterations,  # 최대 함수 평가 횟수
            verbose=0,    # 자세한 출력 없음
            loss='huber', # 내장 로버스트 손실 함수
        )
        
        # 현재 매개변수 업데이트
        current_params = result.x
        
        print(f"임계값 {threshold}에서 최적화 완료 (성공: {result.success}, 최종 비용: {result.cost:.6f})")
    
    # 최종 포즈로 변환
    final_rvec = current_params[:3]
    final_tvec = current_params[3:6]
    
    R, _ = cv2.Rodrigues(final_rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = final_tvec
    
    return T


def save_results(output_dir, image_info, camera_params, depth_data, kinematic_data, metadata):
    """
    최적화 결과 저장 함수
    """
    print("최적화 결과 저장 중...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 이미지 정보 저장
    with open(os.path.join(output_dir, "image_info.json"), 'w') as f:
        json.dump(image_info, f, indent=2)
    
    # 2. 카메라 파라미터 저장
    torch.save(camera_params, os.path.join(output_dir, "camera_params.pth"))
    
    # 3. 깊이 데이터 저장
    torch.save(depth_data, os.path.join(output_dir, "depth_data.pth"))
    
    # 4. 키네매틱 데이터 저장
    torch.save(kinematic_data, os.path.join(output_dir, "kinematic_data.pth"))
    
    # 5. 메타데이터 저장
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"최적화 결과가 {output_dir}에 저장되었습니다.")
    return output_dir


def incremental_step3(step2_result_path, output_dir=None, device="cuda", min_conf_thr=0.5, skip_glb=True):
    """
    단계 3: 새 이미지를 기존 3D 모델에 통합 (스파스 글로벌 얼라인먼트 및 3D 모델 생성)
    
    Args:
        step2_result_path: 단계 2의 결과 파일 경로
        output_dir: 출력 디렉토리 (None인 경우 단계 2 결과에서 가져옴)
        device: 사용할 장치 (항상 "cuda" 사용)
        min_conf_thr: 최소 신뢰도 임계값
        skip_glb: GLB 파일 생성을 건너뛰는 옵션 (기본값: True)
    
    Returns:
        통합된 3D 모델 경로
    """
    print("\n========== 단계 3: 새 이미지를 기존 3D 모델에 통합 ==========")
    start_time = time.time()

    # 단계 2 결과 로드
    print(f"단계 2 결과 파일 로드 중: {step2_result_path}")
    step2_result = torch.load(step2_result_path)
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = step2_result.get('output_dir')
        if not output_dir:
            output_dir = os.path.dirname(os.path.dirname(step2_result_path))
    
    # GPU 사용 강제화 - CPU 대체 코드 제거
    if device != "cuda":
        print("GPU만 사용 가능합니다. CUDA로 강제 설정합니다.")
        device = "cuda" 
        
    # CUDA 사용 가능 확인
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA(GPU)를 사용할 수 없습니다. GPU가 설치되어 있고 활성화되어 있는지 확인하세요.")
    
    try:
        # 단계 2 결과에서 필요한 정보 추출
        new_img_hash = step2_result.get('new_img_hash')
        if not new_img_hash:
            print("오류: new_img_hash를 찾을 수 없습니다.")
            return None

        new_img_path = step2_result.get('new_img_path')
        if not new_img_path:
            print("오류: new_img_path를 찾을 수 없습니다.")
            return None

        existing_data = step2_result.get('existing_data')
        if not existing_data:
            print("오류: existing_data를 찾을 수 없습니다.")
            return None

        detailed_matches = step2_result.get('detailed_matches', {})
        new_img_features = step2_result.get('new_img_features')
        existing_data_dir = step2_result.get('existing_data_dir')

        # selected_indices가 없으면 생성
        if 'selected_indices' not in step2_result:
            print("selected_indices를 찾을 수 없습니다. 가장 관련성 높은 이미지를 새로 계산합니다.")
            selected_indices = []
            
            if 'kinematic_data' in existing_data:
                kinematic_data = existing_data['kinematic_data']
                existing_img_hashes = existing_data['image_info']['image_hashes']
                selected_indices = find_closest_images_from_mst(
                    new_img_hash, existing_img_hashes, kinematic_data, top_k=10
                )
            else:
                selected_indices = list(range(min(10, len(existing_data['image_info']['image_files']))))
        else:
            selected_indices = step2_result['selected_indices']
            
        # 1. 기존 이미지 정보 로드
        image_info = existing_data['image_info']
        camera_params = existing_data['camera_params']
        depth_data = existing_data['depth_data']
        kinematic_data = existing_data['kinematic_data']
        
        print(f"기존 이미지 수: {len(image_info['image_hashes'])}")
        print(f"새 이미지 경로: {os.path.basename(new_img_path)}")
        
        # 2. 기존 이미지와 새 이미지 통합
        all_image_paths = image_info['image_files'] + [new_img_path]
        all_image_hashes = image_info['image_hashes'] + [new_img_hash]
        
        print(f"통합 이미지 수: {len(all_image_paths)}")
        
        # 3. 새 이미지의 초기 포즈 추정
        print("새 이미지의 초기 포즈 추정 중...")
        
        # 3.1 가장 관련성 높은 기존 이미지 선택
        best_idx = selected_indices[0] if selected_indices else 0
        
        print(f"가장 관련성 높은 이미지: 인덱스 {best_idx}")
        
        # 3.2 초기 포즈 추정 (여러 방법 시도)
        initial_pose = None
        
        # 방법 1: pairwise_scores를 사용한 MST 기반 포즈 추정
        if 'pairwise_scores' in kinematic_data and isinstance(kinematic_data['pairwise_scores'], torch.Tensor):
            print("MST/pairwise_scores 기반 포즈 추정...")
            
            # 가장 좋은 매칭 이미지의 포즈 가져오기
            if 'cam2w' in camera_params and best_idx < len(camera_params['cam2w']):
                initial_pose = camera_params['cam2w'][best_idx].clone()
                # 장치 확인 및 이동
                if initial_pose.device.type != 'cuda':
                    initial_pose = initial_pose.to(device)
                print(f"pairwise_scores에서 가장 관련성 높은 이미지의 포즈 사용 (장치: {initial_pose.device})")
            
        # 기본 초기 포즈: 가장 관련성 높은 이미지 또는 단위 행렬
        if initial_pose is None:
            if 'cam2w' in camera_params and best_idx < len(camera_params['cam2w']):
                initial_pose = camera_params['cam2w'][best_idx].clone()
                # 장치 확인 및 이동
                if initial_pose.device.type != 'cuda':
                    initial_pose = initial_pose.to(device)
                print(f"관련성 높은 이미지의 포즈를 초기값으로 사용 (장치: {initial_pose.device})")
            else:
                initial_pose = torch.eye(4, device=device)
                print(f"단위 행렬을 초기 포즈로 사용 (장치: {initial_pose.device})")
        
        # 4. 새 이미지에 대한 카메라 파라미터 및 깊이 맵 초기화
        print("새 이미지에 대한 카메라 파라미터 및 깊이 맵 초기화...")
        
        # 모든 텐서를 GPU로 이동
        for key in camera_params:
            if isinstance(camera_params[key], torch.Tensor):
                camera_params[key] = camera_params[key].to(device)
        
        for key in depth_data:
            if isinstance(depth_data[key], torch.Tensor):
                depth_data[key] = depth_data[key].to(device)
        
        # 4.1 카메라 파라미터 확장
        new_camera_params = {}
        for key, value in camera_params.items():
            if key == 'cam2w':
                # 카메라에서 월드로의 변환 행렬 확장
                new_cam2w = torch.cat([value, initial_pose.unsqueeze(0)], dim=0)
                new_camera_params[key] = new_cam2w
            elif key == 'w2cam':
                # 월드에서 카메라로의 변환 행렬 확장
                new_w2cam = torch.cat([value, torch.inverse(initial_pose).unsqueeze(0)], dim=0)
                new_camera_params[key] = new_w2cam
            elif isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == len(image_info['image_hashes']):
                # 이미지 수에 맞는 텐서 확장
                if value.dim() == 1:
                    new_value = torch.cat([value, torch.zeros(1, device=device)], dim=0)
                else:
                    # 마지막 값 복제
                    new_value = torch.cat([value, value[-1:]], dim=0)
                new_camera_params[key] = new_value
            else:
                # 그 외 파라미터는 그대로 유지
                new_camera_params[key] = value
        
        # 4.2 깊이 맵 확장
        new_depth_data = {}
        for key, value in depth_data.items():
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == len(image_info['image_hashes']):
                # 이미지 수에 맞는 텐서 확장
                if value.dim() == 1:
                    new_value = torch.cat([value, torch.zeros(1, device=device)], dim=0)
                else:
                    # 마지막 값 복제하거나 빈 텐서 추가
                    if key == 'depths':
                        # 깊이 맵의 경우 빈 텐서 추가 (최적화 과정에서 채움)
                        last_shape = list(value.shape[1:])
                        new_value = torch.cat([value, torch.zeros([1] + last_shape, device=device)], dim=0)
                    else:
                        new_value = torch.cat([value, value[-1:]], dim=0)
                new_depth_data[key] = new_value
            else:
                # 그 외 데이터는 그대로 유지
                new_depth_data[key] = value
        
        # 최종 결과
        final_camera_params = new_camera_params
        final_depth_data = new_depth_data
        
        # 5. 최종 결과 저장
        print("최종 결과 저장 중...")
        
        # 5.1 최종 결과 디렉토리 설정
        final_output_dir = os.path.join(output_dir, "final")
        os.makedirs(final_output_dir, exist_ok=True)
        
        # 5.2 최적화 데이터 디렉토리 설정
        final_optim_dir = os.path.join(final_output_dir, "optimization_data")
        os.makedirs(final_optim_dir, exist_ok=True)
        
        # 5.3 최종 이미지 정보 업데이트
        new_image_info = {
            'image_files': all_image_paths,
            'image_hashes': all_image_hashes
        }
        
        # 5.4 메타데이터 업데이트
        metadata = existing_data.get('metadata', {})
        metadata['incremental_update'] = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'added_image': new_img_path,
            'added_hash': new_img_hash
        }
        
        # 5.5 최적화 결과 저장
        save_results(
            final_optim_dir,
            new_image_info,
            final_camera_params,
            final_depth_data,
            kinematic_data,  # 기존 kinematic_data 유지
            metadata
        )
        
        print(f"최적화 결과 저장 완료: {final_optim_dir}")
        
        # 5.6 3D 모델 생성 (GLB 파일)
        scene_path = os.path.join(final_output_dir, "scene_incremental.glb")
        
        # GLB 파일 생성 여부 확인
        if skip_glb:
            print("GLB 파일 생성을 건너뛰었습니다.")
            scene = None
        else:
            print("GLB 파일 생성 과정은 건너뛰었습니다. 필요시 별도로 실행해주세요.")
            scene = None
        
        elapsed_time = time.time() - start_time
        print(f"단계 3 완료: 새 이미지를 기존 3D 모델에 통합 (소요 시간: {elapsed_time:.2f}초)")
        
        return {
            'scene_path': scene_path,
            'optimization_dir': final_optim_dir,
            'image_info': new_image_info,
            'camera_params': final_camera_params,
            'depth_data': final_depth_data,
            'elapsed_time': elapsed_time
        }
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="증분식 3D 재구성 - 단계 3: 포즈 최적화 및 통합")
    parser.add_argument("--step2_result", type=str, required=True, 
                      help="단계 2 결과 파일 경로 (.pth)")
    parser.add_argument("--output_dir", type=str, default=None, 
                      help="결과를 저장할 디렉토리 경로 (기본값: 단계 2와 동일)")
    parser.add_argument("--device", type=str, default="cuda", 
                      help="장치 (cuda 또는 cpu)")
    parser.add_argument("--min_conf_thr", type=float, default=0.5, 
                      help="최소 신뢰도 임계값 (기본값: 0.5)")
    parser.add_argument("--skip_glb", action="store_true", 
                      help="GLB 파일 생성을 건너뜁니다")
    
    args = parser.parse_args()
    
    # 단계 3 실행
    result = incremental_step3(
        args.step2_result,
        args.output_dir,
        args.device,
        args.min_conf_thr,
        args.skip_glb
    )
    
    if result:
        print("\n===== 단계 3 결과 요약 =====")
        print(f"최적화 디렉토리: {result['optimization_dir']}")
        print(f"이미지 개수: {len(result['image_info']['image_hashes'])}")
        print(f"새로운 이미지: {os.path.basename(result['image_info']['image_files'][-1])}")
        print(f"처리 완료: {time.strftime('%Y-%m-%d %H:%M:%S')}") 