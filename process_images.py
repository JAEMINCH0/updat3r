#!/usr/bin/env python3
# 이미지 폴더에서 3D 모델을 생성하는 스크립트

import os
import sys
import torch
import argparse
import tempfile
import numpy as np
import json
from pathlib import Path
from glob import glob
from collections import defaultdict
import shutil

# mast3r 모듈 경로 추가
sys.path.append('/home/jmc/updat3r/mast3r')

from mast3r.model import AsymmetricMASt3R
from mast3r.image_pairs import make_pairs
import mast3r.cloud_opt.sparse_ga as sparse_ga_module
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment, compute_min_spanning_tree
from mast3r.utils.misc import hash_md5

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy, to_cpu


def get_3D_model_from_scene(silent, scene_state, min_conf_thr=2, as_pointcloud=False, 
                           mask_sky=False, clean_depth=False, transparent_cams=False, 
                           cam_size=0.05, TSDF_thresh=0):
    """3D 모델(glb 파일)을 생성하는 함수"""
    from mast3r.demo import _convert_scene_output_to_glb
    
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    # 메모리 설정 최적화
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048,expandable_segments:True,garbage_collection_threshold:0.8"
    
    # 메모리 정리
    torch.cuda.empty_cache()
    
    # 씬에서 최적화된 값 가져오기
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    # 깊이맵, 포즈 및 내부 매개변수에서 3D 포인트 클라우드 생성
    if TSDF_thresh > 0:
        from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
        
        # GPU 메모리 사용량 최적화를 위한 설정
        # 1. 배치 크기 조정 - 메모리에 따라 적절히 조정
        TSDF_batchsize = int(1e6)  # 더 적절한 배치 크기로 설정 (5e6 -> 1e6)
        print(f"현재 CUDA 메모리 사용량 (TSDF 처리 전): {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        torch.cuda.empty_cache()
        print(f"캐시 정리 후 CUDA 메모리 사용량 (TSDF 처리 전): {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
        
        # 2. Half precision (float16) 사용
        original_dtype = scene.pts3d[0].dtype # type: ignore
        for i in range(len(scene.pts3d)): # type: ignore
            scene.pts3d[i] = scene.pts3d[i].half() # type: ignore
            
        print(f"TSDF 처리 시작: 배치 크기 {TSDF_batchsize}, half precision 사용")
        
        # 3. TSDF 처리 최적화 객체 생성
        try:
            tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh, TSDF_batchsize=TSDF_batchsize) # type: ignore
            
            # 4. 메모리 효율적 처리를 위한 구성
            with torch.cuda.amp.autocast(enabled=True):
                pts3d_np, _, confs_np = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
                
            # 원래 dtype으로 복원
            for i in range(len(scene.pts3d)): # type: ignore
                scene.pts3d[i] = scene.pts3d[i].to(original_dtype) # type: ignore
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"TSDF 처리 중 메모리 부족 오류 발생. TSDF를 사용하지 않고 계속 진행합니다.")
                pts3d_np, _, confs_np = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth)) # type: ignore
            else:
                raise e
    else:
        pts3d_np, _, confs_np = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth)) # type: ignore
    
    # 메모리 사용량 표시
    print(f"메모리 사용량 (TSDF 처리 후 또는 TSDF 미사용 시): {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    
    # 신뢰도 임계값 적용
    msk = to_numpy([c > min_conf_thr for c in confs_np])
    
    # 메모리 정리
    torch.cuda.empty_cache()
    
    # 3D 모델 생성
    return _convert_scene_output_to_glb(outfile, rgbimg, pts3d_np, msk, focals, cams2world, 
                                      as_pointcloud=as_pointcloud, transparent_cams=transparent_cams, 
                                      cam_size=cam_size, silent=silent)


class SparseGAState:
    """스파스 글로벌 얼라인먼트 상태를 저장하는 클래스"""
    def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete


def save_optimization_results(output_dir, image_files, scene, pairs, 
                              current_alignment_info, # 이름 변경: alignment_info_global 객체가 전달됨
                              cache_dir_from_sga # sparse_global_alignment 내부의 cache_path
                             ):
    """
    증분식 3D 재구성을 위해 최적화 결과와 중간 데이터를 저장하는 함수
    """
    print("최적화 결과 및 추가 재료 저장 중...")
    optim_dir = os.path.join(output_dir, "optimization_data")
    os.makedirs(optim_dir, exist_ok=True)

    # 1. 기존 정보 저장 (이미지 정보, 카메라 파라미터, 깊이 데이터, 기본 kinematic 데이터, 메타데이터)
    image_info = {
        'image_files': [os.path.abspath(f) for f in image_files],
        'image_hashes': [hash_md5(f) for f in image_files]
    }
    with open(os.path.join(optim_dir, "image_info.json"), 'w') as f:
        json.dump(image_info, f, indent=2)

    camera_params = {
        'intrinsics': to_cpu(scene.intrinsics), # type: ignore
        'cam2w': to_cpu(scene.cam2w), # type: ignore
        'focals': to_cpu(scene.get_focals()), # type: ignore
        'principal_points': to_cpu(scene.get_principal_points()) # type: ignore
    }
    torch.save(camera_params, os.path.join(optim_dir, "camera_params.pth"))

    depth_data = {
        'depthmaps': to_cpu(scene.depthmaps), # type: ignore
        'pts3d': to_cpu(scene.pts3d), # type: ignore
        'pts3d_colors': scene.pts3d_colors if hasattr(scene, 'pts3d_colors') else None # type: ignore
    }
    torch.save(depth_data, os.path.join(optim_dir, "depth_data.pth"))

    pair_info_dict = defaultdict(list)
    if pairs: # pairs가 None이 아닐 경우에만 처리
        for p_idx, p_val in enumerate(pairs): # 더 안전한 순회를 위해 인덱스와 값을 함께 사용
            if len(p_val) == 2 and isinstance(p_val[0], dict) and isinstance(p_val[1], dict):
                img1_instance = p_val[0].get('instance') # get()을 사용하여 KeyError 방지
                img2_instance = p_val[1].get('instance') # get()을 사용하여 KeyError 방지
                if isinstance(img1_instance, str) and isinstance(img2_instance, str):
                    hash1 = hash_md5(img1_instance)
                    hash2 = hash_md5(img2_instance)
                    pair_info_dict[hash1].append(hash2)
                    pair_info_dict[hash2].append(hash1)
                else:
                    print(f"경고: 유효하지 않은 pair 인스턴스 타입 (인덱스 {p_idx}): {type(img1_instance)}, {type(img2_instance)}")
            else:
                print(f"경고: 유효하지 않은 pair 구조 (인덱스 {p_idx}): {p_val}")


    kinematic_data = {
        'pair_info': dict(pair_info_dict),
    }
    if current_alignment_info.mst is not None: # type: ignore
        kinematic_data['mst'] = {
            'root': int(current_alignment_info.mst[0]), # type: ignore
            'edges': [[int(i), int(j)] for i, j in current_alignment_info.mst[1]] # type: ignore
        }
    if current_alignment_info.pairwise_scores is not None: # type: ignore
        kinematic_data['pairwise_scores'] = to_cpu(current_alignment_info.pairwise_scores) # type: ignore

    torch.save(kinematic_data, os.path.join(optim_dir, "kinematic_data.pth"))

    metadata = {
        'n_images': len(image_files),
        'creation_timestamp': str(Path(output_dir).stat().st_mtime),
        'image_size': scene.imgs[0].shape[:2] if scene.imgs and len(scene.imgs) > 0 else None # type: ignore
    }
    with open(os.path.join(optim_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    # 2. 추가 재료 저장
    # 2.1 특징점 및 디스크립터 관련 정보 저장
    # forward_mast3r의 캐시 내용을 복사 (각 이미지별 폴더)
    features_dir = os.path.join(optim_dir, "features_cache")
    os.makedirs(features_dir, exist_ok=True)
    sga_forward_cache = os.path.join(cache_dir_from_sga, "forward") # `sga_capture_info.forward_results_cache_path` 대신 직접 `cache_dir` 사용
    if os.path.exists(sga_forward_cache):
        for img_hash_folder in os.listdir(sga_forward_cache):
            src_folder = os.path.join(sga_forward_cache, img_hash_folder)
            dst_folder = os.path.join(features_dir, img_hash_folder)
            if os.path.isdir(src_folder):
                shutil.copytree(src_folder, dst_folder, dirs_exist_ok=True)
        print(f"특징점/디스크립터 캐시가 {features_dir}에 저장되었습니다.")
    else:
        print(f"경고: 특징점/디스크립터 캐시 폴더를 찾을 수 없습니다: {sga_forward_cache}")


    # 2.2 정준 뷰 관련 정보 저장 (canonical_paths에 해당하는 파일 복사)
    canonical_views_dir = os.path.join(optim_dir, "canonical_views")
    os.makedirs(canonical_views_dir, exist_ok=True)
    if current_alignment_info.canonical_views_paths: # type: ignore
        for src_path in current_alignment_info.canonical_views_paths: # type: ignore
            if os.path.exists(src_path):
                dst_path = os.path.join(canonical_views_dir, os.path.basename(src_path))
                shutil.copy2(src_path, dst_path)
            else:
                print(f"경고: 정준 뷰 파일을 찾을 수 없습니다: {src_path}")
        print(f"정준 뷰 파일들이 {canonical_views_dir}에 저장되었습니다.")
    else:
        print("저장할 정준 뷰 경로 정보가 없습니다.")

    # 2.3 매칭 상세 정보 저장 (correspondence_paths에 해당하는 파일 복사)
    detailed_matches_dir = os.path.join(optim_dir, "detailed_matches")
    os.makedirs(detailed_matches_dir, exist_ok=True)
    if current_alignment_info.correspondence_paths: # type: ignore
        for pair_key, src_path in current_alignment_info.correspondence_paths.items(): # type: ignore
            if os.path.exists(src_path):
                # 파일명을 pair_key (예: "hash1-hash2.pth")로 사용
                dst_path = os.path.join(detailed_matches_dir, f"{pair_key}.pth")
                shutil.copy2(src_path, dst_path)
            else:
                print(f"경고: 매칭 상세 정보 파일을 찾을 수 없습니다: {src_path} (키: {pair_key})")
        print(f"매칭 상세 정보 파일들이 {detailed_matches_dir}에 저장되었습니다.")
    else:
        print("저장할 매칭 상세 정보 경로 정보가 없습니다.")


    print(f"최적화 결과 및 추가 재료가 {optim_dir}에 저장되었습니다.")
    return optim_dir


# 전역 AlignmentCapture 객체 정의 (스크립트 최상단 또는 클래스 정의 바로 다음)
class AlignmentCapture:
    """스파스 글로벌 얼라인먼트 내부 정보를 캡처하기 위한 클래스"""
    def __init__(self):
        self.pairwise_scores = None
        self.mst = None
        self.forward_results_cache_path = None # forward_mast3r의 cache_path
        self.canonical_views_paths = []      # prepare_canonical_data의 canonical_paths
        self.correspondence_paths = {}       # prepare_canonical_data의 tmp_pairs에서 사용된 correspondence 파일 경로들

alignment_info_global = AlignmentCapture() # 전역 캡처 객체

def process_images(image_dir, output_dir, model_name="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric", 
                  device="cuda", image_size=512):
    """이미지 디렉토리를 처리하여 3D 모델을 생성하는 함수"""
    print(f"이미지 디렉토리: {image_dir}")
    print(f"출력 디렉토리: {output_dir}")
    
    ### 1.1 Make Output Directory
    os.makedirs(output_dir, exist_ok=True)
    
    ### 1.2 Load Model
    print("모델 로드 중...")
    if os.path.isfile(model_name):
        weights_path = model_name
    else:
        weights_path = "naver/" + model_name
    
    # 로컬 모델 경로 확인 (선택 사항)
    local_model_path = "/home/jmc/updat3r/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    if os.path.isdir(local_model_path): # 로컬 경로가 존재하면 사용
        weights_path = local_model_path
        print(f"로컬 모델 사용: {weights_path}")
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    
    ### 1.3 Load Images
    print("이미지 로드 중...")
    image_files = sorted(glob(os.path.join(image_dir, "*.jpg")) + 
                         glob(os.path.join(image_dir, "*.jpeg")) + 
                         glob(os.path.join(image_dir, "*.png")))
    print(f"발견된 이미지: {len(image_files)}개")
    
    if len(image_files) < 2:
        print("이미지가 2개 이상 필요합니다.")
        return None # 오류 발생 시 None 반환
    
    imgs_data = load_images(image_files, size=image_size, verbose=True) # imgs -> imgs_data로 변경 (MASt3R API 변경 가능성 고려)
    
    ### 1.4 Make Cache Directory
    cache_dir = os.path.join(output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # 전역 alignment_info_global 객체에 forward_results_cache_path 설정
    alignment_info_global.forward_results_cache_path = os.path.join(cache_dir, "forward")


    print("이미지 쌍 생성 중...")
    pairs = make_pairs(imgs_data, scene_graph="complete", prefilter=None, symmetrize=True) # imgs -> imgs_data
    
    ### 2.2 Run Sparse Global Alignment
    print("스파스 글로벌 얼라인먼트 실행 중...")
    
    # sparse_global_alignment 함수 모킹하여 중간 결과 캡처
    original_compute_mst = sparse_ga_module.compute_min_spanning_tree
    original_prepare_canonical_data = sparse_ga_module.prepare_canonical_data # 원본 함수 저장
    
    def capturing_compute_mst(pairwise_scores):
        # 전역 alignment_info_global 사용
        alignment_info_global.pairwise_scores = pairwise_scores # type: ignore
        mst = original_compute_mst(pairwise_scores)
        alignment_info_global.mst = mst # type: ignore
        return mst
    
    def capturing_prepare_canonical_data(imgs_paths_list, tmp_pairs_dict, subsample, **kwargs):
        # correspondence_paths 캡처 로직
        if isinstance(tmp_pairs_dict, dict): # tmp_pairs_dict가 사전 타입인지 확인
            for (img1_path, img2_path), data_tuple in tmp_pairs_dict.items():
                if isinstance(data_tuple, tuple) and len(data_tuple) == 2:
                    fwd_paths, corres_path_val = data_tuple # (path1, path2), corres_path
                    if isinstance(fwd_paths, tuple) and len(fwd_paths) == 2: # (path1, path2) 형식 확인
                        key = f"{hash_md5(str(img1_path))}-{hash_md5(str(img2_path))}" # 경로를 문자열로 변환
                        # 전역 alignment_info_global 사용
                        alignment_info_global.correspondence_paths[key] = str(corres_path_val) # 경로를 문자열로 변환
                    else:
                        print(f"경고: 잘못된 forward_paths 형식: {fwd_paths}")
                else:
                    print(f"경고: 잘못된 tmp_pairs_dict 값 형식: {data_tuple}")
        else:
            print(f"경고: tmp_pairs_dict가 사전 타입이 아님: {type(tmp_pairs_dict)}")

        # 원본 함수 호출 - **kwargs를 사용하여 모든 원래 인수 전달
        tmp_pairs_canon, pairwise_scores, canonical_views, canonical_paths_list, preds_21 = \
            original_prepare_canonical_data(imgs_paths_list, tmp_pairs_dict, subsample, **kwargs)
        
        # canonical_views_paths 캡처
        # 전역 alignment_info_global 사용
        if isinstance(canonical_paths_list, list):
            alignment_info_global.canonical_views_paths.extend([str(p) for p in canonical_paths_list]) # 경로를 문자열로 변환
        else:
             print(f"경고: canonical_paths_list가 리스트 타입이 아님: {type(canonical_paths_list)}")
        return tmp_pairs_canon, pairwise_scores, canonical_views, canonical_paths_list, preds_21
    
    # 모킹 함수 적용
    sparse_ga_module.compute_min_spanning_tree = capturing_compute_mst
    sparse_ga_module.prepare_canonical_data = capturing_prepare_canonical_data # 모킹 함수 적용
    
    # 스파스 글로벌 얼라인먼트 실행
    scene = sparse_global_alignment(
        image_files, pairs, cache_dir, model,
        lr1=0.07, niter1=300,  # 코스 얼라인먼트를 위한 학습률과 반복 횟수
        lr2=0.01, niter2=300,  # 정제를 위한 학습률과 반복 횟수
        device=device,
        opt_depth=True,
        shared_intrinsics=False,
        matching_conf_thr=0.0
    )
    
    # 원래 함수 복원
    sparse_ga_module.compute_min_spanning_tree = original_compute_mst
    sparse_ga_module.prepare_canonical_data = original_prepare_canonical_data # 원본 함수 복원
    
    ### 3. 새로 추가: 최적화 결과 저장 (전역 alignment_info_global 전달)
    optim_dir = save_optimization_results(
        output_dir,
        image_files,
        scene,
        pairs,
        alignment_info_global, # 전역 객체 전달
        cache_dir # sga_forward_cache 계산을 위해 cache_dir 전달
    )
    
    ### 4. 3D 모델 생성
    outfile_name = os.path.join(output_dir, "scene.glb")
    
    # 씬 상태 생성
    scene_state = SparseGAState(scene, False, cache_dir, outfile_name)
    
    # 3D 모델 생성
    print("3D 모델 생성 중...")
    outfile = get_3D_model_from_scene(
        False, scene_state,
        min_conf_thr=2.0,
        as_pointcloud=True,
        clean_depth=True,
        cam_size=0.2,
        TSDF_thresh=0.05
    )
    
    if outfile and os.path.exists(outfile): # outfile이 None이 아니고 파일이 실제로 존재하는지 확인
        print(f"3D 모델이 성공적으로 생성되었습니다: {outfile}")
        print(f"증분식 3D 재구성을 위한 추가 데이터가 {optim_dir}에 저장되었습니다.")
        return outfile
    else:
        print(f"3D 모델 생성 실패. outfile: {outfile}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="이미지 폴더에서 3D 모델 생성")
    parser.add_argument("--image_dir", type=str, required=True, help="이미지 디렉토리 경로")
    parser.add_argument("--output_dir", type=str, required=True, help="출력 디렉토리 경로")
    parser.add_argument("--model_name", type=str, 
                        default="MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                        help="모델 이름 또는 경로")
    parser.add_argument("--device", type=str, default="cuda", help="pytorch 장치")
    parser.add_argument("--image_size", type=int, default=512, help="이미지 크기")
    
    args = parser.parse_args()
    
    process_images(
        args.image_dir,
        args.output_dir,
        args.model_name,
        args.device,
        args.image_size
    )