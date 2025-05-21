#!/usr/bin/env python3
# MASt3R Habitat 기반 고급 3D 렌더링 및 GLB 파일 생성기

import os
import sys
import torch
import numpy as np
import trimesh
import copy
import json
import quaternion
import cv2
import collections
import PIL.Image
from pathlib import Path
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation

# mast3r 모듈 경로 추가
sys.path.append('/home/jmc/updat3r/mast3r')

# Habitat 관련 모듈 임포트
# 필요한 경우 아래 임포트를 활성화하세요
try:
    import habitat_sim
    HABITAT_AVAILABLE = True
except ImportError:
    HABITAT_AVAILABLE = False
    print("경고: habitat_sim을 임포트할 수 없습니다. 일부 기능이 제한됩니다.")

try:
    from mast3r.dust3r.datasets_preprocess.habitat.habitat_renderer.habitat_sim_envmaps_renderer import HabitatEnvironmentMapRenderer
    from mast3r.dust3r.datasets_preprocess.habitat.habitat_renderer.multiview_crop_generator import HabitatMultiviewCrops
    from mast3r.dust3r.datasets_preprocess.habitat.habitat_renderer import projections, projections_conversions
    HABITAT_RENDERER_AVAILABLE = True
except ImportError:
    HABITAT_RENDERER_AVAILABLE = False
    print("경고: MASt3R Habitat 렌더러 모듈을 임포트할 수 없습니다. 일부 고급 렌더링 기능이 제한됩니다.")

# MASt3R 스타일 OpenGL 카메라 좌표계 변환 매트릭스
OPENGL = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

# 카메라 색상 프리셋
CAM_COLORS = [
    [252, 233, 79],   # 노란색
    [138, 226, 52],   # 녹색
    [114, 159, 207],  # 파란색
    [173, 127, 168],  # 보라색
    [239, 41, 41],    # 빨간색
    [233, 185, 110],  # 주황색
]

# 필요한 유틸리티 함수
def to_numpy(tensor):
    """텐서를 NumPy 배열로 변환"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, list) and len(tensor) > 0 and isinstance(tensor[0], torch.Tensor):
        return [t.detach().cpu().numpy() for t in tensor]
    return tensor

def load_optimization_data(optimization_dir):
    """
    최적화 데이터를 로드하는 함수
    
    Args:
        optimization_dir: 최적화 데이터 디렉토리 경로
        
    Returns:
        카메라 파라미터와 깊이 데이터
    """
    print(f"최적화 데이터 로드 중: {optimization_dir}")
    
    # 카메라 파라미터 로드
    camera_params_path = os.path.join(optimization_dir, "camera_params.pth")
    if not os.path.exists(camera_params_path):
        raise FileNotFoundError(f"카메라 파라미터 파일을 찾을 수 없음: {camera_params_path}")
    
    camera_params = torch.load(camera_params_path)
    
    # 깊이 맵 데이터 로드
    depth_data_path = os.path.join(optimization_dir, "depth_data.pth")
    if not os.path.exists(depth_data_path):
        raise FileNotFoundError(f"깊이 데이터 파일을 찾을 수 없음: {depth_data_path}")
    
    depth_data = torch.load(depth_data_path)
    
    # 이미지 정보 로드
    image_info_path = os.path.join(optimization_dir, "image_info.json")
    if os.path.exists(image_info_path):
        import json
        with open(image_info_path, 'r') as f:
            image_info = json.load(f)
    else:
        image_info = None
    
    return camera_params, depth_data, image_info

# --- 지오메트릭 변환 유틸리티 함수 ---
def geotrf(transform, pts):
    """포인트를 지정된 변환으로 변환하는 유틸리티 함수"""
    return (transform[:3, :3] @ pts.T).T + transform[:3, 3]

# --- 포인트 클라우드를 메시로 변환하는 함수 ---
def pts3d_to_trimesh(img, pts3d, mask=None, max_points=100000, subsample_factor=None):
    """
    이미지, 3D 포인트, 및 마스크를 사용하여 색상이 입혀진 기본 메시 생성
    
    Args:
        img: 이미지 데이터 (H, W, 3)
        pts3d: 3D 포인트 데이터 (H, W, 3)
        mask: 유효한 포인트 마스크 (H, W), None이면 모든 포인트 사용
        max_points: 처리할 최대 포인트 수 (너무 큼 경우 다운샘플링 적용)
        subsample_factor: 원본 포인트의 배수로 다운샘플링 (예: 10은 10번째 포인트만 사용)
    
    Returns:
        trimesh.Trimesh 객체 또는 trimesh.PointCloud
    """
    print(f"3D 포인트 클라우드 크기: {pts3d.shape}")
    
    if mask is None:
        mask = np.ones(pts3d.shape[:2], dtype=bool)
    
    # 유효한 포인트만 필터링 (다른 메소드 사용 - 마스크 멘뷰 생성)
    valid_mask = mask & np.all(np.isfinite(pts3d), axis=-1)
    
    # 원본 플랫 배열 크기 추정 및 다운샘플링 필요여부 놀정
    flat_size = np.sum(valid_mask)
    need_sampling = flat_size > max_points
    
    if need_sampling:
        print(f"경고: 너무 많은 포인트 ({flat_size:,}). 다운샘플링 적용...")
        
        # 자동 다운샘플링 인자 계산
        if subsample_factor is None:
            subsample_factor = max(2, int(flat_size / max_points) + 1)
            print(f"다운샘플링 인자: {subsample_factor} (예상 포인트 수: ~{flat_size/subsample_factor:,.0f})")
        
        # 새로운 소크기 마스크 생성
        h, w = valid_mask.shape
        subsample_mask = np.zeros_like(valid_mask)
        
        # 그리드 기반 다운샘플링
        for i in range(0, h, subsample_factor):
            for j in range(0, w, subsample_factor):
                if i < h and j < w and valid_mask[i, j]:
                    subsample_mask[i, j] = True
        
        # 새 마스크 적용
        valid_mask = subsample_mask
        sampled_size = np.sum(valid_mask)
        print(f"다운샘플링 후 포인트 수: {sampled_size:,}")
        
        # 포인트가 여전히 너무 많으면 포인트 클라우드로 처리
        if sampled_size > max_points * 2:
            print(f"다운샘플링 후에도 포인트가 너무 많음 ({sampled_size:,}). 메시 대신 포인트 클라우드 반환")
            pts_filtered = pts3d[valid_mask]
            color_filtered = img[valid_mask]
            if color_filtered.dtype != np.uint8:
                color_filtered = (color_filtered * 255).astype(np.uint8)
            return trimesh.PointCloud(vertices=pts_filtered, colors=color_filtered)
    
    # 필터링된 데이터 추출
    y_indices, x_indices = np.where(valid_mask)
    pts_filtered = pts3d[valid_mask]
    color_filtered = img[valid_mask]
    
    # 처리할 포인트 개수
    n = len(pts_filtered)
    if n < 4:  # 메시 생성을 위해 최소한 몇 개의 포인트가 필요함
        print("포인트 개수가 부족하여 기본 메시 반환")
        return trimesh.Trimesh(vertices=pts_filtered, faces=[])
        
    # 색상 변환 (필요시)
    if color_filtered.dtype != np.uint8:
        color_filtered = (color_filtered * 255).astype(np.uint8)
    
    try:
        # 상대 좌표 생성 (2D 표면에 대해 삼각화)
        points_2d = np.column_stack([x_indices, y_indices])
        
        print(f"삼각분할 시도 중 (포인트 수: {n:,})")
        
        # Delaunay 삼각분할 실행
        from scipy.spatial import Delaunay
        try:
            tri = Delaunay(points_2d)
            faces = tri.simplices
            
            # 메시 생성
            mesh = trimesh.Trimesh(
                vertices=pts_filtered,
                faces=faces,
                vertex_colors=color_filtered
            )
            print(f"삼각분할 성공: {len(faces):,} 개의 삼각형 생성")
            return mesh
        except Exception as e:
            print(f"삼각분할 실패, 포인트 클라우드로 반환: {e}")
            return trimesh.PointCloud(vertices=pts_filtered, colors=color_filtered)
    except Exception as e:
        print(f"처리 오류, 포인트 클라우드로 반환: {e}")
        return trimesh.PointCloud(vertices=pts_filtered, colors=color_filtered)  # 포인트 클라우드만 반환

def cat_meshes(meshes):
    """
    여러 메시를 하나로 결합
    
    Args:
        meshes: 메시 리스트
    
    Returns:
        결합된 메시 데이터 디터셔너리
    """
    if not meshes:
        return {"vertices": np.zeros((0, 3)), "faces": np.zeros((0, 3), dtype=int)}
    
    # 비어있지 않은 메시만 필터링
    meshes = [m for m in meshes if len(m.vertices) > 0]
    if not meshes:
        return {"vertices": np.zeros((0, 3)), "faces": np.zeros((0, 3), dtype=int)}
    
    vertices = []
    faces = []
    vertex_colors = []
    offset = 0
    
    for mesh in meshes:
        n = len(mesh.vertices)
        vertices.append(mesh.vertices)
        faces.append(mesh.faces + offset)
        vertex_colors.append(mesh.visual.vertex_colors[:, :3] if hasattr(mesh.visual, 'vertex_colors') else np.ones((n, 3), dtype=np.uint8) * 200)
        offset += n
    
    return {
        "vertices": np.vstack(vertices),
        "faces": np.vstack(faces),
        "vertex_colors": np.vstack(vertex_colors)
    }

# --- 카메라 및 이미지 프레임 시각화 함수 ---
def add_scene_cam(scene, pose_c2w, edge_color, image=None, focal=None, imsize=None, screen_width=0.03, is_new_image=False):
    """
    씬에 카메라와 이미지 프레임 추가
    
    Args:
        scene: trimesh.Scene 객체
        pose_c2w: 카메라->월드 변환 행렬
        edge_color: 카메라 에지 색상
        image: 카메라 이미지 (선택사항)
        focal: 촉점거리 (선택사항)
        imsize: 이미지 크기 (W, H) (선택사항)
        screen_width: 화면 크기 스케일
        is_new_image: 새로 추가된 이미지인지 여부 (새 이미지는 빠간색으로 강조표시)
    """
    if image is not None:
        image = np.asarray(image)
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255*image)
    elif imsize is not None:
        W, H = imsize
    elif focal is not None:
        H = W = focal / 1.1
    else:
        H = W = 1

    if isinstance(focal, np.ndarray):
        focal = focal[0]
    if not focal:
        focal = min(H,W) * 1.1  # 기본값

    # 카메라 프러스텀 생성 (콘 형태)
    height = max(screen_width/10, focal * screen_width / H)
    width = screen_width * 0.5**0.5
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height  # 콘의 끝점을 광학 중심에 맞춤
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W/H
    transform = pose_c2w @ OPENGL @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)

    # 이미지 프레임 생성
    if image is not None:
        vertices = geotrf(transform, cam.vertices[[4, 5, 1, 3]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
        img = trimesh.Trimesh(vertices=vertices, faces=faces)
        uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        img.visual = trimesh.visual.TextureVisuals(uv_coords, image=PIL.Image.fromarray(image))
        scene.add_geometry(img)

    # 카메라 프러스텀 메시 생성
    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler('z', np.deg2rad(2)).as_matrix()
    vertices = np.r_[cam.vertices, 0.95*cam.vertices, geotrf(rot2, cam.vertices)]
    vertices = geotrf(transform, vertices)
    faces = []
    
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)
        a3, b3, c3 = face + 2*len(cam.vertices)

        # 프러스텀 에지 추가
        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b, b3))
        faces.append((a, a3, c))
        faces.append((c3, b, c))

    # 배롱 없음
    faces += [(c, b, a) for a, b, c in faces]

    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # 색상 설정 (새 이미지는 더 굵고 분명한 빠간색으로 표시)
    if is_new_image:
        # 빠간색 강조, 불투명한 색상
        display_color = [255, 30, 30, 255]  # 더 진한 빠간색, 불투명
    else:
        # 기본 색상, 반투명
        display_color = edge_color + [150]  # 기본 색상에 반투명하게 설정
    
    cam.visual.face_colors = np.array(display_color)
    scene.add_geometry(cam)

# --- MASt3R/DUSt3R 스타일의 노이즈 필터링 및 렌더링 함수들 ---

def filter_noise_dust3r(points, colors, centroid=None, distance_percentile=0.99):
    """
    Dust3R의 denoise 방식을 사용한 포인트 클라우드 노이즈 제거
    
    Args:
        points: 포인트 클라우드 (N, 3)
        colors: 포인트 색상 (N, 3)
        centroid: 중심점 (None이면 중앙값 사용)
        distance_percentile: 거리 임계값 백분위 (기본값: 0.99)
        
    Returns:
        필터링된 포인트와 색상
    """
    if len(points) < 4:  # 최소 포인트 수 필요
        return points, colors
    
    # 중심점 계산 (중앙값 사용)
    if centroid is None:
        centroid = np.median(points, axis=0)
    
    # 중심점으로부터의 거리 계산
    dist_to_centroid = np.linalg.norm(points - centroid, axis=1)
    
    # 거리 임계값 (백분위 사용)
    dist_thr = np.quantile(dist_to_centroid, distance_percentile)
    
    # 임계값보다 가까운 포인트만 유지
    valid = (dist_to_centroid < dist_thr)
    
    print(f"Dust3R 스타일 필터링: {len(points)} -> {np.sum(valid)} 포인트 ({np.sum(valid)/len(points)*100:.2f}%)")
    
    return points[valid], colors[valid]

# 포인트 클라우드 중첩 필터링 함수 제거 - Dust3R 스타일만 사용

# 의미론적 필터링(높이 기반) 함수 제거 - Dust3R 스타일만 사용

def filter_noise_combined(points, colors, filter_type="dust3r"):
    """
    노이즈 제거 - Dust3R 스타일만 사용
    
    Args:
        points: 포인트 클라우드 (N, 3)
        colors: 포인트 색상 (N, 3)
        filter_type: 필터링 유형 ("dust3r")
        
    Returns:
        필터링된 포인트와 색상
    """
    print(f"원본 포인트 수: {len(points)}")
    print("노이즈 필터링 시작 (Dust3R 스타일만 사용)...")
    
    # Dust3R 스타일 필터링만 사용
    filtered_points, filtered_colors = filter_noise_dust3r(
        points, colors, distance_percentile=0.99
    )
    
    print(f"필터링 후 포인트 수: {len(filtered_points)}")
    return filtered_points, filtered_colors

def create_pointcloud_glb(output_path, pts3d_data, camera_poses=None, cam_size=0.05, image_info=None, 
                          filter_noise=True, pts3d_colors=None, as_pointcloud=False, force_pointcloud=False):
    """
    포인트 클라우드와 카메라 포즈에서 GLB 파일을 직접 생성 (MASt3R 스타일)
    
    Args:
        output_path: 출력 GLB 파일 경로
        pts3d_data: 3D 포인트 데이터 리스트
        camera_poses: 카메라 포즈 행렬 리스트
        cam_size: 카메라 모델 크기
        image_info: 이미지 정보
        filter_noise: 노이즈 필터링 활성화 여부
        pts3d_colors: 3D 포인트 색상 데이터 (None이면 이미지에서 색상 추출)
        as_pointcloud: 포인트 클라우드로 표시할지 여부 (False면 메시로 변환)
        force_pointcloud: 강제로 포인트 클라우드 모드 사용 (대용량 데이터에 유용)
    """
    print(f"MASt3R 스타일 GLB 파일 생성 시작: {output_path}")
    
    # 1. 데이터를 NumPy로 변환
    pts3d_np = to_numpy(pts3d_data)
    camera_poses_np = to_numpy(camera_poses) if camera_poses is not None else None
    pts3d_colors_np = to_numpy(pts3d_colors) if pts3d_colors is not None else None
    
    # 2. 이미지 로드
    imgs = []
    focals = []
    
    if image_info and 'image_files' in image_info:
        from PIL import Image
        
        # 새 이미지 경로 추출 (강조 표시용)
        new_image_path = None
        if 'new_images' in image_info:
            new_image_path = image_info.get('new_images', [])[0] if image_info.get('new_images', []) else None
        
        for img_path in image_info['image_files']:
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                    img = np.array(img)
                    if img.ndim == 2:  # 그레이스케일 이미지
                        img = np.stack([img, img, img], axis=2)
                    elif img.shape[2] == 4:  # RGBA 이미지
                        img = img[:, :, :3]
                    imgs.append(img)
                    
                    # 초점 거리 설정 (이미지 크기 기반 추정)
                    focal = min(img.shape[0], img.shape[1]) * 1.1
                    focals.append(focal)
                except Exception as e:
                    print(f"이미지 로드 오류: {e}")
                    # 더미 이미지와 초점거리 추가
                    imgs.append(np.ones((512, 512, 3), dtype=np.uint8) * 200)
                    focals.append(512 * 1.1)
            else:
                # 더미 이미지와 초점거리 추가
                imgs.append(np.ones((512, 512, 3), dtype=np.uint8) * 200)
                focals.append(512 * 1.1)
    
    # 3. 새 Scene 생성
    scene = trimesh.Scene()
    
    # 원본 포인트 클라우드 사이즈 추정 및 자동 전략 결정
    estimated_size = 0
    if pts3d_np is not None:
        for pts in pts3d_np:
            if pts is not None:
                if len(pts.shape) >= 2:
                    estimated_size += np.prod(pts.shape[:2])
    
    # 대형 데이터셋 감지 (10,000,000 이상의 포인트)
    if estimated_size > 10000000 and not force_pointcloud:
        print(f"경고: 아주 큰 포인트 클라우드 감지 ({estimated_size:,} 포인트)")
        print("강제로 포인트 클라우드 모드로 전환 (메모리 오버플로우 방지)")
        force_pointcloud = True
    
    # 4. 포인트 클라우드나 메시 생성
    if pts3d_np is not None:
        if as_pointcloud or force_pointcloud:
            # 포인트 클라우드로 표시
            # 포인트 클라우드 데이터 준비
            all_pts = []
            all_colors = []
            
            # 색상 데이터 소스 결정
            use_stored_colors = pts3d_colors_np is not None
            
            for i, pts in enumerate(pts3d_np):
                if pts is None or len(pts) == 0:
                    continue
                    
                # 포인트 형태 확인 및 처리
                if len(pts.shape) > 2:  # (H, W, 3) 형태
                    h, w = pts.shape[:2]
                    pts_flat = pts.reshape(-1, 3)
                else:  # 이미 (N, 3) 형태
                    pts_flat = pts
                
                # 유효한 포인트만 필터링
                valid_mask = np.isfinite(pts_flat.sum(axis=1))
                valid_pts = pts_flat[valid_mask]
                
                # 색상 데이터 준비
                if use_stored_colors and i < len(pts3d_colors_np) and pts3d_colors_np[i] is not None:
                    # 저장된 색상 사용
                    colors_np = pts3d_colors_np[i]
                    if len(colors_np.shape) > 2:  # (H, W, 3) 형태
                        colors_flat = colors_np.reshape(-1, 3)
                        valid_colors = colors_flat[valid_mask]
                    else:  # 이미 (N, 3) 형태
                        valid_colors = colors_np[valid_mask] if len(colors_np) == len(pts_flat) else np.ones((len(valid_pts), 3), dtype=np.uint8) * 200
                elif i < len(imgs):
                    # 이미지에서 색상 데이터 추출
                    img = imgs[i]
                    
                    # 기본 색상
                    valid_colors = np.ones((len(valid_pts), 3), dtype=np.uint8) * 200
                    
                    if len(pts.shape) > 2:  # 2D 그리드 구조가 있을 때
                        h, w = pts.shape[:2]
                        if h * w == len(pts_flat) and h > 0 and w > 0:
                            # 이미지에서 색상 매핑
                            img_h, img_w = img.shape[:2]
                            scale_h, scale_w = img_h / h, img_w / w
                            
                            colors_2d = np.zeros((h, w, 3), dtype=np.uint8)
                            for y in range(h):
                                for x in range(w):
                                    img_y = min(int(y * scale_h), img_h-1)
                                    img_x = min(int(x * scale_w), img_w-1)
                                    colors_2d[y, x] = img[img_y, img_x]
                            
                            colors = colors_2d.reshape(-1, 3)
                            valid_colors = colors[valid_mask]
                else:
                    # 기본 색상 설정
                    valid_colors = np.ones((len(valid_pts), 3), dtype=np.uint8) * 200
                
                # 데이터 추가
                all_pts.append(valid_pts)
                all_colors.append(valid_colors)
            
            # 모든 포인트 데이터 결합
            if all_pts and all_colors:
                combined_pts = np.vstack(all_pts)
                combined_colors = np.vstack(all_colors)
                
                # 노이즈 필터링
                if filter_noise and len(combined_pts) > 100:
                    combined_pts, combined_colors = filter_noise_combined(
                        combined_pts, combined_colors
                    )
                
                # 포인트 클라우드 추가
                point_cloud = trimesh.PointCloud(vertices=combined_pts, colors=combined_colors)
                scene.add_geometry(point_cloud)
        else:
            # 메시로 변환하여 표시 (MASt3R 스타일)
            print("3D 포인트를 삼각분할 메시로 변환")
            
            # 마스크 생성
            masks = []
            for pts in pts3d_np:
                if pts is None or len(pts) == 0:
                    masks.append(None)
                    continue
                
                if len(pts.shape) > 2:  # (H, W, 3) 형태
                    mask = np.isfinite(pts.sum(axis=2))
                else:  # (N, 3) 형태
                    mask = np.isfinite(pts.sum(axis=1)).reshape(-1, 1)
                masks.append(mask)
            
            # 메시 생성
            meshes = []
            for i, (pts, mask) in enumerate(zip(pts3d_np, masks)):
                if pts is None or mask is None or len(pts) == 0:
                    continue
                if i < len(imgs):
                    print(f"이미지 {i}에 대한 메시 생성")
                    mesh = pts3d_to_trimesh(imgs[i], pts, mask)
                    meshes.append(mesh)
            
            # 메시 결합 및 추가
            if meshes:
                print(f"생성된 메시 수: {len(meshes)}")
                combined_mesh = trimesh.Trimesh(**cat_meshes(meshes))
                scene.add_geometry(combined_mesh)
    
    # 5. 카메라 및 이미지 프레임 추가
    if camera_poses_np is not None and len(imgs) > 0:
        # 새 이미지 경로 확인 (있다면 강조 표시용)
        new_image_path = None
        if image_info and 'new_images' in image_info and image_info['new_images']:
            new_image_path = image_info['new_images'][0]
        
        print(f"카메라 및 이미지 프레임 추가: {len(camera_poses_np)}개")
        
        for i, pose in enumerate(camera_poses_np):
            if pose is None or not np.isfinite(pose).all():
                continue
            
            # 카메라 색상 결정
            cam_color = CAM_COLORS[i % len(CAM_COLORS)]
            
            # 새 이미지 여부 확인
            is_new_image = False
            if new_image_path and i < len(image_info['image_files']) and image_info['image_files'][i] == new_image_path:
                is_new_image = True
                print(f"새 이미지 강조 표시: {new_image_path}")
            
            # 카메라 이미지 설정
            img = imgs[i] if i < len(imgs) else None
            focal = focals[i] if i < len(focals) else None
            
            # 카메라 및 이미지 프레임 추가
            add_scene_cam(
                scene, 
                pose, 
                cam_color, 
                img, 
                focal, 
                imsize=img.shape[1::-1] if img is not None else None,
                screen_width=cam_size,
                is_new_image=is_new_image
            )
    
    # 5. GLB 파일로 내보내기
    try:
        # 씬 내보내기
        exported_file = scene.export(output_path)
        print(f"GLB 파일 생성 성공: {output_path}")
        return output_path
    except Exception as e:
        print(f"GLB 파일 생성 오류: {e}")
        return None

# Habitat 환경 맵 렌더러 클래스 (간소화 버전)
class SimplifiedHabitatRenderer:
    """Habitat 기반 렌더링 시스템의 간소화된 버전"""
    def __init__(self, scene_path=None):
        self.scene_path = scene_path
        self.sim = None
        self.renderer_initialized = False
    
    def initialize(self, width=512, height=512):
        """렌더러 초기화"""
        if not HABITAT_AVAILABLE or not HABITAT_RENDERER_AVAILABLE:
            print("Habitat 시뮬레이터를 사용할 수 없어 초기화를 건너뜁니다.")
            return False
            
        try:
            # Habitat 시뮬레이터 설정
            backend_cfg = habitat_sim.SimulatorConfiguration()
            backend_cfg.scene_id = self.scene_path if self.scene_path else ""
            backend_cfg.enable_physics = False
            
            # 에이전트 설정
            agent_cfg = habitat_sim.agent.AgentConfiguration()
            
            # 센서 설정
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = "rgba_camera"
            sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            sensor_spec.resolution = [height, width]
            sensor_spec.position = [0.0, 0.0, 0.0]
            
            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth_camera"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [height, width]
            depth_sensor_spec.position = [0.0, 0.0, 0.0]
            
            agent_cfg.sensor_specifications = [sensor_spec, depth_sensor_spec]
            
            # 시뮬레이터 설정 구성
            sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
            
            # 시뮬레이터 초기화
            self.sim = habitat_sim.Simulator(sim_cfg)
            
            self.renderer_initialized = True
            print("Habitat 시뮬레이터 초기화 성공")
            return True
        except Exception as e:
            print(f"Habitat 시뮬레이터 초기화 오류: {e}")
            self.renderer_initialized = False
            return False
    
    def render_view(self, position, rotation):
        """특정 위치와 회전에서 뷰 렌더링"""
        if not self.renderer_initialized or self.sim is None:
            print("렌더러가 초기화되지 않았습니다.")
            return None
            
        try:
            # 에이전트 상태 설정
            agent = self.sim.get_agent(0)
            agent_state = habitat_sim.AgentState()
            agent_state.position = position
            agent_state.rotation = rotation
            agent.set_state(agent_state)
            
            # 관측 얻기
            observations = self.sim.get_sensor_observations()
            
            return observations
        except Exception as e:
            print(f"뷰 렌더링 오류: {e}")
            return None
    
    def close(self):
        """시뮬레이터 정리"""
        if self.sim is not None:
            self.sim.close()
            self.sim = None
            self.renderer_initialized = False

# 고급 3D 뷰 생성 및 렌더링 함수
def create_enhanced_3d_views(camera_params, image_info, optimization_dir, render_quality="high"):
    """MASt3R 스타일의 고급 3D 뷰 생성
    
    Args:
        camera_params: 카메라 파라미터
        image_info: 이미지 정보
        optimization_dir: 최적화 데이터 디렉토리
        render_quality: 렌더링 품질 ('low', 'medium', 'high')
    
    Returns:
        렌더링된 뷰 정보 딕셔너리
    """
    # 하비탯 렌더러 사용 가능 여부 확인
    if not HABITAT_AVAILABLE or not HABITAT_RENDERER_AVAILABLE:
        print("고급 렌더링에 필요한 Habitat 모듈을 사용할 수 없습니다.")
        print("기본 렌더링 방식으로 진행합니다.")
        return None
    
    # 카메라 포즈 준비
    camera_poses = camera_params.get('cam2w', None)
    if camera_poses is None:
        print("카메라 포즈 정보가 없습니다.")
        return None
    
    # 해상도 설정 (품질에 따라)
    if render_quality == "low":
        resolution = (256, 256)
    elif render_quality == "medium":
        resolution = (512, 512)
    else:  # high
        resolution = (1024, 1024)
    
    # 렌더러 초기화
    print(f"MASt3R 스타일 {render_quality} 품질 3D 렌더링 시작...")
    renderer = SimplifiedHabitatRenderer()
    initialization_success = renderer.initialize(width=resolution[0], height=resolution[1])
    
    if not initialization_success:
        print("렌더러 초기화 실패. 기본 렌더링으로 진행합니다.")
        return None
    
    # 렌더링 결과 저장 경로 생성
    render_output_dir = os.path.join(optimization_dir, "mast3r_renders")
    os.makedirs(render_output_dir, exist_ok=True)
    
    # 각 카메라 포즈에서 뷰 렌더링
    views_data = []
    for i, pose in enumerate(camera_poses):
        if pose is None or not torch.isfinite(pose).all():
            continue
            
        # 포즈 변환 (torch -> numpy)
        pose_np = to_numpy(pose)
        
        # 위치 및 회전 추출
        position = pose_np[:3, 3]
        
        # 회전 행렬 -> 쿼터니언으로 변환 (habitat-sim 형식)
        rot_matrix = pose_np[:3, :3]
        try:
            q = quaternion.from_rotation_matrix(rot_matrix)
            
            # 뷰 렌더링
            view = renderer.render_view(position, q)
            if view is not None:
                # RGBA 이미지와 깊이 맵 저장
                rgba_img = view["rgba_camera"]
                depth_map = view["depth_camera"]
                
                # 이미지 저장
                img_path = os.path.join(render_output_dir, f"view_{i:03d}.png")
                depth_path = os.path.join(render_output_dir, f"depth_{i:03d}.exr")
                
                cv2.imwrite(img_path, rgba_img[:, :, :3])
                
                # EXR 포맷으로 깊이 맵 저장
                try:
                    cv2.imwrite(depth_path, depth_map)
                except Exception as e:
                    print(f"깊이 맵 저장 오류: {e}")
                
                views_data.append({
                    "camera_idx": i,
                    "image_path": img_path,
                    "depth_path": depth_path,
                    "camera_pose": pose_np.tolist()
                })
        except Exception as e:
            print(f"카메라 {i} 렌더링 오류: {e}")
    
    # 렌더러 종료
    renderer.close()
    
    # 뷰 데이터 메타정보 저장
    metadata_path = os.path.join(render_output_dir, "views_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump({
            "views": views_data,
            "render_quality": render_quality,
            "resolution": resolution
        }, f, indent=2)
    
    print(f"MASt3R 스타일 렌더링 완료: {len(views_data)}개 뷰 생성")
    print(f"결과 저장 경로: {render_output_dir}")
    
    return {
        "render_output_dir": render_output_dir,
        "views_data": views_data,
        "metadata_path": metadata_path
    }

def main(optimization_dir, output_glb, filter_noise=True, use_mast3r_rendering=True, render_quality="high", as_pointcloud=False, force_pointcloud=False):
    """
    최적화 데이터에서 GLB 파일 생성 및 고급 렌더링 수행
    
    Args:
        optimization_dir: 최적화 데이터 디렉토리 경로
        output_glb: 출력 GLB 파일 경로
        filter_noise: 노이즈 필터링 활성화 여부
        use_mast3r_rendering: MASt3R 스타일 렌더링 사용 여부
        render_quality: 렌더링 품질 ('low', 'medium', 'high')
        as_pointcloud: 포인트 클라우드 형태로 표시할지 여부 (False면 메시로 변환)
    """
    # 최적화 데이터 로드
    camera_params, depth_data, image_info = load_optimization_data(optimization_dir)
    
    # 새 이미지 경로가 있으면 image_info에 추가 (강조 표시용)
    if 'new_image_path' in depth_data:
        new_image_path = depth_data['new_image_path']
        if new_image_path and os.path.exists(new_image_path):
            print(f"새 이미지 발견: {new_image_path}")
            if image_info is None:
                image_info = {}
            if 'new_images' not in image_info:
                image_info['new_images'] = []
            image_info['new_images'].append(new_image_path)
    
    # MASt3R 스타일 고급 렌더링 수행 (옵션)
    mast3r_rendering_data = None
    if use_mast3r_rendering and (HABITAT_AVAILABLE and HABITAT_RENDERER_AVAILABLE):
        mast3r_rendering_data = create_enhanced_3d_views(
            camera_params, 
            image_info, 
            optimization_dir,
            render_quality
        )
    elif use_mast3r_rendering:
        print("Habitat 시뮬레이터 모듈을 사용할 수 없어 고급 렌더링을 건너뜁니다.")
    
    # 포인트 클라우드 데이터 가져오기
    pts3d_data = depth_data.get('pts3d', None)
    
    # 포인트 클라우드 색상 데이터 가져오기
    pts3d_colors = depth_data.get('pts3d_colors', None)
    if pts3d_colors is not None:
        print(f"포인트 클라우드 색상 데이터 발견: {len(pts3d_colors)}개")
    else:
        print("포인트 클라우드 색상 데이터가 없습니다. 이미지에서 색상을 추출합니다.")
    
    # 카메라 포즈 가져오기
    camera_poses = camera_params.get('cam2w', None)
    
    # 데이터 사이즈 추정 (대용량 화일일 경우 자동 검색)
    estimated_points = 0
    if pts3d_data is not None:
        for pts in pts3d_data:
            if pts is not None:
                if isinstance(pts, torch.Tensor):
                    if len(pts.shape) >= 2:
                        estimated_points += np.prod(pts.shape[:2])
                else:
                    if len(pts.shape) >= 2:
                        estimated_points += np.prod(pts.shape[:2])
    
    auto_force_pointcloud = False
    if estimated_points > 10000000 and not force_pointcloud:
        print(f"경고: 대형 데이터셋 발견 ({estimated_points:,} 포인트 추정)")
        print("메모리 효율성을 위해 자동으로 포인트 클라우드 모드로 전환합니다.")
        auto_force_pointcloud = True
    
    # GLB 파일 생성 (기본적으로 메시로 변환하여 더 나은 시각화)
    glb_path = create_pointcloud_glb(
        output_glb,
        pts3d_data,
        camera_poses,
        cam_size=0.1,
        image_info=image_info,
        filter_noise=filter_noise,
        pts3d_colors=pts3d_colors,
        as_pointcloud=as_pointcloud,  # 기본적으로 False로 설정되어 메시로 변환됨
        force_pointcloud=force_pointcloud or auto_force_pointcloud  # 자동 감지 또는 수동 설정을 통해 포인트 클라우드 모드 사용
    )
    
    # MASt3R 렌더링을 사용한 경우 추가 정보 반환
    if mast3r_rendering_data is not None:
        # 메타데이터 업데이트 및 저장
        metadata_path = os.path.join(os.path.dirname(output_glb), "mast3r_metadata.json")
        metadata = {
            "glb_path": output_glb,
            "optimization_dir": optimization_dir,
            "mast3r_rendering": mast3r_rendering_data,
            "creation_time": str(np.datetime64('now'))
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"MASt3R 렌더링 메타데이터 저장: {metadata_path}")
    
    return glb_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MASt3R Habitat 기반 고급 3D 렌더링 및 GLB 파일 생성")
    parser.add_argument("--optimization_dir", type=str, required=True, 
                        help="최적화 데이터 디렉토리 경로")
    parser.add_argument("--output_glb", type=str, required=True, 
                        help="출력 GLB 파일 경로")
    parser.add_argument("--filter_noise", action="store_true", default=True,
                        help="Dust3R 스타일 노이즈 필터링 활성화")
    parser.add_argument("--use_mast3r_rendering", action="store_true", default=True,
                        help="MASt3R 스타일 고급 렌더링 사용")
    parser.add_argument("--render_quality", type=str, default="high",
                        choices=["low", "medium", "high"],
                        help="렌더링 품질 설정")
    parser.add_argument("--as_pointcloud", action="store_true", default=False,
                        help="포인트 클라우드 형태로 표시 (기본값: 메시로 변환)")
    parser.add_argument("--force_pointcloud", action="store_true", default=False,
                        help="대형 데이터셋에 강제로 포인트 클라우드 사용 (메모리 절약)")
    parser.add_argument("--new_image_path", type=str, default=None,
                        help="특별히 강조할 새 이미지 경로 (빨간색으로 표시)")
    
    args = parser.parse_args()
    
    # 새 이미지 경로가 제공된 경우 확인
    if args.new_image_path and os.path.exists(args.new_image_path):
        # optimization_dir에서 depth_data.pth 파일을 로드하여 새 이미지 경로 추가
        depth_data_path = os.path.join(args.optimization_dir, "depth_data.pth")
        if os.path.exists(depth_data_path):
            depth_data = torch.load(depth_data_path)
            depth_data['new_image_path'] = args.new_image_path
            torch.save(depth_data, depth_data_path)
            print(f"새 이미지 경로 추가됨: {args.new_image_path}")
    
    main(
        args.optimization_dir, 
        args.output_glb, 
        args.filter_noise, 
        args.use_mast3r_rendering,
        args.render_quality,
        args.as_pointcloud,
        args.force_pointcloud
    )