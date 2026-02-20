#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Полный пайплайн: генерация датасета → обучение GNN → совмещение моделей → булевы операции
Стабильно работает на macOS с pythonocc-core 7.9.0
"""

# ════════════════════════════════════════════════════════════════════════════════
# 0. БЕЗОПАСНАЯ ИНИЦИАЛИЗАЦИЯ (ОБЯЗАТЕЛЬНО В ПЕРВОЙ ЯЧЕЙКЕ!)
# ════════════════════════════════════════════════════════════════════════════════
import os
import gc
import sys
import numpy as np
import torch

# 🔴 КРИТИЧЕСКИ ВАЖНО для стабильности на macOS:
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# Импорты для работы с геометрией
try:
    from OCC.Core import STEPControl, TopExp, TopAbs
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Trsf
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
    from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Common, BRepAlgoAPI_Cut
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.Geom import Geom_Plane
    from OCC.Core.BRepPrimAPI import (
        BRepPrimAPI_MakeBox,
        BRepPrimAPI_MakeCylinder,
        BRepPrimAPI_MakeSphere,
        BRepPrimAPI_MakeCone
    )
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("Установите: conda install -c conda-forge pythonocc-core=7.9.0")
    sys.exit(1)

# Импорты для машинного обучения
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader
from scipy.spatial import cKDTree
import json

print("✅ Безопасная инициализация завершена")
print(f"   Python: {sys.version.split()[0]}")
print(f"   PyTorch: {torch.__version__}")
print(f"   NumPy: {np.__version__}")
gc.collect()


# ════════════════════════════════════════════════════════════════════════════════
# 1. БЕЗОПАСНАЯ ЗАГРУЗКА И ПАРСИНГ МОДЕЛЕЙ
# ════════════════════════════════════════════════════════════════════════════════

def safe_read_step_file(filename, max_attempts=3):
    """Безопасная загрузка STEP-файла с защитой от сегфолтов"""
    for attempt in range(max_attempts):
        try:
            reader = STEPControl.STEPControl_Reader()
            status = reader.ReadFile(str(filename))
            if status != 1:  # IFSelect_RetDone = 1
                raise RuntimeError(f"Ошибка чтения файла {filename}: статус {status}")
            
            reader.TransferRoots()
            shape = reader.OneShape()
            
            # КРИТИЧЕСКИ ВАЖНО: освобождаем ресурсы до возврата
            del reader
            gc.collect()
            
            return shape
            
        except Exception as e:
            print(f"⚠️ Попытка {attempt+1}/{max_attempts} не удалась для {filename}: {e}")
            gc.collect()
            if attempt == max_attempts - 1:
                print(f"❌ Не удалось загрузить {filename} после {max_attempts} попыток")
                return None


def extract_topology(shape):
    """Извлекает вершины и связи грань-вершина (гарантированно рабочая для pythonocc-core 7.9.0)"""
    vertices = []
    vertex_map = {}
    face_vertex_indices = []

    face_explorer = TopExp.TopExp_Explorer(shape, TopAbs.TopAbs_FACE)
    while face_explorer.More():
        face = face_explorer.Current()
        local_vertices = []

        edge_explorer = TopExp.TopExp_Explorer(face, TopAbs.TopAbs_EDGE)
        while edge_explorer.More():
            edge = edge_explorer.Current()
            vertex_explorer = TopExp.TopExp_Explorer(edge, TopAbs.TopAbs_VERTEX)
            while vertex_explorer.More():
                vertex = vertex_explorer.Current()
                p = BRep_Tool.Pnt(vertex)
                key = (round(p.X(), 6), round(p.Y(), 6), round(p.Z(), 6))
                if key not in vertex_map:
                    vertex_map[key] = len(vertices)
                    vertices.append(np.array([p.X(), p.Y(), p.Z()]))
                local_vertices.append(vertex_map[key])
                vertex_explorer.Next()
            edge_explorer.Next()

        if local_vertices:
            local_vertices = list(dict.fromkeys(local_vertices))
            face_vertex_indices.append(local_vertices)
        face_explorer.Next()

    gc.collect()
    return np.array(vertices), face_vertex_indices


def compute_center_of_mass(shape):
    """Вычисляет центр масс тела и возвращает как список [x, y, z]"""
    props = GProp_GProps()
    try:
        brepgprop_VolumeProperties(shape, props)
        cog = props.CentreOfMass()
        return [float(cog.X()), float(cog.Y()), float(cog.Z())]
    except:
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib_Add
        bbox = Bnd_Box()
        brepbndlib_Add(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        return [float((xmin+xmax)/2), float((ymin+ymax)/2), float((zmin+zmax)/2)]


# ════════════════════════════════════════════════════════════════════════════════
# 2. ПОСТРОЕНИЕ ГРАФА ИЗ ТОПОЛОГИИ
# ════════════════════════════════════════════════════════════════════════════════

def normalize_coordinates(vertices):
    if len(vertices) == 0:
        return vertices
    center = vertices.mean(axis=0)
    scale = np.max(np.abs(vertices - center)) + 1e-8
    return (vertices - center) / scale


def build_graph(vertices, face_vertex_indices):
    n_vertices = len(vertices)
    n_faces = len(face_vertex_indices)

    vertices_norm = normalize_coordinates(vertices)
    node_coords = np.zeros((n_vertices + n_faces, 3))
    node_types = np.zeros(n_vertices + n_faces, dtype=int)

    # Вершины
    node_coords[:n_vertices] = vertices_norm
    node_types[:n_vertices] = 0

    # Грани
    for i, vtx_ids in enumerate(face_vertex_indices):
        if vtx_ids:
            center = vertices_norm[vtx_ids].mean(axis=0)
            node_coords[n_vertices + i] = center
            node_types[n_vertices + i] = 1

    # Рёбра
    edge_index = []
    for face_id, vtx_ids in enumerate(face_vertex_indices):
        for vtx_id in vtx_ids:
            edge_index.append([n_vertices + face_id, vtx_id])
            edge_index.append([vtx_id, n_vertices + face_id])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = torch.tensor(node_coords, dtype=torch.float)
    node_type = torch.tensor(node_types, dtype=torch.long)

    gc.collect()
    return Data(x=x, edge_index=edge_index, node_type=node_type)


# ════════════════════════════════════════════════════════════════════════════════
# 3. ГЕНЕРАЦИЯ СИНТЕТИЧЕСКОГО ДАТАСЕТА (4 типа деталей)
# ════════════════════════════════════════════════════════════════════════════════

def create_synthetic_part(part_type="bracket", seed=None, size_variation=1.0):
    """
    Генерация синтетической детали с полной разметкой (все значения сериализуемы в JSON)
    
    Поддерживаемые типы: "bracket", "flange", "block_with_holes", "t_bracket"
    """
    if seed is not None:
        np.random.seed(seed)
    
    base_size = 100.0 * size_variation
    thickness = 10.0 * size_variation
    
    if part_type == "bracket":
        # Кронштейн: основание + вертикальная пластина + 2 отверстия
        base = BRepPrimAPI_MakeBox(base_size, base_size*0.5, thickness).Shape()
        wall = BRepPrimAPI_MakeBox(thickness, base_size*0.5, base_size*0.4).Shape()
        
        trsf = gp_Trsf()
        trsf.SetTranslation(gp_Vec(base_size - thickness, 0, thickness))
        transform = BRepBuilderAPI_Transform(wall, trsf)
        wall_pos = transform.Shape()
        
        part = BRepAlgoAPI_Fuse(base, wall_pos).Shape()
        
        holes = []
        for i in range(2):
            hole = BRepPrimAPI_MakeCylinder(5.0 * size_variation, thickness*1.5).Shape()
            trsf = gp_Trsf()
            trsf.SetTranslation(gp_Vec(base_size*0.3 + i*base_size*0.4, base_size*0.25, -thickness*0.25))
            transform = BRepBuilderAPI_Transform(hole, trsf)
            hole_pos = transform.Shape()
            part = BRepAlgoAPI_Cut(part, hole_pos).Shape()
            holes.append({
                'center': [
                    float(base_size*0.3 + i*base_size*0.4),
                    float(base_size*0.25),
                    float(thickness/2)
                ],
                'diameter': float(10.0 * size_variation),
                'type': 'through_hole'
            })
        
        cog = compute_center_of_mass(part)
        annotations = {
            "center_of_mass": cog,
            "reference_planes": [
                {
                    "center": [float(base_size/2), float(base_size*0.25), float(thickness/2)],
                    "normal": [0.0, 0.0, 1.0],
                    "area": float(base_size*base_size*0.5),
                    "role": 3
                },
                {
                    "center": [float(base_size - thickness/2), float(base_size*0.25), float(base_size*0.2 + thickness)],
                    "normal": [1.0, 0.0, 0.0],
                    "area": float(base_size*0.5*base_size*0.4),
                    "role": 3
                },
                {
                    "center": [float(base_size/2), float(base_size*0.25), 0.0],
                    "normal": [0.0, 0.0, -1.0],
                    "area": float(base_size*base_size*0.5),
                    "role": 3
                }
            ],
            "fastening_elements": holes,
            "functional_surfaces": [
                {
                    "center": [float(base_size - thickness/2), float(base_size*0.25), float(base_size*0.2 + thickness + base_size*0.2)],
                    "normal": [0.0, 0.0, 1.0],
                    "area": float(thickness*base_size*0.5),
                    "role": 1
                }
            ],
            "part_type": "bracket"
        }
        return part, annotations
    
    elif part_type == "flange":
        # Фланец: диск с 4 отверстиями по окружности
        radius = base_size * 0.5
        flange = BRepPrimAPI_MakeCylinder(radius, thickness).Shape()
        
        # Центральное отверстие
        center_hole = BRepPrimAPI_MakeCylinder(radius*0.2, thickness*1.5).Shape()
        part = BRepAlgoAPI_Cut(flange, center_hole).Shape()
        
        # 4 отверстия по окружности
        holes = []
        for i in range(4):
            angle = np.radians(i * 90)
            x = radius * 0.6 * np.cos(angle)
            y = radius * 0.6 * np.sin(angle)
            hole = BRepPrimAPI_MakeCylinder(6.0 * size_variation, thickness*1.5).Shape()
            trsf = gp_Trsf()
            trsf.SetTranslation(gp_Vec(x, y, -thickness*0.25))
            transform = BRepBuilderAPI_Transform(hole, trsf)
            hole_pos = transform.Shape()
            part = BRepAlgoAPI_Cut(part, hole_pos).Shape()
            holes.append({
                'center': [float(x), float(y), float(thickness/2)],
                'diameter': float(12.0 * size_variation),
                'type': 'mounting_hole'
            })
        
        cog = compute_center_of_mass(part)
        annotations = {
            "center_of_mass": cog,
            "reference_planes": [
                {
                    "center": [0.0, 0.0, float(thickness/2)],
                    "normal": [0.0, 0.0, 1.0],
                    "area": float(np.pi * radius**2),
                    "role": 3
                },
                {
                    "center": [0.0, 0.0, 0.0],
                    "normal": [0.0, 0.0, -1.0],
                    "area": float(np.pi * radius**2),
                    "role": 3
                }
            ],
            "fastening_elements": holes,
            "functional_surfaces": [
                {
                    "center": [0.0, 0.0, float(thickness/2)],
                    "normal": [0.0, 0.0, 1.0],
                    "area": float(np.pi * (radius**2 - (radius*0.2)**2)),
                    "role": 1
                }
            ],
            "part_type": "flange"
        }
        return part, annotations
    
    elif part_type == "block_with_holes":
        # Брусок с 6 отверстиями (2 ряда по 3)
        block = BRepPrimAPI_MakeBox(base_size, base_size, base_size*0.5).Shape()
        
        holes = []
        for i in range(2):  # 2 ряда
            for j in range(3):  # 3 отверстия в ряду
                x = base_size * 0.25 + j * base_size * 0.25
                y = base_size * 0.3 + i * base_size * 0.4
                hole = BRepPrimAPI_MakeCylinder(4.0 * size_variation, base_size*0.6).Shape()
                trsf = gp_Trsf()
                trsf.SetTranslation(gp_Vec(x, y, -base_size*0.05))
                transform = BRepBuilderAPI_Transform(hole, trsf)
                hole_pos = transform.Shape()
                block = BRepAlgoAPI_Cut(block, hole_pos).Shape()
                holes.append({
                    'center': [float(x), float(y), float(base_size*0.25)],
                    'diameter': float(8.0 * size_variation),
                    'type': 'grid_hole'
                })
        
        cog = compute_center_of_mass(block)
        annotations = {
            "center_of_mass": cog,
            "reference_planes": [
                {
                    "center": [float(base_size/2), float(base_size/2), float(base_size*0.25)],
                    "normal": [0.0, 0.0, 1.0],
                    "area": float(base_size**2),
                    "role": 3
                },
                {
                    "center": [float(base_size/2), float(base_size/2), 0.0],
                    "normal": [0.0, 0.0, -1.0],
                    "area": float(base_size**2),
                    "role": 3
                },
                {
                    "center": [0.0, float(base_size/2), float(base_size*0.25)],
                    "normal": [-1.0, 0.0, 0.0],
                    "area": float(base_size*base_size*0.5),
                    "role": 3
                }
            ],
            "fastening_elements": holes,
            "functional_surfaces": [],
            "part_type": "block_with_holes"
        }
        return block, annotations
    
    elif part_type == "t_bracket":
        # T-образный кронштейн
        base = BRepPrimAPI_MakeBox(base_size, thickness, thickness).Shape()
        vertical = BRepPrimAPI_MakeBox(thickness, base_size*0.6, base_size*0.3).Shape()
        
        trsf = gp_Trsf()
        trsf.SetTranslation(gp_Vec((base_size-thickness)/2, 0, thickness))
        transform = BRepBuilderAPI_Transform(vertical, trsf)
        vertical_pos = transform.Shape()
        
        part = BRepAlgoAPI_Fuse(base, vertical_pos).Shape()
        
        # Отверстия в основании
        holes = []
        for i in range(2):
            hole = BRepPrimAPI_MakeCylinder(5.0 * size_variation, thickness*1.5).Shape()
            trsf = gp_Trsf()
            trsf.SetTranslation(gp_Vec(base_size*0.25 + i*base_size*0.5, thickness/2, -thickness*0.25))
            transform = BRepBuilderAPI_Transform(hole, trsf)
            hole_pos = transform.Shape()
            part = BRepAlgoAPI_Cut(part, hole_pos).Shape()
            holes.append({
                'center': [
                    float(base_size*0.25 + i*base_size*0.5),
                    float(thickness/2),
                    float(thickness/2)
                ],
                'diameter': float(10.0 * size_variation),
                'type': 'mounting_hole'
            })
        
        cog = compute_center_of_mass(part)
        annotations = {
            "center_of_mass": cog,
            "reference_planes": [
                {
                    "center": [float(base_size/2), float(thickness/2), float(thickness/2)],
                    "normal": [0.0, 0.0, 1.0],
                    "area": float(base_size*thickness),
                    "role": 3
                },
                {
                    "center": [float(base_size/2), float(thickness/2), 0.0],
                    "normal": [0.0, 0.0, -1.0],
                    "area": float(base_size*thickness),
                    "role": 3
                },
                {
                    "center": [float((base_size-thickness)/2 + thickness/2), float(base_size*0.3), float(base_size*0.15 + thickness)],
                    "normal": [1.0, 0.0, 0.0],
                    "area": float(thickness*base_size*0.6),
                    "role": 3
                }
            ],
            "fastening_elements": holes,
            "functional_surfaces": [],
            "part_type": "t_bracket"
        }
        return part, annotations
    
    else:
        raise ValueError(f"Неизвестный тип детали: {part_type}")


def generate_enhanced_dataset(output_dir="enhanced_dataset", n_samples=50):
    """
    Генерация улучшенного синтетического датасета с сериализуемыми данными
    
    Структура выхода:
        enhanced_dataset/
        ├── raw/              # STEP-файлы оригинальных моделей
        ├── annotations/      # JSON с полной разметкой
        └── pairs/            # Пары моделей с известными преобразованиями
    """
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pairs"), exist_ok=True)
    
    part_types = ["bracket", "flange", "block_with_holes", "t_bracket"]
    pair_id = 0
    
    print(f"🔄 Генерация {n_samples} синтетических деталей...")
    
    for i in range(n_samples):
        part_type = np.random.choice(part_types)
        size_var = np.random.uniform(0.8, 1.2)
        
        shape_orig, ann_orig = create_synthetic_part(
            part_type=part_type, 
            seed=i,
            size_variation=size_var
        )
        
        orig_filename = f"{part_type}_{i:06d}.step"
        orig_path = os.path.join(output_dir, "raw", orig_filename)
        writer = STEPControl_Writer()
        writer.Transfer(shape_orig, STEPControl_AsIs)
        writer.Write(orig_path)
        
        ann_path = os.path.join(output_dir, "annotations", f"{part_type}_{i:06d}.json")
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(ann_orig, f, indent=2, ensure_ascii=False)
        
        gc.collect()  # Очистка памяти после каждой модели
    
    print(f"✅ Сгенерировано {n_samples} моделей")
    print(f"📁 Датасет сохранён в: {output_dir}")
    return output_dir


# ════════════════════════════════════════════════════════════════════════════════
# 4. ДАТАСЕТ ДЛЯ PYTORCH GEOMETRIC
# ════════════════════════════════════════════════════════════════════════════════

class FixedSyntheticDataset(Dataset):
    """
    Датасет для обучения на синтетических данных с корректным сопоставлением аннотаций
    """
    def __init__(self, root="enhanced_dataset", transform=None, pre_transform=None):
        self.role_mapping = {
            "decorative": 0,
            "functional": 1,
            "fastening": 2,
            "reference_plane": 3
        }
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_dir(self):
        return os.path.join(self.root, "raw")
    
    @property
    def ann_dir(self):
        return os.path.join(self.root, "annotations")
    
    @property
    def raw_file_names(self):
        files = [f for f in os.listdir(self.raw_dir) if f.endswith(".step") and "_trans_" not in f]
        return sorted(files)
    
    @property
    def processed_file_names(self):
        return [f"data_{idx:06d}.pt" for idx in range(len(self.raw_file_names))]
    
    def process(self):
        print(f"🔄 Подготовка {len(self.raw_file_names)} моделей...")
        
        for idx, step_file in enumerate(self.raw_file_names):
            # 1. Загрузка модели
            reader = STEPControl.STEPControl_Reader()
            reader.ReadFile(os.path.join(self.raw_dir, step_file))
            reader.TransferRoots()
            shape = reader.OneShape()
            
            # 2. Извлечение топологии → ИСХОДНЫЕ координаты в мм
            vertices, face_vertex_indices = extract_topology(shape)
            
            # 3. Загрузка аннотаций
            ann_file = os.path.splitext(step_file)[0] + ".json"
            ann_path = os.path.join(self.ann_dir, ann_file)
            annotations = {}
            if os.path.exists(ann_path):
                try:
                    with open(ann_path, "r", encoding="utf-8") as f:
                        annotations = json.load(f)
                except Exception as e:
                    print(f"⚠️ Ошибка загрузки {ann_file}: {e}")
                    continue
            
            # 4. СОПОСТАВЛЕНИЕ В ИСХОДНЫХ КООРДИНАТАХ (мм)
            n_vertices = len(vertices)
            n_faces = len(face_vertex_indices)
            node_roles = np.zeros(n_vertices + n_faces, dtype=np.int64)
            node_roles[:n_vertices] = self.role_mapping["decorative"]  # вершины → декоративные
            
            # Вычисляем центроиды граней в ИСХОДНЫХ координатах (мм)
            face_centers_mm = []
            for vtx_ids in face_vertex_indices:
                if vtx_ids:
                    center = vertices[vtx_ids].mean(axis=0)
                    face_centers_mm.append(center)
            face_centers_mm = np.array(face_centers_mm)
            
            # Сопоставление аннотаций
            assigned = np.zeros(n_faces, dtype=bool)
            
            # Опорные плоскости (роль 3) — приоритет
            for ref_plane in annotations.get("reference_planes", []):
                ref_center = np.array(ref_plane["center"])
                if len(face_centers_mm) > 0:
                    distances = np.linalg.norm(face_centers_mm - ref_center, axis=1)
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] < 50.0:  # увеличенный порог 50 мм
                        node_roles[n_vertices + closest_idx] = self.role_mapping["reference_plane"]
                        assigned[closest_idx] = True
            
            # Крепёжные элементы (роль 2)
            for fastening in annotations.get("fastening_elements", []):
                fast_center = np.array(fastening["center"])
                if len(face_centers_mm) > 0:
                    distances = np.linalg.norm(face_centers_mm - fast_center, axis=1)
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] < 20.0 and not assigned[closest_idx]:
                        node_roles[n_vertices + closest_idx] = self.role_mapping["fastening"]
                        assigned[closest_idx] = True
            
            # Функциональные поверхности (роль 1)
            for func_surf in annotations.get("functional_surfaces", []):
                func_center = np.array(func_surf["center"])
                if len(face_centers_mm) > 0:
                    distances = np.linalg.norm(face_centers_mm - func_center, axis=1)
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] < 20.0 and not assigned[closest_idx]:
                        node_roles[n_vertices + closest_idx] = self.role_mapping["functional"]
                        assigned[closest_idx] = True
            
            # Остальные грани → функциональные (роль 1)
            for i in range(n_faces):
                if not assigned[i]:
                    node_roles[n_vertices + i] = self.role_mapping["functional"]
            
            # 5. Строим граф С НОРМАЛИЗАЦИЕЙ
            data = build_graph(vertices, face_vertex_indices)
            data.y = torch.tensor(node_roles, dtype=torch.long)
            
            # 6. Сохранение
            torch.save(data, os.path.join(self.processed_dir, f"data_{idx:06d}.pt"))
        
        print(f"✅ Подготовлено {len(self.raw_file_names)} моделей")
        print(f"   Графы сохранены в: {self.processed_dir}")
    
    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        return torch.load(
            os.path.join(self.processed_dir, self.processed_file_names[idx]),
            weights_only=False
        )


# ════════════════════════════════════════════════════════════════════════════════
# 5. АРХИТЕКТУРА GNN
# ════════════════════════════════════════════════════════════════════════════════

class GNNModel(torch.nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64, num_roles=4):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_dim, heads=4, concat=True, dropout=0.2)
        self.conv2 = GATv2Conv(hidden_dim * 4, hidden_dim, heads=2, concat=False, dropout=0.2)
        self.role_classifier = torch.nn.Linear(hidden_dim, num_roles)
        self.graph_proj = torch.nn.Linear(hidden_dim, 64)

    def forward(self, x, edge_index, batch):
        x = torch.nn.functional.relu(self.conv1(x, edge_index))
        x = torch.nn.functional.relu(self.conv2(x, edge_index))
        roles = self.role_classifier(x)
        graph_emb = global_mean_pool(x, batch)
        graph_emb = self.graph_proj(graph_emb)
        return roles, graph_emb


# ════════════════════════════════════════════════════════════════════════════════
# 6. СОВМЕЩЕНИЕ МОДЕЛЕЙ И БУЛЕВЫ ОПЕРАЦИИ
# ════════════════════════════════════════════════════════════════════════════════

def align_with_umeyama(src_pts, tgt_pts):
    """Алгоритм Умэямы для вычисления оптимального преобразования"""
    assert src_pts.shape == tgt_pts.shape and src_pts.shape[0] >= 3
    src_mean = src_pts.mean(axis=0)
    tgt_mean = tgt_pts.mean(axis=0)
    src_c = src_pts - src_mean
    tgt_c = tgt_pts - tgt_mean
    H = src_c.T @ tgt_c
    U, S, Vt = np.linalg.svd(H)
    R_opt = Vt.T @ U.T
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = Vt.T @ U.T
    t_opt = tgt_mean - R_opt @ src_mean
    return R_opt, t_opt


def find_elements_by_role(role_probs, node_types, coords, role_idx=3, top_k=3):
    """
    Находит элементы заданной роли (0=декор, 1=функциональная, 2=отверстия, 3=опорные плоскости)
    """
    face_mask = (node_types == 1)
    if not np.any(face_mask):
        return np.array([])
    
    scores = role_probs[face_mask, role_idx]
    top_indices = np.argsort(-scores)[:top_k]
    face_indices = np.where(face_mask)[0][top_indices]
    return coords[face_indices]


def extract_large_planes(shape, min_area=10.0, max_planes=5):
    """
    Извлечение крупных плоских граней без нейросети (полностью совместимо с pythonocc-core 7.9.0)
    """
    planes = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    
    while explorer.More():
        face = explorer.Current()
        surface = BRep_Tool.Surface(face)
        
        if not surface.IsKind("Geom_Plane"):
            explorer.Next()
            continue
        
        props = GProp_GProps()
        brepgprop_SurfaceProperties(face, props)
        area = props.Mass()
        
        if area < min_area:
            explorer.Next()
            continue
        
        cog = props.CentreOfMass()
        
        try:
            plane = Geom_Plane.DownCast(surface)
            normal = plane.Axis().Direction()
        except:
            explorer.Next()
            continue
        
        planes.append({
            'area': float(area),
            'center': np.array([cog.X(), cog.Y(), cog.Z()]),
            'normal': np.array([normal.X(), normal.Y(), normal.Z()])
        })
        explorer.Next()
    
    planes.sort(key=lambda x: x['area'], reverse=True)
    gc.collect()
    return planes[:max_planes]


def evaluate_alignment(shape1, shape2_aligned, n_max_vertices=2000):
    """Оценка качества совмещения через расстояния между вершинами"""
    # Минимальное расстояние Хаусдорфа
    dist_checker = BRepExtrema_DistShapeShape()
    dist_checker.LoadS1(shape1)
    dist_checker.LoadS2(shape2_aligned)
    dist_checker.Perform()
    
    if not dist_checker.IsDone():
        return {'success': False, 'error': 'Расчёт расстояния не удался'}
    
    min_dist = dist_checker.Value()
    
    # Извлечение вершин
    vertices1, _ = extract_topology(shape1)
    vertices2, _ = extract_topology(shape2_aligned)
    
    # Ограничение числа вершин для скорости
    if len(vertices1) > n_max_vertices:
        indices = np.random.choice(len(vertices1), n_max_vertices, replace=False)
        vertices1 = vertices1[indices]
    if len(vertices2) > n_max_vertices:
        indices = np.random.choice(len(vertices2), n_max_vertices, replace=False)
        vertices2 = vertices2[indices]
    
    if len(vertices1) == 0 or len(vertices2) == 0:
        return {'success': False, 'error': 'Не удалось извлечь вершины'}
    
    # Расчёт расстояний
    tree2 = cKDTree(vertices2)
    dists12, _ = tree2.query(vertices1, k=1)
    
    tree1 = cKDTree(vertices1)
    dists21, _ = tree1.query(vertices2, k=1)
    
    hausdorff_sym = max(dists12.max(), dists21.max())
    mean_dist = (dists12.mean() + dists21.mean()) / 2.0
    
    gc.collect()
    return {
        'success': True,
        'hausdorff_min': min_dist,
        'hausdorff_symmetric': hausdorff_sym,
        'mean_distance': mean_dist,
        'max_distance': max(dists12.max(), dists21.max()),
        'rms_distance': np.sqrt((dists12**2).mean()),
        'inlier_ratio_0.1mm': (dists12 < 0.1).mean(),
        'inlier_ratio_1.0mm': (dists12 < 1.0).mean(),
        'sample_count': len(vertices1)
    }


def boolean_operations(shape1, shape2_aligned, tolerance=1e-3):
    """
    Выполняет булевы операции напрямую с TopoDS_Shape (без приведения типов)
    """
    results = {}
    errors = []
    
    # Объединение
    try:
        fuse = BRepAlgoAPI_Fuse(shape1, shape2_aligned)
        fuse.SetFuzzyValue(tolerance)
        if fuse.IsDone():
            results['fuse'] = fuse.Shape()
        else:
            errors.append("fuse")
    except Exception as e:
        errors.append(f"fuse: {str(e)}")
    
    # Пересечение
    try:
        common = BRepAlgoAPI_Common(shape1, shape2_aligned)
        common.SetFuzzyValue(tolerance)
        if common.IsDone():
            results['common'] = common.Shape()
        else:
            errors.append("common")
    except Exception as e:
        errors.append(f"common: {str(e)}")
    
    # Разность (A \ B)
    try:
        cut1 = BRepAlgoAPI_Cut(shape1, shape2_aligned)
        cut1.SetFuzzyValue(tolerance)
        if cut1.IsDone():
            results['diff1'] = cut1.Shape()
        else:
            errors.append("diff1")
    except Exception as e:
        errors.append(f"diff1: {str(e)}")
    
    # Разность (B \ A)
    try:
        cut2 = BRepAlgoAPI_Cut(shape2_aligned, shape1)
        cut2.SetFuzzyValue(tolerance)
        if cut2.IsDone():
            results['diff2'] = cut2.Shape()
        else:
            errors.append("diff2")
    except Exception as e:
        errors.append(f"diff2: {str(e)}")
    
    # Симметрическая разность
    if 'diff1' in results and 'diff2' in results:
        try:
            symdiff = BRepAlgoAPI_Fuse(results['diff1'], results['diff2'])
            symdiff.SetFuzzyValue(tolerance)
            if symdiff.IsDone():
                results['symdiff'] = symdiff.Shape()
        except:
            pass
    
    if errors:
        results['errors'] = errors
    
    gc.collect()
    return results


def compute_volumes(shape):
    """Вычисляет объём и площадь поверхности (работает с любым TopoDS_Shape)"""
    volume = 0.0
    area = 0.0
    
    try:
        props = GProp_GProps()
        brepgprop_VolumeProperties(shape, props)
        volume = props.Mass()
    except:
        volume = 0.0
    
    try:
        props = GProp_GProps()
        brepgprop_SurfaceProperties(shape, props)
        area = props.Mass()
    except:
        area = 0.0
    
    return volume, area


def analyze_differences(shape1, shape2_aligned, tolerance=1e-3):
    """Полный анализ различий через булевы операции"""
    results = boolean_operations(shape1, shape2_aligned, tolerance)
    
    vol1, area1 = compute_volumes(shape1)
    vol2, area2 = compute_volumes(shape2_aligned)
    
    vol_diff1 = compute_volumes(results['diff1'])[0] if 'diff1' in results else 0.0
    vol_diff2 = compute_volumes(results['diff2'])[0] if 'diff2' in results else 0.0
    vol_common = compute_volumes(results['common'])[0] if 'common' in results else 0.0
    
    total_vol = max(vol1, vol2)
    diff_percent = ((vol_diff1 + vol_diff2) / total_vol * 100) if total_vol > 0 else 0.0
    
    gc.collect()
    return {
        'volume_model1': vol1,
        'volume_model2': vol2,
        'volume_unique_to_1': vol_diff1,
        'volume_unique_to_2': vol_diff2,
        'volume_common': vol_common,
        'difference_percent': diff_percent,
        'area_model1': area1,
        'area_model2': area2,
        'results': results
    }


def save_boolean_results(results, output_dir="outputs/boolean"):
    """Сохраняет результаты булевых операций в STEP-файлы"""
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    
    mapping = {
        'fuse': 'union.step',
        'common': 'intersection.step',
        'diff1': 'unique_to_model1.step',
        'diff2': 'unique_to_model2.step',
        'symdiff': 'all_differences.step'
    }
    
    for key, filename in mapping.items():
        if key in results:
            try:
                writer = STEPControl_Writer()
                writer.Transfer(results[key], STEPControl_AsIs)
                status = writer.Write(os.path.join(output_dir, filename))
                if status == 1:
                    saved.append(filename)
            except Exception as e:
                print(f"⚠️ Не удалось сохранить {filename}: {e}")
    
    gc.collect()
    return saved


# ════════════════════════════════════════════════════════════════════════════════
# 7. ГЛАВНЫЙ ПАЙПЛАЙН
# ════════════════════════════════════════════════════════════════════════════════

def main_pipeline(step_file_1, step_file_2, use_gnn=False, model_path="gnn_best.pth"):
    """
    Полный пайплайн сравнения и совмещения двух моделей с использованием:
    - Центра масс (всегда)
    - Отверстий (роль 2)
    - Опорных плоскостей (роль 3)
    """
    # 1. Загрузка моделей с защитой от сегфолтов
    print("📦 Загрузка моделей...")
    shape1 = safe_read_step_file(step_file_1)
    shape2 = safe_read_step_file(step_file_2)
    
    if shape1 is None or shape2 is None:
        print("❌ Не удалось загрузить одну или обе модели")
        return None
    
    # 2. Совмещение
    print("🔧 Совмещение моделей...")
    R_mat, t_vec = None, None
    
    if use_gnn:
        # Проверка существования файла модели
        if not os.path.exists(model_path):
            print(f"⚠️ Модель {model_path} не найдена — переключаюсь на эвристику")
            use_gnn = False
        else:
            try:
                # Загрузка модели
                model = GNNModel(in_channels=3, hidden_dim=64, num_roles=4)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                
                # Извлечение топологии и построение графов
                v1, fv1 = extract_topology(shape1)
                v2, fv2 = extract_topology(shape2)
                g1 = build_graph(v1, fv1)
                g2 = build_graph(v2, fv2)
                
                # Предсказание ролей
                with torch.no_grad():
                    batch1 = torch.zeros(g1.x.size(0), dtype=torch.long)
                    batch2 = torch.zeros(g2.x.size(0), dtype=torch.long)
                    roles1, _ = model(g1.x, g1.edge_index, batch1)
                    roles2, _ = model(g2.x, g2.edge_index, batch2)
                
                # Вычисление центра масс (всегда доступен)
                cog1 = np.array(compute_center_of_mass(shape1))
                cog2 = np.array(compute_center_of_mass(shape2))
                
                # Поиск отверстий (роль 2) и опорных плоскостей (роль 3)
                holes1 = find_elements_by_role(roles1.softmax(dim=1).numpy(), g1.node_type.numpy(), g1.x[:, :3].numpy(), role_idx=2, top_k=2)
                holes2 = find_elements_by_role(roles2.softmax(dim=1).numpy(), g2.node_type.numpy(), g2.x[:, :3].numpy(), role_idx=2, top_k=2)
                ref_planes1 = find_elements_by_role(roles1.softmax(dim=1).numpy(), g1.node_type.numpy(), g1.x[:, :3].numpy(), role_idx=3, top_k=2)
                ref_planes2 = find_elements_by_role(roles2.softmax(dim=1).numpy(), g2.node_type.numpy(), g2.x[:, :3].numpy(), role_idx=3, top_k=2)
                
                # Сбор точек для совмещения (в порядке приоритета)
                points1 = [cog1]  # Центр масс — первая точка!
                points2 = [cog2]
                points1.extend(holes1[:min(2, len(holes1))])      # До 2 отверстий
                points2.extend(holes2[:min(2, len(holes2))])
                points1.extend(ref_planes1[:min(2, len(ref_planes1))])  # До 2 опорных плоскостей
                points2.extend(ref_planes2[:min(2, len(ref_planes2))])
                
                # Совмещение по первым 3 точкам
                if len(points1) >= 3 and len(points2) >= 3:
                    src_pts = np.array(points1[:3])
                    tgt_pts = np.array(points2[:3])
                    R_mat, t_vec = align_with_umeyama(src_pts, tgt_pts)
                    print(f"✅ Совмещение через GNN выполнено по {len(points1[:3])} точкам:")
                    print(f"   • Центр масс")
                    print(f"   • {len(holes1[:2])} отверстий")
                    print(f"   • {len(ref_planes1[:2])} опорных плоскостей")
                else:
                    print(f"⚠️ Недостаточно точек для совмещения (найдено: {len(points1)}/{len(points2)}, нужно ≥3) — переключаюсь на эвристику")
                    use_gnn = False
            except Exception as e:
                print(f"❌ Ошибка при использовании GNN ({type(e).__name__}): {e}")
                print("⚠️ Переключаюсь на эвристику без GNN")
                use_gnn = False
    
    # Резервный метод: эвристика по крупным плоским граням
    if not use_gnn:
        planes1 = extract_large_planes(shape1, min_area=10.0, max_planes=5)
        planes2 = extract_large_planes(shape2, min_area=10.0, max_planes=5)
        
        if len(planes1) >= 3 and len(planes2) >= 3:
            src_pts = np.array([p['center'] for p in planes1[:3]])
            tgt_pts = np.array([p['center'] for p in planes2[:3]])
            R_mat, t_vec = align_with_umeyama(src_pts, tgt_pts)
            print("✅ Совмещение через эвристику выполнено (по 3 крупнейшим плоскостям)")
        else:
            print("⚠️ Недостаточно плоских граней — применяю тождественное преобразование")
            R_mat, t_vec = np.eye(3), np.zeros(3)
    
    # 3. Применение преобразования
    trsf = gp_Trsf()
    trsf.SetValues(
        float(R_mat[0,0]), float(R_mat[0,1]), float(R_mat[0,2]), float(t_vec[0]),
        float(R_mat[1,0]), float(R_mat[1,1]), float(R_mat[1,2]), float(t_vec[1]),
        float(R_mat[2,0]), float(R_mat[2,1]), float(R_mat[2,2]), float(t_vec[2])
    )
    
    transform = BRepBuilderAPI_Transform(shape2, trsf)
    shape2_aligned = transform.Shape()
    gc.collect()
    
    # 4. Сохранение совмещённой модели
    os.makedirs("outputs/results", exist_ok=True)
    writer = STEPControl_Writer()
    writer.Transfer(shape2_aligned, STEPControl_AsIs)
    writer.Write("outputs/results/aligned_model.step")
    print("💾 Совмещённая модель сохранена: outputs/results/aligned_model.step")
    
    # 5. Оценка качества совмещения
    print("\n📊 Оценка качества совмещения...")
    metrics = evaluate_alignment(shape1, shape2_aligned, n_max_vertices=2000)
    if metrics['success']:
        print(f"   Средняя ошибка: {metrics['mean_distance']:.4f} мм")
        print(f"   Точек < 0.1 мм: {metrics['inlier_ratio_0.1mm']*100:.1f}%")
    
    # 6. Анализ различий через булевы операции
    print("\n🔍 Анализ различий через булевы операции...")
    analysis = analyze_differences(shape1, shape2_aligned, tolerance=1e-3)
    
    print(f"   Объём модели 1:     {analysis['volume_model1']:.2f} мм³")
    print(f"   Объём модели 2:     {analysis['volume_model2']:.2f} мм³")
    print(f"   Различия:           {analysis['difference_percent']:.2f}%")
    
    # Интерпретация
    if analysis['difference_percent'] < 0.1:
        print("✅ Модели практически идентичны")
    elif analysis['difference_percent'] < 1.0:
        print("⚠️  Модели очень похожи (возможны допуски)")
    elif analysis['difference_percent'] < 5.0:
        print("🔶 Модели умеренно различаются")
    else:
        print("❌ Модели значительно различаются")
    
    # 7. Сохранение результатов булевых операций
    saved_files = save_boolean_results(analysis['results'], "outputs/boolean")
    if saved_files:
        print(f"\n💾 Сохранены файлы для визуализации:")
        for fname in saved_files:
            print(f"   • outputs/boolean/{fname}")
        print("\n💡 Откройте 'all_differences.step' в FreeCAD для визуализации различий")
    
    gc.collect()
    return {
        'alignment_metrics': metrics,
        'difference_analysis': analysis,
        'saved_files': saved_files
    }


# ════════════════════════════════════════════════════════════════════════════════
# 8. ЗАПУСК ОБУЧЕНИЯ И ТЕСТИРОВАНИЯ
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*70)
    print("🚀 ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА: ГЕНЕРАЦИЯ → ОБУЧЕНИЕ → СОВМЕЩЕНИЕ → БУЛЕВЫ ОПЕРАЦИИ")
    print("="*70)
    
    # Шаг 1: Генерация датасета (50 моделей для быстрого старта)
    print("\n🔄 Шаг 1: Генерация синтетического датасета (50 моделей)...")
    dataset_path = generate_enhanced_dataset("enhanced_dataset", n_samples=50)
    
    # Шаг 2: Подготовка датасета для обучения
    print("\n🔄 Шаг 2: Подготовка датасета для обучения...")
    dataset = FixedSyntheticDataset(root="enhanced_dataset")
    
    # Проверка разметки
    test_data = dataset[0]
    face_mask = (test_data.node_type == 1)
    role3_count = (test_data.y[face_mask] == 3).sum().item()
    print(f"✅ Проверка разметки: найдено опорных плоскостей (роль 3) в первой модели: {role3_count}/14")
    
    if role3_count < 3:
        print("❌ КРИТИЧЕСКАЯ ОШИБКА: В разметке нет опорных плоскостей!")
        print("   Проверьте функцию create_synthetic_part — должны быть реализованы все 4 типа деталей")
        sys.exit(1)
    
    # Шаг 3: Обучение модели
    print("\n🔄 Шаг 3: Обучение GNN на синтетических данных...")
    
    # Разделение на тренировочную/валидационную выборки
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    from torch.utils.data import random_split
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # уменьшенный батч для стабильности
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Инициализация модели
    device = torch.device('cpu')
    model = GNNModel(in_channels=3, hidden_dim=64, num_roles=4).to(device)
    
    # Усиленные веса для роли 3 (опорные плоскости)
    class_weights = torch.tensor([0.2, 0.3, 1.0, 10.0], dtype=torch.float, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    print("\nОбучение (10 эпох для быстрого старта)...\n")
    
    for epoch in range(10):  # 10 эпох достаточно для демонстрации
        # Тренировка
        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
            roles_pred, _ = model(data.x, data.edge_index, batch)
            
            face_mask = (data.node_type == 1)
            if face_mask.sum() > 0:
                loss = criterion(roles_pred[face_mask], data.y[face_mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.num_graphs
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # Валидация
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
                roles_pred, _ = model(data.x, data.edge_index, batch)
                
                face_mask = (data.node_type == 1)
                if face_mask.sum() > 0:
                    loss = criterion(roles_pred[face_mask], data.y[face_mask])
                    val_loss += loss.item() * data.num_graphs
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "gnn_best.pth")
            status = "⭐"
        else:
            status = ""
        
        print(f"Эпоха {epoch+1:2d}/10 | Loss: {avg_train_loss:.4f} → {avg_val_loss:.4f} {status}")
    
    print(f"\n✅ Обучение завершено! Модель сохранена в: gnn_best.pth")
    
    # Шаг 4: Тестирование на реальных моделях
    print("\n🔄 Шаг 4: Тестирование обученной модели на реальных моделях...")
    
    # Загрузите ВАШИ реальные модели (замените пути на ваши файлы)
    STEP_FILE_1 = "test.step"
    STEP_FILE_2 = "test1.step"
    
    if not os.path.exists(STEP_FILE_1) or not os.path.exists(STEP_FILE_2):
        print(f"⚠️ Файлы {STEP_FILE_1} или {STEP_FILE_2} не найдены")
        print("   Создаём тестовые модели из датасета для демонстрации...")
        
        # Используем первые две модели из датасета как тестовые
        reader1 = STEPControl.STEPControl_Reader()
        reader1.ReadFile("enhanced_dataset/raw/bracket_000000.step")
        reader1.TransferRoots()
        shape1_test = reader1.OneShape()
        
        reader2 = STEPControl.STEPControl_Reader()
        reader2.ReadFile("enhanced_dataset/raw/flange_000001.step")
        reader2.TransferRoots()
        shape2_test = reader2.OneShape()
        
        # Сохраняем как тестовые файлы
        writer1 = STEPControl_Writer()
        writer1.Transfer(shape1_test, STEPControl_AsIs)
        writer1.Write("test.step")
        
        writer2 = STEPControl_Writer()
        writer2.Transfer(shape2_test, STEPControl_AsIs)
        writer2.Write("test1.step")
        
        STEP_FILE_1 = "test.step"
        STEP_FILE_2 = "test1.step"
        print(f"✅ Созданы тестовые файлы: {STEP_FILE_1}, {STEP_FILE_2}")
    
    # Запуск пайплайна с обученной моделью
    print("\n🚀 Запуск пайплайна совмещения с обученной GNN...")
    results = main_pipeline(
        step_file_1=STEP_FILE_1,
        step_file_2=STEP_FILE_2,
        use_gnn=True,
        model_path="gnn_best.pth"
    )
    
    if results is not None:
        print("\n" + "="*70)
        print("✅ ПАЙПЛАЙН ЗАВЕРШЁН УСПЕШНО!")
        print("="*70)
        print("Результаты:")
        print("  • outputs/results/aligned_model.step — совмещённая модель")
        print("  • outputs/boolean/ — результаты булевых операций")
        print("\n💡 Советы:")
        print("  1. Откройте aligned_model.step в FreeCAD для визуальной проверки совмещения")
        print("  2. Откройте all_differences.step для анализа различий между моделями")
        print("  3. Для промышленного использования обучите модель на 500+ моделях")
    else:
        print("\n❌ Пайплайн завершился с ошибкой")