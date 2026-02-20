import os
import json
import numpy as np
import torch
from torch_geometric.data import Dataset, Data
from OCC.Core.STEPControl import STEPControl_Reader

class FixedSyntheticDataset(Dataset):
    """
    Исправленный датасет с правильным сопоставлением аннотаций в исходных координатах (мм)
    """
    def __init__(self, root="synthetic_dataset", transform=None, pre_transform=None):
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
            reader = STEPControl_Reader()
            reader.ReadFile(os.path.join(self.raw_dir, step_file))
            reader.TransferRoots()
            shape = reader.OneShape()
            
            # 2. Извлечение топологии → получаем ИСХОДНЫЕ координаты в мм
            vertices, face_vertex_indices = extract_topology(shape)
            # vertices сейчас в миллиметрах, например: [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], ...]
            
            # 3. Загрузка аннотаций
            ann_file = os.path.splitext(step_file)[0] + ".json"
            ann_path = os.path.join(self.ann_dir, ann_file)
            annotations = {}
            if os.path.exists(ann_path):
                with open(ann_path, "r") as f:
                    annotations = json.load(f)
            
            # 4. СОПОСТАВЛЕНИЕ В ИСХОДНЫХ КООРДИНАТАХ (мм) — КЛЮЧЕВОЙ МОМЕНТ!
            n_vertices = len(vertices)
            n_faces = len(face_vertex_indices)
            node_roles = np.zeros(n_vertices + n_faces, dtype=np.int64)
            node_roles[:n_vertices] = self.role_mapping["decorative"]  # вершины → декоративные
            
            # Вычисляем центроиды граней в ИСХОДНЫХ координатах (мм)
            face_centers_mm = []
            for vtx_ids in face_vertex_indices:
                if vtx_ids:
                    center = vertices[vtx_ids].mean(axis=0)  # vertices — в мм!
                    face_centers_mm.append(center)
            face_centers_mm = np.array(face_centers_mm)
            
            # Отладочная информация
            if idx == 0:
                print(f"\n🔍 Отладка для {step_file}:")
                print(f"   Всего граней: {n_faces}")
                print(f"   Центроиды граней (первые 3): {face_centers_mm[:3]}")
                print(f"   Аннотации из JSON: {annotations.get('reference_planes', [])}")
            
            # Сопоставление аннотаций с гранями по близости в мм
            assigned = np.zeros(n_faces, dtype=bool)
            
            # Опорные плоскости (роль 3) — приоритет
            ref_planes_found = 0
            for ref_plane in annotations.get("reference_planes", []):
                ref_center = np.array(ref_plane["center"])  # тоже в мм!
                if len(face_centers_mm) > 0:
                    distances = np.linalg.norm(face_centers_mm - ref_center, axis=1)
                    closest_idx = np.argmin(distances)
                    
                    # Отладка
                    if idx == 0:
                        print(f"   Аннотация центр: {ref_center}, ближайшая грань: {closest_idx}, расстояние: {distances[closest_idx]:.2f} мм")
                    
                    if distances[closest_idx] < 50.0:  # Увеличенный порог до 50 мм для надёжности
                        node_roles[n_vertices + closest_idx] = self.role_mapping["reference_plane"]
                        assigned[closest_idx] = True
                        ref_planes_found += 1
            
            if idx == 0:
                print(f"   Найдено опорных плоскостей: {ref_planes_found}")
            
            # Крепёжные элементы (роль 2)
            for fastening in annotations.get("fastening_elements", []):
                fast_center = np.array(fastening["center"])
                if len(face_centers_mm) > 0:
                    distances = np.linalg.norm(face_centers_mm - fast_center, axis=1)
                    closest_idx = np.argmin(distances)
                    if distances[closest_idx] < 20.0 and not assigned[closest_idx]:  # порог 20 мм
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
            
            # 5. Теперь строим граф С НОРМАЛИЗАЦИЕЙ (как в оригинальном коде)
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