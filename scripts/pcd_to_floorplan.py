import shutil
import argparse
import numpy as np
import os
import sys
from load_assets import *
import matplotlib.pyplot as plt
import pickle
import copy
from floorplan_sota import generate, load_cfg, DEFAULT_CFG

s3dis_labels = get_s3dis_labels()
FURNITURE_CLASSES = [7, 8, 9, 10, 11]
EXCLUDED_CLASSES = [0] + FURNITURE_CLASSES
STRUCTURAL_CLASSES_FOR_FLOORPLAN = [1, 2, 3, 4, 5, 6]

def discover_planes(source_pcd, min_points=200, distance_threshold=0.02, ransac_n=3, num_iterations=1000):
    """Iteratively finds all significant planes in a point cloud using RANSAC."""
    if len(source_pcd.points) < min_points: return []
    plane_models = []
    pcd_to_segment = copy.deepcopy(source_pcd)
    while len(pcd_to_segment.points) > min_points:
        plane_model, inliers = pcd_to_segment.segment_plane(
            distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
        if len(inliers) < min_points: break
        plane_models.append(plane_model)
        pcd_to_segment = pcd_to_segment.select_by_index(inliers, invert=True)
    return plane_models

def refine_with_planes(base_pcd, clutter_pcd, plane_models, distance_threshold=0.02):
    """Refines a category with points from clutter that fit the given plane models."""
    if not plane_models or len(clutter_pcd.points) == 0: return base_pcd, clutter_pcd
    clutter_pts = np.asarray(clutter_pcd.points)
    inlier_indices_set = set()
    for model in plane_models:
        distances = np.abs(np.dot(clutter_pts, model[:3]) + model[3])
        inlier_indices_set.update(np.where(distances < distance_threshold)[0])
    if not inlier_indices_set: return base_pcd, clutter_pcd
    final_indices = list(inlier_indices_set)
    newly_found_pcd = clutter_pcd.select_by_index(final_indices)
    pcd_processed = base_pcd + newly_found_pcd
    remaining_clutter = clutter_pcd.select_by_index(final_indices, invert=True)
    return pcd_processed, remaining_clutter

def find_circle_from_three_points(p1, p2, p3):
    """Calculates the center and radius of a circle defined by three 2D points."""
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    
    if abs(det) < 1.0e-10: 
        return None, None
        
    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return (cx, cy), radius

def discover_cylinders_manual(source_pcd, min_points=200, distance_threshold=0.05, ransac_iterations=1000):
    """
    Iteratively finds vertical cylinders in a point cloud using a manual RANSAC
    by fitting circles in a 2D projection.
    """
    if len(source_pcd.points) < min_points:
        return []

    cylinder_models = []
    remaining_points = np.asarray(source_pcd.points)

    while len(remaining_points) > min_points:
        points_2d = remaining_points[:, [0, 2]]
        
        best_inlier_count = 0
        best_model = None
        best_inlier_indices = None

        for _ in range(ransac_iterations):
            sample_indices = np.random.choice(len(points_2d), 3, replace=False)
            sample = points_2d[sample_indices]
            
            center, radius = find_circle_from_three_points(sample[0], sample[1], sample[2])
            if center is None:
                continue

            distances_from_center = np.linalg.norm(points_2d - center, axis=1)
            errors = np.abs(distances_from_center - radius)
            inlier_indices = np.where(errors < distance_threshold)[0]
            current_inlier_count = len(inlier_indices)
            
            if current_inlier_count > best_inlier_count:
                best_inlier_count = current_inlier_count
                best_model = [np.array([center[0], 0, center[1]]), np.array([0, 1, 0]), radius]
                best_inlier_indices = inlier_indices
        
        if best_inlier_count > min_points:
            cylinder_models.append(best_model)
            mask = np.ones(len(remaining_points), dtype=bool)
            mask[best_inlier_indices] = False
            remaining_points = remaining_points[mask]
        else:
            break
    return cylinder_models

def refine_with_cylinders(base_pcd, clutter_pcd, cylinder_models, distance_threshold=0.05):
    """Refines a category with points from clutter that fit the given cylinder models."""
    if not cylinder_models or clutter_pcd.is_empty():
        return base_pcd, clutter_pcd
        
    clutter_pts = np.asarray(clutter_pcd.points)
    inlier_indices_set = set()

    for center, axis, radius in cylinder_models:
        point_on_axis = center
        axis_direction = axis
        
        vec_p_c = clutter_pts - point_on_axis
        dist_to_axis = np.linalg.norm(np.cross(vec_p_c, axis_direction), axis=1)
        
        inlier_indices_set.update(np.where(np.abs(dist_to_axis - radius) < distance_threshold)[0])

    if not inlier_indices_set:
        return base_pcd, clutter_pcd

    final_indices = list(inlier_indices_set)
    newly_found_pcd = clutter_pcd.select_by_index(final_indices)
    pcd_processed = base_pcd + newly_found_pcd
    remaining_clutter = clutter_pcd.select_by_index(final_indices, invert=True)
    
    return pcd_processed, remaining_clutter

def save_floor_plan(pc_names, pcs, pipeline_r, vis_open3d, vizualize_prediction, refine_cylinders=False):
    vis_points = []
    for i, data in enumerate(pcs):
        name = pc_names[i]
        #results_r = pipeline_r.run_inference(data)
        #print(type(results_r))
        #with open("results.pkl", "wb") as f:
        #    pickle.dump(results_r, f) 
        # FOR DEBUGGING LOADING PRESAVED PICKLE FILE
        with open("results.pkl", "rb") as f:
            results_r = pickle.load(f)
        pts, features, original_labels = data['point'], data['feat'], data['label']
        pred_label_r = (results_r['predict_labels'] + 1).astype(np.int32)
        pred_label_r[0] = 0
        vis_d = {
            "name": name,
            "points": pts,
            "labels": original_labels,
            "features": features,
            'pred': pred_label_r,
        }
        vis_points.append(vis_d)

        pred_labels = results_r['predict_labels'].astype(np.int32)

        print("--- Stage 1: Initial Semantic Segregation ---")
        initial_pcds = {}
        s3dis_labels = {
                0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'beam', 4: 'column',
                5: 'window', 6: 'door', 7: 'table', 8: 'chair', 9: 'sofa',
                10: 'bookcase', 11: 'board', 12: 'clutter'
            }
        for label_id, class_name in s3dis_labels.items():
            mask = (pred_labels == label_id)
            if np.any(mask):
                pcd, _, _ = open3d_pcd(pts[mask], features[mask])
                initial_pcds[class_name] = pcd
        
        print("\n--- Creating the scene based on RAW SEMANTICS ONLY ---")
            
        pcd_before_refinement = o3d.geometry.PointCloud()
            
        for class_id in STRUCTURAL_CLASSES_FOR_FLOORPLAN:
            class_name = s3dis_labels.get(class_id)
            if class_name in initial_pcds:
                pcd_before_refinement += initial_pcds[class_name]

        print("\n--- Stage 2: Isolating Unwanted Classes and Defining Candidate Pool ---")
        pcd_excluded = o3d.geometry.PointCloud()
        for class_id in EXCLUDED_CLASSES:
            class_name = s3dis_labels.get(class_id)
            if class_name in initial_pcds:
                pcd_excluded += initial_pcds[class_name]
        print(f"Isolated {len(pcd_excluded.points)} ceiling and furniture points to be ignored.")

        pcd_unassigned_clutter = initial_pcds.get("clutter", o3d.geometry.PointCloud())
        print(f"Starting with {len(pcd_unassigned_clutter.points)} points in the unassigned clutter pool.")

        print("\n--- Stage 3: Iterative Geometric Refinement ---")
        final_pcds = {s3dis_labels[k]: initial_pcds.get(s3dis_labels[k], o3d.geometry.PointCloud()) 
                      for k in STRUCTURAL_CLASSES_FOR_FLOORPLAN}
        PLANAR_CLASSES = [1, 2, 5, 6]  # floor, wall, window, door
        CYLINDRICAL_CLASSES = [4, 3] # column, beam
        PROCESSING_ORDER = PLANAR_CLASSES
        if refine_cylinders:
            print("Cylinder refinement is ENABLED by user.")
            PROCESSING_ORDER += CYLINDRICAL_CLASSES
        else:
            print("Cylinder refinement is DISABLED. Skipping columns and beams.")
        for class_id in PROCESSING_ORDER:
            class_name = s3dis_labels.get(class_id)
            if class_name in initial_pcds:
                print(f"\n--- Refining {class_name.upper()} ---")
                base_pcd = initial_pcds[class_name]
                downsampled_pcd = base_pcd.voxel_down_sample(voxel_size=0.05)
                if class_id in PLANAR_CLASSES:
                    models = discover_planes(downsampled_pcd)
                    print(f"Discovered {len(models)} plane model(s).")
                    if models:
                        final_pcds[class_name], pcd_unassigned_clutter = refine_with_planes(
                            final_pcds[class_name], pcd_unassigned_clutter, models)
                        print(f"Refined {class_name}. Clutter points remaining: {len(pcd_unassigned_clutter.points)}")
                
                elif class_id in CYLINDRICAL_CLASSES:
                    models = discover_cylinders_manual(downsampled_pcd)
                    print(f"Discovered {len(models)} Manual RANSAC cylinder model(s).")
                    if models:
                        final_pcds[class_name], pcd_unassigned_clutter = refine_with_cylinders(
                            final_pcds[class_name], pcd_unassigned_clutter, models)
                        print(f"Refined {class_name}. Clutter points remaining: {len(pcd_unassigned_clutter.points)}")
        print("\n--- Stage 4: Assembling Final Point Cloud for Floor Plan ---")
        pcd_for_floorplan = o3d.geometry.PointCloud()
        for class_id in STRUCTURAL_CLASSES_FOR_FLOORPLAN:
            class_name = s3dis_labels.get(class_id)
            if class_name in final_pcds:
                pcd_for_floorplan += final_pcds[class_name]
        print(f"Final point cloud for floor plan has {len(pcd_for_floorplan.points)} points.")

        if vis_open3d:
            print("\n--- Stage 5: Visualizing Final Scene Breakdown ---")
            if not pcd_before_refinement.is_empty():
                print(f"Showing raw structural pcd with {len(pcd_before_refinement.points)} points.")
                o3d.visualization.draw_geometries(
                    [pcd_before_refinement],
                    window_name="BEFORE Refinement (Raw Semantics)"
                )
            o3d.visualization.draw_geometries(
                [pcd_for_floorplan],
                window_name="Final Breakdown processed"
            )
        print("\n--- Final Stage: Running the PCD to Floorplan Pipeline ---")
        empty = o3d.geometry.PointCloud()
        clouds = {"floor": final_pcds.get("floor", empty), "ceiling": initial_pcds.get("ceiling", empty),
                  "walls": final_pcds.get("wall", empty), "window": final_pcds.get("window", empty),
                  "door": final_pcds.get("door", empty), "column": final_pcds.get("column", empty)}
        generate(clouds, output_path, load_cfg(DEFAULT_CFG))
        print("DONE! Saved the floor plan results!")
        print("="*20)
    if visualize_prediction==True:
        print("Vizualizing segmentation results with Open3d ML....")
        ml3d_visualizer(pcs_with_pred=vis_points)

def main(data_path, out_path, visualize_prediction=False, vis_open3d=False, refine_cylinders=False):
    os.makedirs(out_path, exist_ok = True)
    pipeline = get_pipeline()
    pcs = get_custom_data(data_path)
    print("Extracting floor plan...")
    pcs_with_pred = save_floor_plan(["input_indoor_scene"], [pcs], pipeline, vis_open3d, visualize_prediction,refine_cylinders)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" 3D semantic segmentation of point clouds and processing the segmentation result for floor plan")
    parser.add_argument("--data_path", help="The path to input PLY mesh file")
    parser.add_argument("--refine_cylinders", action='store_true', help="Enable RANSAC cylinder fitting for columns and beams. DEFAULT = FALSE")
    parser.add_argument("--vis_prediction", action='store_true', help="Option to visualize semantic segmentation result. DEFAULT = FALSE")
    parser.add_argument("--vis_open3d", action='store_true',help="Option to visualize floor plan in 3D from open3d. DEFAULT = FALSE")
    parser.add_argument("--output_dir", help="The path to store floor plan images")
    args = parser.parse_args()

    input_path = args.data_path
    output_path = args.output_dir
    visualize_prediction = args.vis_prediction
    vis_open3d =  args.vis_open3d
    refine_cylinders_enabled = args.refine_cylinders

    if not args.data_path or not args.output_dir:
        parser.print_help(sys.stderr)
        sys.exit(1)

    main(input_path, output_path, visualize_prediction, vis_open3d, refine_cylinders_enabled)