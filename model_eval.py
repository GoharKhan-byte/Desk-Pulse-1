# import os
# from ultralytics import YOLO

# # 1. Load trained model weights
# model = YOLO("best.pt")

# # 2. Get absolute path to data.yaml configuration
# yaml_path = os.path.abspath("data.yaml")

# # 3. Run validation on CPU
# metrics = model.val(
#     data=yaml_path,
#     split="val",
#     plots=False,
#     imgsz=640,
#     batch=16,
#     conf=0.25,
#     iou=0.6,
#     device="cpu"
# )

# # 4. Overall Performance Summary
# print("\n" + "=" * 50)
# print("       OVERALL MODEL EVALUATION METRICS       ")
# print("=" * 50)
# print(f"Precision (P)    : {metrics.results_dict.get('metrics/precision(B)', 0):.4f}")
# print(f"Recall (R)       : {metrics.results_dict.get('metrics/recall(B)', 0):.4f}")
# print(f"mAP@50           : {metrics.box.map50:.4f}")
# print(f"mAP@50-95        : {metrics.box.map:.4f}")
# print("=" * 50)

# # 5. Built-in Detailed Breakdown
# # Prints class instance counts, Precision, Recall, mAP50, and mAP50-95 cleanly
# metrics.summary()



import os
from ultralytics import YOLO

# 1. Load trained model weights
model = YOLO("best.pt")

# Display embedded model classes (helps spot data.yaml mismatches immediately)
print("\n" + "=" * 50)
print("EMBEDDED MODEL CLASSES")
print("=" * 50)
print(f"Total Classes (nc): {len(model.names)}")
print(f"Class Map         : {model.names}")
print("=" * 50 + "\n")

# 2. Get absolute path to data.yaml configuration
yaml_path = os.path.abspath("data.yaml")

# 3. Run validation on CPU
metrics = model.val(
    data=yaml_path,
    split="val",
    plots=False,      # Prevents confusion matrix index crashes if dataset class counts mismatch
    imgsz=640,
    batch=16,
    conf=0.25,
    iou=0.6,
    device="cpu"
)

# 4. Overall Performance Summary
print("\n" + "=" * 50)
print("       OVERALL MODEL EVALUATION METRICS       ")
print("=" * 50)
print(f"Mean Precision (P)    : {metrics.results_dict.get('metrics/precision(B)', 0):.4f}")
print(f"Mean Recall (R)       : {metrics.results_dict.get('metrics/recall(B)', 0):.4f}")
print(f"mAP@50                : {metrics.box.map50:.4f}")
print(f"mAP@50-95             : {metrics.box.map:.4f}")
print("=" * 50)

# 5. Robust Per-Class Metric Breakdown
print("\n" + "=" * 50)
print("          PER-CLASS METRIC BREAKDOWN          ")
print("=" * 50)

# ap_class_index maps array indices to the actual evaluated class IDs
evaluated_ids = getattr(metrics, "ap_class_index", [])

for class_id, class_name in model.names.items():
    if class_id in evaluated_ids:
        # Match class_id to internal result array index
        idx = list(evaluated_ids).index(class_id)
        
        # Extract individual class mAP scores safely
        map50_95 = metrics.box.maps[idx] if idx < len(metrics.box.maps) else 0.0
        
        print(f"Class {class_id} ({class_name:<22}): mAP50-95 = {map50_95:.4f}")
    else:
        print(f"Class {class_id} ({class_name:<22}): N/A (No targets in validation labels)")

print("=" * 50)
