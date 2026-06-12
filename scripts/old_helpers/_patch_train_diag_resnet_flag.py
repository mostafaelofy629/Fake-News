import nbformat
from pathlib import Path

nb_path = Path("03_Multimodal_Model_Training_FINAL_Enhanced.ipynb")
nb = nbformat.read(nb_path, as_version=4)

patched = False
for cell in nb.cells:
    if cell.cell_type == "code" and "FINAL DIAGNOSTIC: evaluate saved models on the training split" in cell.source:
        if "TRAIN_EVAL_INCLUDE_RESNET50" not in cell.source:
            cell.source = cell.source.replace(
                "TRAIN_WRONG_THRESHOLD = 6\n",
                "TRAIN_WRONG_THRESHOLD = 6\nTRAIN_EVAL_INCLUDE_RESNET50 = True  # Set False for a faster diagnostic that skips the 3 ResNet50 train passes.\n",
            )
        if "if not TRAIN_EVAL_INCLUDE_RESNET50" not in cell.source:
            cell.source = cell.source.replace(
                "    for weight_path in deep_weight_files:\n",
                "    if not TRAIN_EVAL_INCLUDE_RESNET50:\n"
                "        deep_weight_files = [p for p in deep_weight_files if not p.name.startswith('frozen_resnet50')]\n"
                "        print('TRAIN_EVAL_INCLUDE_RESNET50=False, skipping frozen ResNet50 train diagnostics.')\n"
                "    for weight_path in deep_weight_files:\n",
            )
        cell.source = cell.source.replace(
            "        print('Evaluated:', model_name)\n",
            "        print('Evaluated:', model_name, flush=True)\n",
        )
        patched = True
        break

if not patched:
    raise SystemExit("Training diagnostic cell not found.")

nbformat.write(nb, nb_path)
print("Patched training diagnostic ResNet50 flag.")
