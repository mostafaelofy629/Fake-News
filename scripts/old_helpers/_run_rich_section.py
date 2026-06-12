"""
Run only the final rich multimodal section from Notebook 03.

This executes the lightweight setup cells plus:
OCR + BLIP + BERT + ResNet50 enriched fusion, then refreshes the report table.

Notes:
- The first full run is slow because it extracts OCR, BLIP captions, BERT embeddings,
  and ResNet50 features.
- Later runs are much faster because the rich features are cached in final_artifacts/.
- To force a full rebuild, edit the notebook rich cell and set:
  RICH_FORCE_REBUILD_FEATURES = True
"""

import contextlib
import nbformat
import traceback
from pathlib import Path
from IPython.display import display

nb_path = Path("03_Multimodal_Model_Training_FINAL_Enhanced.ipynb")
log_path = Path("final_artifacts") / "rich_multimodal_run.log"
log_path.parent.mkdir(exist_ok=True)

nb = nbformat.read(nb_path, as_version=4)

required_markers = [
    "# FINAL UPGRADE 03-M1",
    "# FINAL UPGRADE 03-M2",
    "# FINAL UPGRADE 03-M3",
    "# FINAL UPGRADE 03-M4",
    "# FINAL UPGRADE 03-M7",
    "# FINAL UPGRADE 03-M6",
]

indices = []
for marker in required_markers:
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and marker in cell.source:
            indices.append(i)
            break
    else:
        raise RuntimeError(f"Could not find notebook cell with marker: {marker}")

ns = {"display": display, "__name__": "__main__"}

with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
    for idx in indices:
        cell = nb.cells[idx]
        title = cell.source.splitlines()[0] if cell.source.splitlines() else ""
        print(f"\n===== RUNNING CELL {idx}: {title} =====", flush=True)
        try:
            exec(compile(cell.source, f"{nb_path.name}:cell-{idx}", "exec"), ns)
            print(f"===== FINISHED CELL {idx} =====", flush=True)
        except Exception as exc:
            print(f"!!!!! FAILED CELL {idx}: {exc}", flush=True)
            traceback.print_exc()
            raise

print(f"Rich multimodal run finished. Log: {log_path}")
