import json
import pandas as pd
from sklearn.metrics import roc_auc_score

gt = pd.read_csv('runs/submission_test/d3_fov/ground_truth.csv').sort_values('position').reset_index(drop=True)
gt['y'] = (gt['class_label'] != 'non-dysplastic').astype(int)

variants = ['baseline', 'a_letterbox_16x9', 'b1_fov_radius_0.7x', 'b2_fov_radius_1.3x',
            'c_off_centre', 'd_square_no_mask', 'e1_rescale_1920x1080', 'e2_rescale_720x576',
            'f1_aspect_4x3', 'f2_aspect_16x9']
rows = []
for v in variants:
    with open(f'runs/submission_test/d3_fov/{v}/output/stacked-neoplastic-lesion-likelihoods.json') as fh:
        probs = json.load(fh)
    stats = json.load(open(f'runs/submission_test/d3_fov/{v}/output/rare26_run_stats.json'))
    auc = roc_auc_score(gt['y'], probs)
    rows.append((v, stats['fallback_frac'] * 100, round(float(auc), 4)))

header = "{:24s} {:>10s} {:>8s}".format("variant", "fallback%", "AUROC")
print(header)
out = [header]
for v, fb, auc in rows:
    flag = "  <-- CANDIDATE" if (fb > 10 or auc < 0.8) else ""
    line = "{:24s} {:9.2f}% {:8.4f}{}".format(v, fb, auc, flag)
    print(line)
    out.append(line)

with open('reports/d3_fov_sweep_table.txt', 'w') as fh:
    fh.write("\n".join(out) + "\n")
