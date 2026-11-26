"""Convert the STSGCN PEMS release (Hugging Face Davidham3/pems-stsgcn, Apache-2.0; the parquet repack of
github.com/Davidham3/STSGCN's .npz) into this repo's format.

  datasets/PEMS/<DS>/<DS>_series.parquet, <DS>_edges.parquet   (downloaded)
  -> datasets/PEMS/<DS>/flow.npy   (T, 1, 1, N) float32: the flow channel, sensors along the width
     datasets/PEMS/<DS>/hops.npy   (N, N) int16: hop distance on the undirected sensor graph (-1 = unreachable)

  curl -L -o datasets/PEMS/PEMS04/PEMS04_series.parquet \\
      https://huggingface.co/datasets/Davidham3/pems-stsgcn/resolve/main/PEMS04/PEMS04_series.parquet
  python -I tools/import_pems.py PEMS04 PEMS08
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trafficpred.data.files import ROOT   # noqa: E402


def hops(n, edges):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    out = np.full((n, n), -1, np.int16)
    for s in range(n):
        out[s, s], frontier, d = 0, [s], 0
        while frontier:
            d += 1
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if out[s, v] < 0:
                        out[s, v] = d
                        nxt.append(v)
            frontier = nxt
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("datasets", nargs="+", help="e.g. PEMS04 PEMS08")
    args = ap.parse_args()
    import pyarrow.parquet as pq
    for ds in args.datasets:
        folder = os.path.join(ROOT, "PEMS", ds)
        table = pq.read_table(os.path.join(folder, f"{ds}_series.parquet"))
        flow = np.asarray(table.column("flow").to_pylist(), dtype=np.float64)        # (T, N)
        if not np.allclose(flow, flow.round()) or flow.min() < 0:
            raise SystemExit(f"{ds}: flow isn't non-negative counts")
        T, N = flow.shape
        np.save(os.path.join(folder, "flow.npy"), flow.astype(np.float32)[:, None, None, :])
        e = pq.read_table(os.path.join(folder, f"{ds}_edges.parquet")).to_pandas()
        h = hops(N, zip(e["from_id"].astype(int), e["to_id"].astype(int)))
        np.save(os.path.join(folder, "hops.npy"), h)
        print(f"{ds}: {T} steps x {N} sensors, flow mean {flow.mean():.1f} max {flow.max():.0f}, "
              f"zeros {np.mean(flow == 0):.2%}; {len(e)} edges, unreachable pairs {np.mean(h < 0):.1%}")


if __name__ == "__main__":
    main()
