"""
Optimized chunked runner with adaptive parameters by dimension.
"""
import numpy as np
import time, sys, json, os
import gan_experiment as g

PROGRESS_FILE = '/home/claude/progress.json'
OUTPUT_XLSX = '/mnt/user-data/outputs/gan_experiment_results.xlsx'

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {'results': [], 'done_keys': []}

def save_progress(state):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(state, f)

def run_chunked(n_repeats=3):
    state = load_progress()
    results = state['results']
    done = set(tuple(k) for k in state['done_keys'])

    combos = [(d,n,s) for d in g.DISTS for n in g.SAMPLE_SIZES for s in g.STRATEGIES]
    total = len(combos) * n_repeats
    cnt = len(done)

    print(f"Resuming from {cnt}/{total} completed runs\n")

    for dist_name, n_samp, strat_name in combos:
        info = g.DISTS[dist_name]
        dim = info['dim']
        rng_ref = np.random.default_rng(999)
        ref = g.sample_dist(dist_name, 5000, rng_ref)

        for rep in range(1, n_repeats+1):
            key = (dist_name, n_samp, strat_name, rep)
            if key in done:
                continue

            cnt += 1
            print(f"[{cnt:3d}/{total}] {info['label']:24s} n={n_samp:3d}  {strat_name:12s}  rep={rep}", end="", flush=True)

            g._PID[0] = 0
            seed = 42 + rep*1000 + n_samp*7 + abs(hash(dist_name))%9999
            np.random.seed(seed)
            real = g.sample_dist(dist_name, n_samp, np.random.default_rng(seed))

            # Adaptive parameters by dimension
            if dim >= 10:
                h, nz = 64, 12
                mi, ev, wu, pat = 3000, 200, 1000, 1000
            else:
                h, nz = 64, 8
                mi, ev, wu, pat = 5000, 200, 1500, 1500

            cls = g.STRATEGIES[strat_name]

            try:
                gan = cls(dim, noise_dim=nz, hidden=h)
                res = g.train(gan, real, ref, max_iter=mi, eval_every=ev,
                             warmup=wu, patience=pat)
                tag = "CONV" if res['converged'] else "MAX "
                print(f"  -> {tag} it={res['iterations']:5d} W={res['wasserstein']:.4f} t={res['time_seconds']:.1f}s")
            except Exception as e:
                print(f"  -> ERR: {e}")
                res = {'converged':False, 'iterations':0, 'time_seconds':0,
                       'wasserstein':float('nan'), 'ks':float('nan'),
                       'mmd':float('nan'), 'energy':float('nan'),
                       'best_wasserstein': float('nan')}

            row = {
                'distribution': info['label'],
                'dim': dim,
                'n_samples': n_samp,
                'strategy': strat_name,
                'repeat': rep,
            }
            for k2,v in res.items():
                row[k2] = v
            results.append(row)
            done.add(key)
            state['results'] = results
            state['done_keys'] = [list(k) for k in done]
            save_progress(state)

    return results


if __name__ == '__main__':
    t0 = time.time()
    results = run_chunked(n_repeats=3)
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    for r in results:
        for k,v in r.items():
            if isinstance(v, float) and (v != v):
                r[k] = np.nan

    g.write_excel(results, OUTPUT_XLSX)
    print("Done!")
