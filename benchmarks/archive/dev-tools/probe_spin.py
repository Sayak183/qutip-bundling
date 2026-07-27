import json
d = json.load(open('data/cost_scaling_spin_chain.json'))
for p in d['points']:
    sc = p.get('native_ref_selfcheck')
    dim = p['dim']
    method = p.get('reference_method')
    t = p.get('t_native_ref')
    verdict = None if not sc else (sc.get('passed'), format(sc.get('max_abs_dev'), '.1e'))
    print('dim', dim, '| method:', method, '| t_native:', t, '| selfcheck:', verdict)
