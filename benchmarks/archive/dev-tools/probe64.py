import json
d = json.load(open('data/cost_scaling_oscillator_bath.json'))
p = [q for q in d['points'] if q['dim']==64][0]
print('ref_method:', p.get('reference_method'))
print('t_native_ref:', p.get('t_native_ref'))
print('selfcheck:', p.get('native_ref_selfcheck'))
print('sweep rows:')
for e in p['m_sweep'] or []:
    print('  ', {k: e.get(k) for k in ('M','rmse','diverged')})
