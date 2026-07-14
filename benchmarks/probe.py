import json
d = json.load(open('data/cost_scaling_oscillator_bath.json'))
p = [q for q in d['points'] if q['dim'] == 64][0]
print('ref:', p['reference_method'], ' t_native:', p['t_native_ref'])
print('validation:', d['native_ref_validation'])
for r in p['m_sweep']:
    print('M=%4d  RMSE=%.3e (+/-%.1e)  mse=%.3e  sem2=%.3e'
          % (r['M'], r['rmse'], r['rmse_std'], r['mse'], r['sem_sq']))
