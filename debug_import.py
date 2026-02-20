import importlib, sys, traceback
sys.path.insert(0, r'c:/Users/SISTEMAS/Documents/epp_streamlit_app')
try:
    m = importlib.import_module('core.id_reader')
    print('OK', [n for n in dir(m) if not n.startswith('_')])
except Exception as e:
    traceback.print_exc()
    print('FAILED:', e)
