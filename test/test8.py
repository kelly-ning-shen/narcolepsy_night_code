from pathlib import Path

i = 0
xml_path = 'data/mnc/cnc/chc/chc001-nsrr.xml'
s = xml_path.split('.')[0]
s = Path(f'{s}_15min_{i}.pkl')
# path_str = Path('/usr/hello/demo.py')
# base = path_str.parent
name = s.stem
n = name.split('_')[0]
print(type(s))
print(s)
print(s.exists())