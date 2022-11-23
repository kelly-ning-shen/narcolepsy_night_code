xml = 'G:/NSRR/mnc/cnc/test/chc001-nsrr.xml'

def readLabel(file):
    label_dict = {
        'wake': 0,
        'NREM1': 1,
        'NREM2': 2,
        'NREM3': 3,
        'REM': 4
    }
    labels = []
    start = -1
    import xml.etree.ElementTree as ET
    ScoredEvents = ET.parse(file).find('Instances').findall('Instance')
    for event in ScoredEvents:
        # event_type = event.find('EventType').text
        # if event_type != 'Stages|Stages':
        #     continue
        event_concept = event.attrib['class']
        start_time = float(event.find('Start').text)
        duration = float(event.find('Duration').text)
        assert duration % 30 == 0
        N = int(duration // 30) # if not 30s one label
        if event_concept in label_dict.keys():
            lls = [label_dict[event_concept]] * N
        else:
            lls = [-1] * N
        if start == -1:
            start = start_time
        labels.extend(lls)
    return start, labels # start 记录最开始的时间点

if __name__ == '__main__':
    start, labels = readLabel(xml)
    print(start)
    print(labels)