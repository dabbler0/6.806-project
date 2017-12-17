qdict = {}
with open('Android/corpus.tsv') as f:
    for line in f:
        qid, title, body = line.split('\t')
        qdict[int(qid)] = title

left_side = [
    101399,
    139879,
    34926,
    55917,
    149539,
    61328,
    25938,
    49852,
    49248
]

right_side = [
    169664,
    3437,
    57640,
    69331,
    103047,
    113599,
    27417,
    114405,
    139653
]

for a, b in zip(left_side, right_side):
    print(qdict[a].split(' ')[0], qdict[b].split(' ')[0])
