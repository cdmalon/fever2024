import json
import sys

all = json.loads(open(sys.argv[1], "r", encoding="utf8").read())

for example in all:
  evidence = example["evidence"]
  n = len(evidence)
  if(n > 0):
    for i in range(n, 10):
      j = (i-n) % n
      evidence.append(evidence[j])

print(json.dumps(all, indent=4))

