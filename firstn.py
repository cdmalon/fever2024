import json
import sys

all = json.loads(open(sys.argv[1], "r", encoding="utf8").read())
subset = all[:int(sys.argv[2])]
print(json.dumps(subset, indent=4))

