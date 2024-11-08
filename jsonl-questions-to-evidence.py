import sys
import json

in_fp = open(sys.argv[1], "r", encoding="utf8")

all = []
for line in in_fp.readlines():
  example = json.loads(line)
  example["evidence"] = example["questions"]
  all.append(example)

print(json.dumps(all, indent=4))

