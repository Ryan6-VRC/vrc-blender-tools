import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from avatarprep.core import proportions as P

root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiles")
files = sorted(glob.glob(os.path.join(root, "*.json")))
assert files, "no profiles found"
bad = []
for f in files:
    try:
        e = P.load_edge(f)
        assert e["source_base"] and e["target_base"], "missing base endpoints: " + f
    except Exception as exc:
        bad.append("%s: %s" % (os.path.basename(f), exc))
if bad:
    print("FAIL: " + "; ".join(bad)); sys.exit(1)
print("OK: %d profiles load + carry base endpoints" % len(files))
