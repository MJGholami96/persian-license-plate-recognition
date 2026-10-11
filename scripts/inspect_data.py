from pathlib import Path
from collections import Counter

root = Path("data/plates")

files = [p for p in root.rglob("*") if p.is_file()]
print("Number of all files: ", len(files))
print("Suffix: ", Counter(p.suffix.lower() for p in files))

print("\nFolders: ")
for d in sorted(p for p in root.rglob("*") if p.is_dir()):
    print(" ", d)

print("\nFirst 10 files")
for p in sorted(files)[:10]:
    print(" ", p)

xmls = sorted(p for p in files if p.suffix.lower() == ".xml")
if xmls:
    print("\nContent of the first XML:", xmls[0])
    print(xmls[0].read_text(encoding="utf-8", errors="replace"))