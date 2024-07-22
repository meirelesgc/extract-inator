import subprocess
import os
from sys import argv

for index, doc in enumerate(os.listdir("documents/graph-documents")):
    if doc.endswith(".pdf"):
        subprocess.run(args=["python", "extract-inator/scripts/graph-plot.py", doc])
