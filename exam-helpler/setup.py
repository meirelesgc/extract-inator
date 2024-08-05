import os
import subprocess

for index, doc in enumerate(os.listdir("documents/TNPGE-documents")):
    if doc.endswith(".pdf"):
        subprocess.run(
            args=["python", "extract-inator/exam-helpler/TNGPE.py", doc])
