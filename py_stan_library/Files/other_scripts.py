# Running External Programs
# Say you want to run ls and capture all the output

import subprocess
# we can spawn other processes

completed = subprocess.run(["python3.7", "Classes.py"],
                           capture_output=True,
                           text=True,
                           check=True)

print(completed.stdout)
