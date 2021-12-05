from pathlib import Path
from z3 import *


def smts():

    base_path = Path(__file__).parent
    base_path = (base_path / "../benchmark").resolve()

    for root, dirs, files in os.walk(base_path):
        for file in files:
            full_file_path = os.path.join(root, file)

            assert os.path.isfile(full_file_path)

            yield full_file_path


f = None

i = 0
for smt_path in smts():
    f = parse_smt2_file(smt_path)
    i += 1
    print(i)
