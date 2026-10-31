"""Run a candidate-set model through the same immutable round-two protocol."""
import json
import sys
from . import round2, setnet
from .research import fingerprint


def main():
    source_hashes = {p:fingerprint(p) for p in [__file__, 'src/tinychess/setnet.py']}
    round2.run = setnet.run
    round2.main()
    name = sys.argv[sys.argv.index('--experiment')+1]
    path = round2.ROOT/'runs'/(name+'.json')
    report = json.loads(path.read_text())
    report['source_hashes'].update(source_hashes)
    path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
