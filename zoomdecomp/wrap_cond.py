r"""Wrap condition codes (a1 ... b3 and bare c / d) in \cond{...} in generated LaTeX tables, so that their
typography is set by one macro. Comments, math and the arguments of reference-like commands are left alone, and text
already inside \cond{...} is skipped, so the pass is idempotent. The table generators call wrap_dir() after writing."""
import re
from pathlib import Path

PROTECT = re.compile(r"""
    (?<!\\)%.*?$                                              # comment
  | \$\$.*?\$\$ | (?<!\\)\$.*?(?<!\\)\$                        # math
  | \\(?:cite[a-z]*|[cC]ref|ref|label|input|includegraphics|url|begin|end|cond|bibliography|bibliographystyle)\*?(?:\[[^\]]*\])*\{[^}]*\}
""", re.X | re.M | re.S)
AB = re.compile(r"(?<![A-Za-z0-9\\_{])([ab][1-3])(?![A-Za-z0-9_}])")
CD = re.compile(r"(?<![A-Za-z0-9\\_{'’.-])([cd])(?![A-Za-z0-9_}'’-])(?!\.[a-z])")


def wrap_text(s, bare_cd=True, report=None):
    """Return s with the codes wrapped. report(context) is called for every bare c / d that is wrapped."""
    out, pos = [], 0
    def text(seg):
        def sub_cd(m):
            if report: report(seg[max(0, m.start() - 45):m.end() + 35].replace("\n", " "))
            return r"\cond{" + m.group(1) + "}"
        if bare_cd: seg = CD.sub(sub_cd, seg)
        return AB.sub(r"\\cond{\1}", seg)
    for m in PROTECT.finditer(s):
        out += [text(s[pos:m.start()]), m.group(0)]; pos = m.end()
    out.append(text(s[pos:]))
    return "".join(out)


def wrap_dir(d, quiet=False):
    """Wrap the codes in every .tex file of a directory of generated tables (idempotent)."""
    n = 0
    for f in sorted(Path(d).glob("*.tex")):
        s = f.read_text(); t = wrap_text(s)
        if t != s: f.write_text(t); n += 1
    if not quiet: print(f"wrap_cond: {n} file(s) updated in {d}")
    return n
