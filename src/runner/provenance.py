import hashlib
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


def source_digest(directory:Path):
    digest=hashlib.sha256()
    for path in sorted(directory.rglob("*.py")):
        digest.update(str(path.relative_to(directory)).encode()+b"\0"+path.read_bytes())
    return digest.hexdigest()


def software_versions():
    result = {}
    for name in ("torch","boltz","botorch","gpytorch","numpy","scipy","biopython","tmtools","ax-platform"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result
