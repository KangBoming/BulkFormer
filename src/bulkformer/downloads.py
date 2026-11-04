"""Download complete model resources from their pinned Google Drive release."""
import json
from pathlib import Path
from .inference import sha256, verify_bundle


def download_model(name, model_dir=None):
    if name not in {'atlas', 'latest'}:
        raise ValueError('Choose atlas or latest')
    import gdown
    directory = Path(model_dir or f'model/{name}')
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(__file__).parent / 'manifests' / f'{name}.json'
    release = json.loads(manifest_path.read_text())
    for filename, info in release['downloads'].items():
        path = directory / filename
        if path.exists() and sha256(path) == info['sha256']:
            continue
        temporary = path.with_suffix(path.suffix + '.partial')
        gdown.download(id=info['google_drive_id'], output=str(temporary),
                       quiet=False, resume=True, timeout=(30, 120), retries=2)
        if (not temporary.exists() or temporary.stat().st_size != info['bytes']
                or sha256(temporary) != info['sha256']):
            temporary.unlink(missing_ok=True)
            raise ValueError(f'Download verification failed: {filename}')
        temporary.replace(path)
    verify_bundle(directory, name)
    return directory
