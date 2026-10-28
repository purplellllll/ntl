"""Restore a downloaded NTL release and verify all archival file hashes."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path

def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def extract(archive,dest):
    with tarfile.open(archive,'r:gz') as tar:
        for item in tar.getmembers():
            assert (dest/item.name).resolve().is_relative_to(dest.resolve()),'Unsafe archive path'
            assert item.isfile() or item.isdir(),'Links/devices are not allowed'
        tar.extractall(dest,filter='data')

def assemble(parts,target,expected):
    target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists(),'Do not overwrite an existing backbone'
    temp=target.with_suffix('.partial')
    with temp.open('wb') as out:
        for part in parts:
            with part.open('rb') as inp: shutil.copyfileobj(inp,out,8*1024**2)
    assert temp.stat().st_size==expected['bytes'] and digest(temp)==expected['sha256'],'Backbone hash mismatch'
    temp.replace(target)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--assets-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.assets_dir/'RELEASE_MANIFEST.json').read_text())
    for name,info in manifest['assets'].items():
        assert Path(name).name==name
        path=args.assets_dir/name
        assert path.stat().st_size==info['bytes'] and digest(path)==info['sha256'],name
    args.output.mkdir(parents=True,exist_ok=False)
    extract(args.assets_dir/manifest['checkpoint_archive'],args.output)
    extract(args.assets_dir/manifest['materials_archive'],args.output)
    assemble([args.assets_dir/name for name in manifest['backbone_parts']],
             args.output/'base/qwen3-8b-base-nf4/model.safetensors',manifest['backbone'])
    for name in [manifest['checkpoint_archive'],manifest['checkpoint_archive']+'.sha256']:
        shutil.copy2(args.assets_dir/name,args.output/name)
    inventory=json.loads((args.output/'SERVER_INVENTORY.json').read_text())
    for name,info in inventory['files'].items():
        assert (args.output/name).stat().st_size==info['bytes'] and digest(args.output/name)==info['sha256'],name
    print('Restored and verified:',args.output)

if __name__=='__main__': main()
