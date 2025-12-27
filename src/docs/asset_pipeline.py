"""
Documentation Asset Pipeline for Simulation Content

Manages the processing and organization of simulation-related assets
for documentation purposes including images, videos, and example files.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
from datetime import datetime


class DocumentationAssetPipeline:
    def __init__(self, source_dir: str = "examples/gazebo", docs_dir: str = "site/docs"):
        self.source_dir = Path(source_dir)
        self.docs_dir = Path(docs_dir)
        self.assets_dir = self.docs_dir / "assets" / "simulations"

        # Create assets directory if it doesn't exist
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        # Define supported asset types
        self.asset_types = {
            'images': ['.png', '.jpg', '.jpeg', '.gif', '.svg'],
            'videos': ['.mp4', '.avi', '.mov', '.webm'],
            'models': ['.sdf', '.urdf', '.dae', '.stl', '.obj'],
            'configs': ['.yaml', '.yml', '.json', '.xml'],
            'code': ['.py', '.cpp', '.c', '.js', '.ts', '.launch.py', '.sdf']
        }

    def scan_assets(self) -> Dict[str, List[Path]]:
        """Scan for all simulation-related assets in the source directory"""
        assets = {asset_type: [] for asset_type in self.asset_types}

        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                file_path = Path(root) / file
                file_ext = file_path.suffix.lower()

                for asset_type, extensions in self.asset_types.items():
                    if file_ext in extensions:
                        assets[asset_type].append(file_path)
                        break

        return assets

    def copy_asset(self, source_path: Path, destination_name: Optional[str] = None) -> Path:
        """Copy an asset to the documentation assets directory"""
        if destination_name is None:
            destination_name = source_path.name

        destination_path = self.assets_dir / destination_name
        shutil.copy2(source_path, destination_path)

        return destination_path

    def copy_assets_by_type(self, asset_type: str, max_count: int = 10) -> List[Path]:
        """Copy assets of a specific type to the documentation directory"""
        all_assets = self.scan_assets()
        assets_to_copy = all_assets.get(asset_type, [])[:max_count]

        copied_paths = []
        for asset_path in assets_to_copy:
            try:
                # Create a unique name to avoid conflicts
                asset_name = f"{asset_type}_{asset_path.name}"
                copied_path = self.copy_asset(asset_path, asset_name)
                copied_paths.append(copied_path)
                print(f"Copied {asset_path} -> {copied_path}")
            except Exception as e:
                print(f"Error copying {asset_path}: {e}")

        return copied_paths

    def generate_asset_manifest(self) -> Dict:
        """Generate a manifest of all assets with metadata"""
        assets = self.scan_assets()
        manifest = {
            'generated_at': datetime.now().isoformat(),
            'total_assets': 0,
            'assets_by_type': {},
            'checksums': {}
        }

        total_count = 0
        for asset_type, asset_list in assets.items():
            type_count = 0
            type_assets = []

            for asset_path in asset_list:
                # Calculate checksum
                checksum = self._calculate_checksum(asset_path)

                asset_info = {
                    'source_path': str(asset_path),
                    'size': asset_path.stat().st_size,
                    'modified': datetime.fromtimestamp(asset_path.stat().st_mtime).isoformat(),
                    'checksum': checksum
                }

                type_assets.append(asset_info)
                manifest['checksums'][str(asset_path)] = checksum
                type_count += 1
                total_count += 1

            manifest['assets_by_type'][asset_type] = type_assets
            manifest['total_assets'] = total_count

        # Save manifest to file
        manifest_path = self.assets_dir / "asset_manifest.json"
        import json
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def organize_assets_by_module(self, module_name: str = "module2") -> Dict[str, List[Path]]:
        """Organize assets by module for easier documentation reference"""
        all_assets = self.scan_assets()
        module_assets_dir = self.assets_dir / module_name
        module_assets_dir.mkdir(exist_ok=True)

        organized_paths = {}
        for asset_type, asset_list in all_assets.items():
            type_dir = module_assets_dir / asset_type
            type_dir.mkdir(exist_ok=True)

            organized_paths[asset_type] = []
            for asset_path in asset_list:
                try:
                    # Copy asset to module-specific directory
                    destination = type_dir / asset_path.name
                    shutil.copy2(asset_path, destination)
                    organized_paths[asset_type].append(destination)
                except Exception as e:
                    print(f"Error organizing {asset_path}: {e}")

        return organized_paths


if __name__ == "__main__":
    # Example usage
    pipeline = DocumentationAssetPipeline()

    print("Scanning assets...")
    assets = pipeline.scan_assets()

    for asset_type, asset_list in assets.items():
        print(f"{asset_type}: {len(asset_list)} files")
        for asset in asset_list[:3]:  # Show first 3 of each type
            print(f"  - {asset}")

    print("\nOrganizing assets by module...")
    organized = pipeline.organize_assets_by_module("module2")

    print("\nGenerating asset manifest...")
    manifest = pipeline.generate_asset_manifest()
    print(f"Manifest generated with {manifest['total_assets']} assets")