"""
Simulation Asset Management System for Documentation

Manages the organization, tracking, and integration of simulation assets
for documentation purposes, including images, videos, code examples, and models.
"""

import os
import shutil
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import re


@dataclass
class AssetInfo:
    """Information about a documentation asset"""
    name: str
    path: str
    size: int
    checksum: str
    type: str  # image, video, model, code, config, etc.
    tags: List[str]
    created_at: str
    updated_at: str
    usage_locations: List[str]  # Pages where this asset is used


class SimulationAssetManager:
    """Manages simulation assets for documentation"""

    def __init__(self, docs_root: str = "site/docs", assets_dir: str = "site/static/assets/simulations"):
        self.docs_root = Path(docs_root)
        self.assets_dir = Path(assets_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        # Asset manifest file
        self.manifest_file = self.assets_dir / "asset_manifest.json"

        # Supported asset types
        self.asset_extensions = {
            'image': {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.bmp', '.webp'},
            'video': {'.mp4', '.webm', '.mov', '.avi', '.mkv'},
            'model': {'.sdf', '.urdf', '.dae', '.stl', '.obj', '.fbx'},
            'code': {'.py', '.cpp', '.c', '.js', '.ts', '.java', '.launch.py', '.xml'},
            'config': {'.yaml', '.yml', '.json', '.toml', '.ini'},
            'document': {'.pdf', '.docx', '.md', '.txt'}
        }

        # Load existing manifest or create new one
        self.assets: Dict[str, AssetInfo] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, AssetInfo]:
        """Load asset manifest from file"""
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                assets = {}
                for name, info_data in data.items():
                    assets[name] = AssetInfo(
                        name=info_data['name'],
                        path=info_data['path'],
                        size=info_data['size'],
                        checksum=info_data['checksum'],
                        type=info_data['type'],
                        tags=info_data['tags'],
                        created_at=info_data['created_at'],
                        updated_at=info_data['updated_at'],
                        usage_locations=info_data['usage_locations']
                    )
                return assets
            except Exception as e:
                print(f"Error loading manifest: {e}")
                return {}
        return {}

    def _save_manifest(self):
        """Save asset manifest to file"""
        data = {}
        for name, asset in self.assets.items():
            data[name] = {
                'name': asset.name,
                'path': asset.path,
                'size': asset.size,
                'checksum': asset.checksum,
                'type': asset.type,
                'tags': asset.tags,
                'created_at': asset.created_at,
                'updated_at': asset.updated_at,
                'usage_locations': asset.usage_locations
            }

        with open(self.manifest_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _get_asset_type(self, file_path: Path) -> str:
        """Determine asset type based on file extension"""
        ext = file_path.suffix.lower()
        for asset_type, extensions in self.asset_extensions.items():
            if ext in extensions:
                return asset_type
        return 'unknown'

    def add_asset(self, source_path: str, destination_name: Optional[str] = None, tags: List[str] = None) -> bool:
        """Add an asset to the management system"""
        source = Path(source_path)
        if not source.exists():
            print(f"Source file does not exist: {source}")
            return False

        # Determine destination name
        if destination_name is None:
            destination_name = source.name
        else:
            # Ensure proper extension is maintained
            source_ext = source.suffix
            if not destination_name.endswith(source_ext):
                destination_name += source_ext

        # Create destination path
        destination = self.assets_dir / destination_name

        try:
            # Copy file to assets directory
            shutil.copy2(source, destination)

            # Create asset info
            checksum = self._calculate_checksum(destination)
            size = destination.stat().st_size
            asset_type = self._get_asset_type(destination)

            asset_info = AssetInfo(
                name=destination_name,
                path=str(destination.relative_to(Path('.'))),
                size=size,
                checksum=checksum,
                type=asset_type,
                tags=tags or [],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                usage_locations=[]
            )

            # Add to assets
            self.assets[destination_name] = asset_info

            # Save manifest
            self._save_manifest()

            print(f"Added asset: {destination_name} ({asset_type})")
            return True

        except Exception as e:
            print(f"Error adding asset {source}: {e}")
            return False

    def remove_asset(self, asset_name: str) -> bool:
        """Remove an asset from the management system"""
        if asset_name not in self.assets:
            print(f"Asset not found: {asset_name}")
            return False

        try:
            asset_path = Path(self.assets[asset_name].path)
            if asset_path.exists():
                asset_path.unlink()

            del self.assets[asset_name]
            self._save_manifest()

            print(f"Removed asset: {asset_name}")
            return True

        except Exception as e:
            print(f"Error removing asset {asset_name}: {e}")
            return False

    def get_asset(self, asset_name: str) -> Optional[AssetInfo]:
        """Get information about an asset"""
        return self.assets.get(asset_name)

    def find_assets_by_type(self, asset_type: str) -> List[AssetInfo]:
        """Find all assets of a specific type"""
        return [asset for asset in self.assets.values() if asset.type == asset_type]

    def find_assets_by_tag(self, tag: str) -> List[AssetInfo]:
        """Find all assets with a specific tag"""
        return [asset for asset in self.assets.values() if tag in asset.tags]

    def find_assets_by_pattern(self, pattern: str) -> List[AssetInfo]:
        """Find assets matching a name pattern"""
        regex = re.compile(pattern, re.IGNORECASE)
        return [asset for asset in self.assets.values() if regex.search(asset.name)]

    def scan_simulation_assets(self, simulation_dir: str = "examples/gazebo") -> List[Tuple[str, str]]:
        """Scan for simulation assets in a directory and add them to management"""
        simulation_path = Path(simulation_dir)
        if not simulation_path.exists():
            print(f"Simulation directory does not exist: {simulation_dir}")
            return []

        added_assets = []
        for root, dirs, files in os.walk(simulation_path):
            for file in files:
                file_path = Path(root) / file
                asset_type = self._get_asset_type(file_path)

                # Only add certain types of assets
                if asset_type in ['image', 'model', 'config', 'code']:
                    asset_name = f"{asset_type}_{file}"
                    success = self.add_asset(str(file_path), asset_name, tags=['simulation', asset_type])
                    if success:
                        added_assets.append((asset_name, str(file_path)))

        return added_assets

    def update_asset_usage(self, asset_name: str, usage_location: str):
        """Update the usage locations for an asset"""
        if asset_name in self.assets:
            asset = self.assets[asset_name]
            if usage_location not in asset.usage_locations:
                asset.usage_locations.append(usage_location)
                asset.updated_at = datetime.now().isoformat()
                self._save_manifest()

    def get_asset_usage_report(self) -> Dict[str, List[str]]:
        """Get a report of which assets are used where"""
        usage_report = {}
        for asset_name, asset in self.assets.items():
            if asset.usage_locations:
                usage_report[asset_name] = asset.usage_locations
        return usage_report

    def validate_asset_integrity(self) -> List[str]:
        """Validate that all assets have correct checksums"""
        invalid_assets = []
        for asset_name, asset in self.assets.items():
            asset_path = Path(asset.path)
            if not asset_path.exists():
                invalid_assets.append(f"{asset_name}: File not found")
                continue

            current_checksum = self._calculate_checksum(asset_path)
            if current_checksum != asset.checksum:
                invalid_assets.append(f"{asset_name}: Checksum mismatch")

        return invalid_assets

    def generate_asset_report(self) -> Dict:
        """Generate a comprehensive asset report"""
        report = {
            'total_assets': len(self.assets),
            'assets_by_type': {},
            'assets_by_tag': {},
            'total_size': 0,
            'created_at': datetime.now().isoformat()
        }

        # Count by type
        for asset in self.assets.values():
            asset_type = asset.type
            if asset_type not in report['assets_by_type']:
                report['assets_by_type'][asset_type] = 0
            report['assets_by_type'][asset_type] += 1

            # Count by tag
            for tag in asset.tags:
                if tag not in report['assets_by_tag']:
                    report['assets_by_tag'][tag] = 0
                report['assets_by_tag'][tag] += 1

            # Add to total size
            report['total_size'] += asset.size

        return report

    def cleanup_unused_assets(self) -> List[str]:
        """Remove assets that are not used in any documentation"""
        unused_assets = []
        for asset_name, asset in self.assets.items():
            if not asset.usage_locations:
                unused_assets.append(asset_name)

        # Remove unused assets
        removed_assets = []
        for asset_name in unused_assets:
            if self.remove_asset(asset_name):
                removed_assets.append(asset_name)

        return removed_assets


def main():
    """Example usage of the asset manager"""
    manager = SimulationAssetManager()

    # Example: Add some assets
    print("Asset manager initialized")
    print(f"Current asset count: {len(manager.assets)}")

    # Scan for simulation assets
    print("\nScanning for simulation assets...")
    added = manager.scan_simulation_assets()
    print(f"Added {len(added)} simulation assets")

    # Generate and print report
    print("\nGenerating asset report...")
    report = manager.generate_asset_report    ()
    print(f"Total assets: {report['total_assets']}")
    print(f"Total size: {report['total_size']} bytes")
    print(f"Assets by type: {report['assets_by_type']}")
    print(f"Assets by tag: {report['assets_by_tag']}")

    # Validate asset integrity
    print("\nValidating asset integrity...")
    invalid = manager.validate_asset_integrity()
    if invalid:
        print(f"Found {len(invalid)} invalid assets:")
        for issue in invalid:
            print(f"  - {issue}")
    else:
        print("All assets are valid")

    # Show usage report
    print("\nAsset usage report:")
    usage = manager.get_asset_usage_report()
    if usage:
        for asset, locations in usage.items():
            print(f"  {asset}: {len(locations)} locations")
    else:
        print("  No usage information available")


if __name__ == "__main__":
    main()